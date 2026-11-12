"""Render Vesper's announcer lines to vox/vesper.json with Kokoro (local, Apache-2.0).

The lines are read straight out of datascraper.html (VOX_EV and VOX_TUT), so the game's text
stays the single source. Lines with {dir} are rendered once per compass word.

Setup (once):  pip install kokoro-onnx soundfile
               download kokoro-v1.0.onnx and voices-v1.0.bin from
               https://github.com/thewh1teagle/kokoro-onnx/releases (model-files-v1.0)
Run:           python tools/vox_render.py --models <folder with the two files> [--voice af_heart]
               python tools/vox_render.py --models ... --samples   (a short comparison of voices)
"""
import argparse, base64, io, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'near side']
SPOKEN_DIR = {'northeast': 'north-east', 'northwest': 'north-west', 'southeast': 'south-east', 'southwest': 'south-west'}


def read_lines():
    src = open(os.path.join(ROOT, 'datascraper.html'), encoding='utf-8').read()
    ev = src[src.index('const VOX_EV={'):]
    ev = ev[:ev.index('\n};')]
    clips = {}
    for m in re.finditer(r'^\s*(\w+):\{p:\d,cd:\d+,(.*)\},?$', ev, re.M):
        key, rest = m.group(1), m.group(2)
        for arr, tag in ((r'\bl:(\[.*?\])', ''), (r'\bnd:(\[.*?\])', 'nd')):
            a = re.search(arr, rest)
            if not a:
                continue
            for i, line in enumerate(json.loads(a.group(1))):
                base = f'{key}{":" + tag if tag else ""}:{i}'
                if '{dir}' in line:
                    for d in DIRS:
                        clips[f'{base}:{d}'] = line.replace('{dir}', SPOKEN_DIR.get(d, d))
                else:
                    clips[base] = line
    tut = src[src.index('const VOX_TUT={'):]
    tut = tut[:tut.index('\n};')]
    for m in re.finditer(r'''^\s*(['"])(.*?)\1:"(.*)",?$''', tut, re.M):
        clips['tut:' + m.group(2)] = m.group(3)
    return clips


def to16k(samples, sr):
    """Kokoro renders at 24 kHz; the in-game radio chain cuts above ~6.8 kHz anyway, so 16 kHz
    loses nothing audible and roughly halves the pack. Windowed-sinc low-pass, then resample."""
    import numpy as np
    if sr == 16000:
        return samples, sr
    n = np.arange(-64, 65)
    fc = 7600 / sr
    h = np.sinc(2 * fc * n) * np.hamming(len(n))
    h /= h.sum()
    y = np.convolve(samples, h, mode='same')
    t = np.arange(0, len(y) / sr, 1 / 16000)
    return np.interp(t, np.arange(len(y)) / sr, y).astype('float32'), 16000


def encode(samples, sr):
    import soundfile as sf
    import numpy as np
    samples, sr = to16k(samples, sr)
    loud = np.nonzero(np.abs(samples) > 0.01)[0]  # trim the silence round each line
    if len(loud):
        samples = samples[max(0, loud[0] - sr // 40):loud[-1] + sr // 12]
    buf = io.BytesIO()
    # MP3 decodes in every browser's Web Audio (Ogg Opus doesn't in older Safari); speech at
    # 16 kHz mono at ~20 kbps is small, and the radio chain hides what little it loses
    sf.write(buf, samples, sr, format='MP3', bitrate_mode='CONSTANT', compression_level=0.93)
    return buf.getvalue()


# ---- ElevenLabs (Eleven v4) workflow: a paste-ready script, then pack the downloaded takes ----
# Delivery direction per event, as Eleven v4 audio tags (free-text tags describing the read work
# best; v4 has no Style/Speed sliders or SSML). Urgency follows each event's priority in the game.
EL_TAGS = {
    'online': '[warm, confident]',
    'stop': '[calm, confident, a hint of a smile]',
    'stopFinal': '[hushed, awed, low and steady]',
    'breach': '[satisfied, warm]',
    'unstable': '[alert, focused, brisk]',
    'collapse': '[dry, deadpan]',
    'ready': '[focused, quick]',
    'struck': '[pleased, brisk]',
    'jackpot': '[excited, delighted]',
    'red': '[urgent, rushed, raised voice]',
    'rupture': '[sharp inhale] [urgent, strained]',
    'drained': '[warm, pleased]',
    'wave': '[alert, crisp, mission-control radio call]',
    'wave:nd': '[alert, crisp, mission-control radio call]',
    'greatWave': '[urgent, tense, raised voice]',
    'patrol': '[alert, wry]',
    'colossus': '[urgent, commanding]',
    'quaker': '[urgent, commanding]',
    'nest': '[satisfied, light]',
    'cable': '[concerned, brisk]',
    'lostRelay': '[concerned, brisk]',
    'lostStab': '[urgent, rushed]',
    'turned': '[quiet, regretful]',
    'distress': '[curious, attentive]',
    'hullHalf': '[tense but controlled, dry]',
    'hullLow': '[shouting, urgent, alarmed]',
    'noDat': '[matter-of-fact, light]',
    'cap': '[matter-of-fact, light]',
    'win': '[relieved, warm, proud]',
    'lose': '[quiet, strained, fading] ...',
    'tut': '[warm, friendly, clear, unhurried]',
}


def slug(cid):
    """Clip id → a safe file name (ids hold ':' and spaces): 'wave:0:near side' → 'wave.0.near_side'."""
    return re.sub(r'[^A-Za-z0-9.]+', '_', cid.replace(':', '.')).strip('_')


def el_tags(cid):
    parts = cid.split(':')
    return EL_TAGS.get(':'.join(parts[:2])) or EL_TAGS.get(parts[0], '')


def write_script():
    clips = read_lines()
    groups = {}
    for cid, text in clips.items():
        groups.setdefault(cid.split(':')[0], []).append((cid, text))
    out = ['# Vesper: Eleven v4 script', '',
           f'{len(clips)} lines. Generate each line on its own with model **Eleven v4**, and save the take as the',
           '**file name shown** (mp3 or wav) in one folder. Then pack them into the game with:', '',
           '    python tools/vox_render.py --from-dir <that folder>', '',
           'Tags in [brackets] direct the delivery and aren\'t spoken. Do 2–3 takes of anything that matters and',
           'keep the best. Lower Stability for the urgent lines, higher for the calm ones and the tutorial.', '']
    for key, items in groups.items():
        out += [f'## {key}', '']
        for cid, text in items:
            out += [f'**{slug(cid)}**', '```', f'{el_tags(cid)} {text}'.strip(), '```', '']
    path = os.path.join(ROOT, 'vox', 'elevenlabs-script.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w', encoding='utf-8').write('\n'.join(out))
    print('wrote', path, len(clips), 'lines')


def pack_from_dir(folder):
    """Pack takes named by slug() (any format soundfile reads: mp3, wav, flac, ogg) into vox/vesper.json."""
    import soundfile as sf
    clips = read_lines()
    files = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder)}
    pack, missing = {'voice': 'elevenlabs', 'clips': {}}, []
    for cid in clips:
        f = files.get(slug(cid))
        if not f:
            missing.append(slug(cid))
            continue
        s, sr = sf.read(f, dtype='float32', always_2d=True)
        pack['clips'][cid] = base64.b64encode(encode(s.mean(axis=1), sr)).decode()
    path = os.path.join(ROOT, 'vox', 'vesper.json')
    json.dump(pack, open(path, 'w'), separators=(',', ':'))
    print('wrote', path, os.path.getsize(path) // 1024, 'KB,', len(pack['clips']), 'clips')
    if missing:
        print(len(missing), 'missing (the game speaks these with the browser voice):', ', '.join(missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models')
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--samples', action='store_true')
    ap.add_argument('--script', action='store_true', help='write vox/elevenlabs-script.md')
    ap.add_argument('--from-dir', help='pack ElevenLabs takes from this folder')
    a = ap.parse_args()
    if a.script:
        return write_script()
    if a.from_dir:
        return pack_from_dir(a.from_dir)
    if not a.models:
        ap.error('--models is required for Kokoro rendering')
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(a.models, 'kokoro-v1.0.onnx'), os.path.join(a.models, 'voices-v1.0.bin'))
    if a.samples:
        out = os.path.join(ROOT, 'vox', 'samples')
        os.makedirs(out, exist_ok=True)
        text = "Contacts inbound from the north-east. Hull at half. I'd rather we didn't make a habit of this."
        for v in ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'bf_emma', 'bf_isabella']:
            s, sr = k.create(text, voice=v, speed=a.speed, lang='en-gb' if v.startswith('b') else 'en-us')
            open(os.path.join(out, v + '.mp3'), 'wb').write(encode(s, sr))
            print('sample', v)
        return
    clips = read_lines()
    lang = 'en-gb' if a.voice.startswith('b') else 'en-us'
    pack = {'voice': a.voice, 'clips': {}}
    for n, (cid, text) in enumerate(clips.items()):
        s, sr = k.create(text, voice=a.voice, speed=a.speed, lang=lang)
        pack['clips'][cid] = base64.b64encode(encode(s, sr)).decode()
        print(f'{n + 1}/{len(clips)} {cid}', flush=True)
    os.makedirs(os.path.join(ROOT, 'vox'), exist_ok=True)
    path = os.path.join(ROOT, 'vox', 'vesper.json')
    json.dump(pack, open(path, 'w'), separators=(',', ':'))
    print('wrote', path, os.path.getsize(path) // 1024, 'KB,', len(clips), 'clips')


if __name__ == '__main__':
    sys.exit(main())
