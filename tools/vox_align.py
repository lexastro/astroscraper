"""Cut a long ElevenLabs take into per-line clips, using local Whisper word timings matched to the script.

Setup: pip install faster-whisper soundfile   (downloads the ~150 MB base.en model on first run)
usage: python tools/vox_align.py take.mp3 FIRST COUNT OUTDIR [slug,slug,...]
  FIRST/COUNT: which lines of vox/elevenlabs-all.txt the take holds (0-based); the optional list
  force-accepts lines you have checked by ear that Whisper mis-heard. Then pack OUTDIR with
  python tools/vox_render.py --from-dir OUTDIR
Writes <slug>.wav for every line whose words were all found, in order, with nothing extra
spoken inside it; prints the lines it rejected."""
import sys, os, json, re, difflib
import warnings
warnings.filterwarnings('ignore')
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vox_render as vr

def transcribe(take):
    """Whisper word timings for a take, transcribed in ~20 s pieces cut at pauses (long audio makes it loop)."""
    from faster_whisper import WhisperModel
    m = WhisperModel('base.en', device='cpu', compute_type='int8')
    a, sr = sf.read(take, dtype='float32', always_2d=True)
    a = a.mean(1)
    a16 = np.interp(np.arange(0, len(a) / sr, 1 / 16000), np.arange(len(a)) / sr, a).astype('float32')
    # transcribe in pieces cut at pauses (~20 s each): long audio makes Whisper loop
    hop = 160; rms = np.sqrt(np.convolve(a16 ** 2, np.ones(hop) / hop, 'same'))[::hop]
    quiet = rms < rms.max() * 0.02
    cuts, last = [0], 0
    for i in range(len(quiet)):
        if i * hop - last > 16000 * 18 and quiet[i:i + 30].all():
            cuts.append(i * hop + 15 * hop); last = cuts[-1]
    cuts.append(len(a16))
    out = []
    for c0, c1 in zip(cuts, cuts[1:]):
        segs, _ = m.transcribe(a16[c0:c1], word_timestamps=True, language='en', condition_on_previous_text=False)
        out += [[round(w.start + c0 / 16000, 2), round(w.end + c0 / 16000, 2), w.word.strip()] for s in segs for w in s.words]
    return out


take, first, count, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
force = set(sys.argv[5].split(',')) if len(sys.argv) > 5 else set()  # slugs checked by ear/eye
lines = list(vr.read_lines().items())[first:first + count]
COMPASS = re.compile(r'\b(north|south) (east|west)\b')
def norm(s):
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower().replace("'", ''))
    s = COMPASS.sub(lambda m: m.group(1) + m.group(2), s)
    s = s.replace('near side', 'nearside')
    # Whisper's spelling noise: plurals/possessives, and a few sound-alikes
    return ' '.join({'vane': 'vein', 'vayne': 'vein', 'companies': 'company'}.get(t, t[:-1] if len(t) > 3 and t.endswith('s') else t) for t in s.split())
words = transcribe(take)
asr = []
for st, en, w in words:
    for t in norm(w).split():
        # Whisper splits "south west" / "near side" into two words: join them back
        if asr and (asr[-1][0], t) in {(a, b) for a in ('north', 'south') for b in ('east', 'west')} | {('near', 'side')}:
            p = asr.pop(); t, st = p[0] + t, p[1]
        asr.append((t, st, en))
script = []
for li, (cid, text) in enumerate(lines):
    t = text.replace('{dir}', '')
    for tok in norm(vr.SPOKEN_DIR.get('', '') + t).split():
        script.append((tok, li))
sm = difflib.SequenceMatcher(None, [s[0] for s in script], [a[0] for a in asr], autojunk=False)
match = {}  # script index -> asr index
for blk in sm.get_matching_blocks():
    for k in range(blk.size):
        match[blk.a + k] = blk.b + k
a, sr = sf.read(take, dtype='float32', always_2d=True)
a = a.mean(1)
os.makedirs(out, exist_ok=True)
span = {}
for li in range(len(lines)):
    idx = [i for i, s in enumerate(script) if s[1] == li]
    got = [match[i] for i in idx if i in match]
    cov = len(got) / len(idx)
    if got:
        lo, hi = min(got), max(got)
        extra = (hi - lo + 1) - len(got)  # asr words inside the line that aren't in the script
        span[li] = (asr[lo][1], asr[hi][2], cov, extra)
    else:
        span[li] = (None, None, 0, 0)
ok, bad = 0, []
for li, (cid, text) in enumerate(lines):
    s, e, cov, extra = span[li]
    if s is None or ((cov < 0.8 or extra > 1) and vr.slug(cid) not in force):
        bad.append(f'{vr.slug(cid)} (found {cov:.0%}, extra {extra})')
        continue
    # widen to the neighbours' edges (halfway into the gaps), then trim silence
    ps = span.get(li - 1, (None, None))[1]
    ns = span.get(li + 1, (None,))[0]
    s0 = max(0.0, (ps + s) / 2 if ps is not None and ps < s else s - 0.3)
    e0 = (e + ns) / 2 if ns is not None and ns > e else e + 0.6
    seg = a[int(s0 * sr):int(min(e0, len(a) / sr) * sr)]
    loud = np.nonzero(np.abs(seg) > 0.01)[0]
    if len(loud):
        seg = seg[max(0, loud[0] - sr // 40):loud[-1] + sr // 12]
    sf.write(os.path.join(out, vr.slug(cid) + '.wav'), seg, sr)
    ok += 1
print(f'{os.path.basename(take)}: {ok}/{len(lines)} clean')
for b in bad:
    print('  rejected', b)
