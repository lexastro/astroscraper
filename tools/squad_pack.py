"""Pack the unit barks into vox/squad.json for the game.

The lines live in vox/squad-lines.json: per unit class, the ElevenLabs voice id and [category, line]
pairs, where the line carries its Eleven v4 delivery tags. A category ending in a digit is another
take of the same category (select2 is the second 'select'). The recorded clips are named
<class>-<category>.mp3 (or .wav) in the folder given; each becomes clip id '<class>:<category>:<n>'.

    python tools/squad_pack.py <folder of clips>
"""
import os, sys, re, json, base64
sys.path.insert(0, os.path.dirname(__file__))
import vox_render as vr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(folder):
    import soundfile as sf
    lines = json.load(open(os.path.join(ROOT, 'vox', 'squad-lines.json'), encoding='utf-8'))
    pack, missing = {'clips': {}}, []
    for cls, d in lines.items():
        for cat, _ in d['lines'] + d.get('extra', []):
            base = re.sub(r'\d+$', '', cat)
            n = int(cat[len(base):] or 1) - 1
            f = next((os.path.join(folder, f'{cls}-{cat}{ext}') for ext in ('.wav', '.mp3') if os.path.exists(os.path.join(folder, f'{cls}-{cat}{ext}'))), None)
            if not f:
                missing.append(f'{cls}-{cat}')
                continue
            s, sr = sf.read(f, dtype='float32', always_2d=True)
            pack['clips'][f'{cls}:{base}:{n}'] = base64.b64encode(vr.encode(s.mean(axis=1), sr)).decode()
    path = os.path.join(ROOT, 'vox', 'squad.json')
    json.dump(pack, open(path, 'w'), separators=(',', ':'))
    print('wrote', path, os.path.getsize(path) // 1024, 'KB,', len(pack['clips']), 'clips')
    if missing:
        print('missing:', ', '.join(missing))


if __name__ == '__main__':
    main(sys.argv[1])
