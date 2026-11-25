"""Pack the recorded enemy sounds into vox/enemies.json for the game.

The sounds are ElevenLabs Sound Effects takes picked on the enemy sound lab page
(vox/samples/enemies/). Each one is levelled to the same peak; a loop gets its end crossfaded
into its start so it repeats without a click.

    python tools/enemy_pack.py <folder of takes>
"""
import os, sys, json, base64
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vox_render as vr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# game id: (file, is a loop)
PICKS = {'howl': ('howl-1', False), 'roar': ('roar-2', False), 'step': ('step-1', False), 'nest': ('nest-2', True)}
XFADE = 1.5  # seconds of a loop's tail folded over its head


def seamless(s, sr):
    n = int(XFADE * sr)
    head, tail = s[:n], s[-n:]
    k = np.linspace(0, 1, n, dtype='float32')
    # equal-power crossfade: the tail fades out while the head fades in
    joined = tail * np.cos(k * np.pi / 2) + head * np.sin(k * np.pi / 2)
    return np.concatenate([joined, s[n:-n]])


def main(folder):
    import soundfile as sf
    pack = {'clips': {}}
    for gid, (name, loop) in PICKS.items():
        s, sr = sf.read(os.path.join(folder, name + '.mp3'), dtype='float32', always_2d=True)
        s = s.mean(axis=1)
        s = s * (0.7 / max(1e-6, np.abs(s).max()))
        if loop:
            s = seamless(s, sr)
            s, sr = vr.resample(s, sr)
            buf = __import__('io').BytesIO()
            sf.write(buf, s, sr, format='MP3', bitrate_mode='CONSTANT', compression_level=0.75)
            data = buf.getvalue()  # (not vr.encode: that trims silence, which would break the loop)
        else:
            data = vr.encode(s, sr)
        pack['clips'][gid] = base64.b64encode(data).decode()
    path = os.path.join(ROOT, 'vox', 'enemies.json')
    json.dump(pack, open(path, 'w'), separators=(',', ':'))
    print('wrote', path, os.path.getsize(path) // 1024, 'KB,', len(pack['clips']), 'clips')


if __name__ == '__main__':
    main(sys.argv[1])
