"""Precompute pitch and loudness tracks for the website's clips as SVG path data.

Pitch is Praat's filtered autocorrelation at its standard settings (50-800 Hz), which the Praat
manual recommends for intonation since 2023: the low-pass filter removes most octave errors, and
the wide range needs no tuning per speaker, so creak and wide pitch excursions are both kept.
The few octave errors it still makes are corrected by hand in CORRECTIONS below, each with the
reason, as in a phonetician's workflow; there is no automatic cleanup.

Each example's clips share one time axis (the longest clip) and one pitch scale (semitones
around the example's median F0), so a reference and its error can be compared by eye.
Dimensions whose examples are single clips (expressiveness, prosodic boundaries) share one
pitch band across the whole dimension, each clip measured from its own median, so a monotone
clip looks flatter than an overacted one. The paths, durations and time labels are written
straight into the matching players in docs/index.html.

Clips are decoded with ffmpeg, which trims the MP3 encoder delay and padding the same way the
browser does; Praat's own MP3 reader keeps them, which put the tracks ~37 ms behind the audio.

Needs ffmpeg and Praat 6.4 or later (filtered autocorrelation is not in parselmouth's bundled
Praat). Praat is looked up as $PRAAT, then `praat` on the PATH, then the macOS app.
Run from the repository root:
    uv run --no-project --with praat-parselmouth --with numpy python docs/assets/audio/make_tracks.py
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth

HERE = Path(__file__).parent
INDEX = HERE.parent.parent / 'index.html'
W, H, PAD = 1000, 64, 6
RATE = 44100
PRAAT = os.environ.get('PRAAT') or shutil.which('praat') or '/Applications/Praat.app/Contents/MacOS/Praat'

# Hand corrections: (clip, start s, end s) spans to treat as unvoiced.
CORRECTIONS = [
    # 40 ms fragment at the second harmonic after the creak at the end of "water", isolated by
    # 110 ms of silence; the spectrogram shows the voice still near 100 Hz.
    ('1-100.mp3', 1.50, 1.57),
]

PITCH_SCRIPT = """form Pitch
  sentence wav
  sentence out
endform
Read from file: wav$
To Pitch (filtered autocorrelation): 0.01, 50, 800, 15, "no", 0.03, 0.09, 0.5, 0.055, 0.35, 0.14
n = Get number of frames
writeFile: out$, ""
for i to n
  t = Get time from frame number: i
  f = Get value in frame: i, "Hertz"
  if f = undefined
    f = 0
  endif
  appendFileLine: out$, fixed$(t, 5), " ", fixed$(f, 3)
endfor
"""

html = INDEX.read_text()
SRC = r'src="assets/audio/([^"]+\.mp3)"'
# Pitch-scale groups: one per example, except a dimension made only of single-clip examples
# becomes one group, so its clips can be compared across players.
groups = []
for dim in html.split('<article class="dim"')[1:]:
    examples = [re.findall(SRC, ex) for ex in re.findall(r'<article class="example.*?</article>', dim, re.S)]
    examples = [e for e in examples if e]
    if len(examples) > 1 and all(len(e) == 1 for e in examples):
        groups.append(('section', [e[0] for e in examples]))
    else:
        groups.extend(('example', e) for e in examples)
seen = [n for _, g in groups for n in g]
assert len(seen) == len(set(seen)), 'a clip landed in two groups'
total = len(seen)
work = Path(tempfile.mkdtemp())
(work / 'pitch.praat').write_text(PITCH_SCRIPT)


def analyse(name):
    wav, out = work / f'{name}.wav', work / f'{name}.f0'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(HERE / name), '-ac', '1', '-ar', str(RATE), str(wav)], check=True)
    subprocess.run([PRAAT, '--run', str(work / 'pitch.praat'), str(wav), str(out)], check=True)
    t, f0 = np.loadtxt(out, unpack=True)
    snd = parselmouth.Sound(str(wav))
    inten = snd.to_intensity(time_step=0.01)
    return snd.duration, t, f0, inten.xs(), inten.values[0]


def semitones(name, t, f0, ref_hz):
    st = np.full(f0.shape, np.nan)
    st[f0 > 0] = 12 * np.log2(f0[f0 > 0] / ref_hz)
    for clip, a, b in CORRECTIONS:
        if clip == name:
            st[(t >= a) & (t <= b)] = np.nan
    return st


def median_hz(clips):
    voiced = np.concatenate([d[2][d[2] > 0] for d in clips])
    return float(np.median(voiced)) if voiced.size else 150.0


tracks = {}
done = 0
for kind, group in groups:
    data = {name: analyse(name) for name in group}
    if kind == 'section':
        # each clip from its own median, so different voices share one band
        refs = {name: median_hz([d]) for name, d in data.items()}
    else:
        ref = median_hz(data.values())
        refs = {name: ref for name in group}
    cleaned = {name: semitones(name, t, f0, refs[name]) for name, (dur, t, f0, ti, db) in data.items()}
    kept = np.concatenate([c[~np.isnan(c)] for c in cleaned.values()])
    lo, hi = (np.percentile(kept, 1) - 1, np.percentile(kept, 99) + 1) if kept.size else (-6.0, 6.0)
    db_max = max(d[4].max() for d in data.values())
    for name, (dur, t, f0, ti, db) in data.items():
        span = dur if kind == 'section' else max(d[0] for d in data.values())
        x = lambda s: round(s / span * W, 1)
        y = lambda v: round(H - PAD - (min(max(v, lo), hi) - lo) / (hi - lo) * (H - 2 * PAD), 1)
        st = cleaned[name]
        # lift the pen at unvoiced frames and at octave-sized jumps (into or out of creak),
        # so the two registers are not joined by a vertical stroke
        parts, pen, last = [], False, None
        for s, v in zip(t[::2], st[::2]):
            if np.isnan(v):
                pen = False
                continue
            if pen and abs(v - last) > 8:
                pen = False
            parts.append(f"{'L' if pen else 'M'}{x(s)} {y(v)}")
            pen, last = True, v
        floor = db_max - 45
        area = [f'M{x(ti[0])} {H}']
        for s, v in zip(ti[::2], db[::2]):
            area.append(f'L{x(s)} {round(H - max(0, v - floor) / 45 * (H * .55), 1)}')
        area.append(f'L{x(ti[-1])} {H}Z')
        tracks[name] = {'dur': round(dur, 3), 'span': round(span, 3), 'pitch': ''.join(parts), 'loud': ''.join(area)}
        done += 1
        print(f'[{done:2}/{total}] {name}: {dur:.3f}s, {len(parts)} pitch points', flush=True)


def fill(match):
    player = match.group(0)
    srcs = re.findall(r'data-src="assets/audio/([^"]+)"', player)
    player = re.sub(r'data-span="[^"]*"', f'data-span="{tracks[srcs[0]]["span"]}"', player, count=1)
    for i, src in enumerate(srcs):
        dur = tracks[src]['dur']
        player = re.sub(rf'(data-src="assets/audio/{re.escape(src)}" data-duration=")[^"]*(">.*?<span class="ab-time">)[^<]*',
                        lambda m: f'{m.group(1)}{dur}{m.group(2)}{dur:.1f}s', player, count=1, flags=re.S)
        for kind in ('loud', 'pitch'):
            player = re.sub(rf'(<path class="{kind}" data-v="{i}" d=")[^"]*(")', lambda m: m.group(1) + tracks[src][kind] + m.group(2), player)
    return player


# Each player runs from its opening tag to its track's fill span (past the play icon's own <svg>).
html, n = re.subn(r'<div class="ab[^"]*" data-sel.*?<span class="ab-fill">', fill, html, flags=re.S)
INDEX.write_text(html)
print(f'{done} ok, 0 failed; updated {n} players in index.html', flush=True)
