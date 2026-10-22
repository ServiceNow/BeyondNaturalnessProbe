"""Generate the inline, animated pipeline diagram for docs/index.html.

Coordinates are in the paper figure's own pixel grid (pipeline.svg rendered 3200 px wide, 3200 x 1085),
measured component by component, so the layout matches the figure exactly.
Usage (from the repository root): python3 docs/assets/paper/make_diagram.py docs/index.html
"""
import re
import sys

out = []
W = out.append


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, size, body, cls='', anchor='middle', d=None, row=None):
    a = f' data-a style="--d:{d}s"' if d is not None else ''
    r = f' data-row="{row}"' if row else ''
    c = f' class="{cls}"' if cls else ''
    W(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{c}{a}{r}>{body}</text>')


def group(x, y, w, h, d):
    W(f'<rect class="dg-group" x="{x}" y="{y}" width="{w}" height="{h}" rx="14" data-a style="--d:{d}s"/>')


def box(x, y, w, h, d, row=None, cls='dg-box', glow=None):
    """A box; `glow` names the moment in the pulse cycle when it lights up (see site.css)."""
    r = f' data-row="{row}"' if row else ''
    W(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="14" data-a style="--d:{d}s"{r}/>')
    if glow:
        lv = f' data-level="{row}"' if row else ''
        W(f'<rect class="dg-glow dg-at{glow}" x="{x}" y="{y}" width="{w}" height="{h}" rx="14"{lv}{r}/>')


def arrow(x1, x2, y, d, cls='', row=None):
    """Horizontal arrow from x1 to the tip at x2; the line draws on, then the head appears."""
    r = f' data-row="{row}"' if row else ''
    W(f'<path class="dg-line dg-draw {cls}" d="M{x1} {y}H{x2 - 20}" pathLength="1" style="--d:{d}s"{r}/>')
    W(f'<path class="dg-head {cls}" d="M{x2} {y}l-24 -11v22z" data-a style="--d:{d + .3:.2f}s"{r}/>')


def lines(*ys, x, y0, size, cls='', d=None, row=None):
    for i, body in enumerate(ys):
        text(x, y0 + i * 30, size, body, cls, d=d, row=row)


def inj(s, at):
    """An injected change; it flickers when the pulse reaches its box (`at` % of the cycle)."""
    return f'<tspan class="dg-inj dg-f{at}">{s}</tspan>'


W('<svg class="dg" viewBox="0 0 3200 1085" role="img" aria-labelledby="dg-title dg-desc">')
W('<defs>'
  '<linearGradient id="dg-shine" x1="0" x2="1"><stop offset="0" class="dg-s0"/><stop offset=".5" class="dg-s1"/><stop offset="1" class="dg-s0"/></linearGradient>'
  '<linearGradient id="dg-scan-g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" class="dg-c0"/><stop offset="1" class="dg-c1"/></linearGradient>'
  '<clipPath id="dg-clip-ann"><rect x="1543" y="256" width="350" height="574" rx="14"/></clipPath>'
  '<clipPath id="dg-clip-r1"><rect x="2483" y="238" width="655" height="125"/></clipPath>'
  '<clipPath id="dg-clip-r2"><rect x="2483" y="406" width="655" height="125"/></clipPath>'
  '</defs>')
W('<title id="dg-title">How the benchmark was built</title>')
W('<desc id="dg-desc">Pipeline: sentences from EmergentTTS-Eval, Harvard Sentences and newly written ones get one '
  'dimension-specific error each, are synthesized with TTS or edited acoustically, labelled by three linguists on every '
  'dimension, then scored by MOS predictors and Audio-LLM judges and correlated with the human labels per dimension.</desc>')

# ---------- Data generation ----------
group(18, 10, 1950, 1068, 0)
text(1083, 56, 40, 'Data Generation', 'dg-h', d=0.05)

# Source corpora
group(38, 76, 432, 982, .1)
text(254, 130, 33, 'Source Corpora', 'dg-h', d=.15)
for i, s in enumerate(['EmergentTTS-Eval', 'Harvard Sentences', 'Original Synthetically Generated']):
    y = 188 + i * 30
    W(f'<circle cx="70" cy="{y - 8}" r="4" class="dg-dot" data-a style="--d:{.2 + i * .05:.2f}s"/>')
    text(86, y, 24, s, anchor='start', d=round(.2 + i * .05, 2))

# Level icons and labels, one per row
W('<g class="dg-lv" data-level="word" data-row="word" data-a style="--d:.35s">'
  '<text x="62" y="404" font-size="36" text-anchor="start" class="dg-h dg-ico-t">ab</text>'
  '<path class="dg-ico-s" d="M62 414h46"/>'
  '<path class="dg-ico-f" d="M112 412l4-14 22-22 10 10-22 22zM130 382l10 10"/>'
  '<text x="97" y="452" font-size="21" text-anchor="middle" class="dg-h dg-lvt">Word</text></g>')
W('<g class="dg-lv" data-level="prosodic" data-row="prosodic" data-a style="--d:.4s">'
  '<path class="dg-ico-s" d="M70 574h58a3 3 0 0 1 3 3v34a3 3 0 0 1-3 3H88l-12 11v-11h-6a3 3 0 0 1-3-3v-34a3 3 0 0 1 3-3z"/>'
  + ''.join(f'<rect class="dg-ico-b" x="{75 + i * 6}" y="{594 - h / 2}" width="3" height="{h}" rx="1.5"/>'
            for i, h in enumerate([10, 18, 26, 14, 22, 30, 16, 24, 12]))
  + '<text x="97" y="659" font-size="22" text-anchor="middle" class="dg-h dg-lvt">Prosodic</text></g>')
W('<g class="dg-lv" data-level="paralinguistic" data-row="paralinguistic" data-a style="--d:.45s">'
  '<circle class="dg-ico-s" cx="82" cy="812" r="13"/><circle class="dg-ico-b" cx="77" cy="809" r="2"/><circle class="dg-ico-b" cx="87" cy="809" r="2"/>'
  '<path class="dg-ico-s" d="M76 816q6 6 12 0"/>'
  '<rect class="dg-ico-s" x="103" y="800" width="24" height="24" rx="3"/><path class="dg-ico-s" d="M108 812l5 5 9-10"/>'
  '<circle class="dg-ico-s" cx="82" cy="842" r="13"/><circle class="dg-ico-b" cx="77" cy="839" r="2"/><circle class="dg-ico-b" cx="87" cy="839" r="2"/>'
  '<path class="dg-ico-s" d="M76 849q6 -6 12 0"/>'
  '<rect class="dg-ico-s" x="103" y="830" width="24" height="24" rx="3"/>'
  '<text x="97" y="871" font-size="21" text-anchor="middle" class="dg-h dg-lvt">Para-</text>'
  '<text x="97" y="895" font-size="21" text-anchor="middle" class="dg-h dg-lvt">linguistic</text></g>')

# Input sentences
box(152, 319, 289, 149, .5, 'word', glow='0')
lines('“Dull stories make her', 'laugh”', x=296, y0=388, size=24, d=.55, row='word')
box(152, 536, 289, 148, .6, 'prosodic', glow='0')
lines('“Type out three lists of', 'orders”', x=296, y0=604, size=24, d=.65, row='prosodic')
box(158, 771, 289, 149, .7, 'paralinguistic', glow='0')
lines('“My beloved dog', 'passed away quietly in', 'his sleep tonight”', x=302, y0=825, size=24, d=.75, row='paralinguistic')

text(104, 1004, 18, '3 Levels', 'dg-sm', d=.8)
text(104, 1027, 18, '10 Dims', 'dg-sm', d=.8)


def docs_icon(x, y, d):
    W(f'<g class="dg-docs" data-a style="--d:{d}s"><path class="dg-ico-s" d="M{x} {y + 10}v48h34"/>'
      f'<path class="dg-ico-s" d="M{x + 6} {y + 4}v48h34"/>'
      f'<path class="dg-ico-s dg-paper" d="M{x + 12} {y - 2}h28l10 10v36h-38z"/>'
      + ''.join(f'<path class="dg-ico-s" d="M{x + 18} {y + 12 + i * 7}h26"/>' for i in range(5)) + '</g>')


text(393, 1010, 25, 'Input', anchor='end', d=.85)
text(393, 1041, 25, 'Sentences', anchor='end', d=.85)
docs_icon(403, 992, .85)

# Dimension-specific error injection
group(493, 76, 780, 982, .3)
text(883, 134, 33, 'Dimension Specific Error Injection', 'dg-h', d=.35)
text(883, 219, 28, 'Apply dimension-specific modifications to input text', d=.4)

arrow(441, 512, 394, .8, row='word')
arrow(441, 518, 610, .9, row='prosodic')
arrow(447, 523, 846, 1.0, row='paralinguistic')

box(513, 319, 318, 149, 1.1, 'word', glow='10')
text(672, 388, 24, 'LLM-altered IPA', 'dg-h', d=1.15, row='word')
text(672, 418, 24, f'laugh→/ˈl{inj("æ", 10)}f/→/ˈl{inj("ʊ", 10)}f/', d=1.15, row='word')
box(520, 536, 310, 148, 1.2, 'prosodic', glow='10')
text(675, 589, 24, 'LLM-altered text', 'dg-h', d=1.25, row='prosodic')
lines('Wrap preposition in', 'quotes to inject stress', x=675, y0=619, size=24, d=1.25, row='prosodic')
box(525, 771, 306, 149, 1.3, 'paralinguistic', glow='10')
text(678, 840, 24, 'API-enforced emotion', 'dg-h', d=1.35, row='paralinguistic')
text(678, 870, 24, f'Apply ‘{inj("euphoric", 10)}’ tag', d=1.35, row='paralinguistic')

arrow(831, 898, 394, 1.5, row='word')
arrow(830, 906, 610, 1.6, row='prosodic')
arrow(831, 906, 846, 1.7, row='paralinguistic')

box(901, 319, 349, 149, 1.8, 'word', glow='30')
text(1075, 388, 24, '“Dull stories make her', d=1.85, row='word')
text(1075, 418, 24, inj(esc('<<ˈ|l|ʊ|f>>'), 30) + '”', d=1.85, row='word')
box(909, 536, 351, 148, 1.9, 'prosodic', glow='30')
text(1084, 604, 24, f'“Type out three lists {inj("“of”", 30)}', d=1.95, row='prosodic')
text(1084, 634, 24, 'orders”', d=1.95, row='prosodic')
box(908, 771, 350, 149, 2.0, 'paralinguistic', glow='30')
text(1083, 825, 24, f'“{inj("[emotion: euphoric]", 30)} My', d=2.05, row='paralinguistic')
lines('beloved dog passed away', 'quietly in his sleep tonight.”', x=1083, y0=855, size=24, d=2.05, row='paralinguistic')

text(1190, 1010, 25, 'Output', anchor='end', d=2.1)
text(1190, 1041, 25, 'Sentences', anchor='end', d=2.1)
docs_icon(1200, 992, 2.1)

# Merge into TTS: the word and prosodic outputs in black, the paralinguistic one in blue
W('<path class="dg-line dg-draw" d="M1250 394H1300V610H1260" pathLength="1" style="--d:2.3s"/>')
W('<path class="dg-line dg-alt dg-draw" d="M1258 846H1300V610" pathLength="1" style="--d:2.35s"/>')
arrow(1300, 1340, 503, 2.55)

W('<g class="dg-tts" data-a style="--d:2.6s">'
  '<text x="1382" y="447" font-size="42" class="dg-h">TTS</text>'
  '<path class="dg-paper-f" d="M1352 472h30l10 10v50h-40z"/>'
  '<path class="dg-gap" d="M1360 492h18M1360 502h22M1360 512h16M1360 522h20"/>'
  '<path class="dg-ink" d="M1384 490h8l14-12v42l-14-12h-8z"/>'
  '<path class="dg-wave" d="M1412 488q7 11 0 22"/>'
  '<path class="dg-wave dg-wave2" d="M1419 482q13 17 0 34"/></g>')
arrow(1422, 1478, 490, 2.9)

# Acoustic manipulation (dashed, blue)
W('<path class="dg-line dg-alt dg-dash" d="M1382 542V618" data-a style="--d:2.9s"/>')
W('<path class="dg-head dg-alt" d="M1382 640l-11-24h22z" data-a style="--d:3.0s"/>')
W('<g class="dg-acoustic" data-a style="--d:3.05s">'
  '<rect class="dg-ico-s dg-thick" x="1352" y="654" width="62" height="62" rx="14"/>'
  + ''.join(f'<rect class="dg-bar" x="{1364 + i * 8}" y="{685 - h / 2}" width="4" height="{h}" rx="2" style="--i:{i}"/>'
            for i, h in enumerate([14, 30, 22, 36, 18]))
  + '<text x="1390" y="752" font-size="22" class="dg-h">Acoustic</text>'
  '<text x="1390" y="781" font-size="22" class="dg-h">Manipulation</text></g>')
W('<path class="dg-line dg-alt dg-dash" d="M1420 686H1456" data-a style="--d:3.2s"/>')
W('<path class="dg-head dg-alt" d="M1478 686l-24-11v22z" data-a style="--d:3.25s"/>')

# Linguist annotation
group(1480, 76, 468, 982, 2.8)
text(1715, 147, 40, 'Linguist', 'dg-h', d=2.85)
text(1715, 197, 40, 'Annotation', 'dg-h', d=2.85)
box(1543, 256, 350, 574, 3.3, glow='70')
W('<g clip-path="url(#dg-clip-ann)"><rect class="dg-scan" x="1543" y="200" width="350" height="56" fill="url(#dg-scan-g)"/></g>')
for i, s in enumerate(['Score every', 'dimension for', 'each speech', 'sample:', 'failure (0)', 'good (1)']):
    text(1718, 445 + i * 46, 36, s, d=3.4)
text(1715, 892, 32, '860 Samples', d=3.5)
text(1865, 1010, 24, 'Majority', anchor='end', d=3.6)
text(1855, 1041, 24, 'Voting', anchor='end', d=3.6)
people = []
for r, (cy, filled) in enumerate([(996, (1, 1, 1)), (1020, (1, 0, 0))]):
    for c, f in enumerate(filled):
        cx = 1884 + c * 22
        cls = 'dg-p dg-p-on' if f else 'dg-p'
        people.append(f'<g class="{cls}" style="--i:{r * 3 + c}"><circle cx="{cx}" cy="{cy - 6}" r="5"/>'
                      f'<path d="M{cx - 9} {cy + 9}v-3a6 6 0 0 1 6-6h6a6 6 0 0 1 6 6v3z"/></g>')
W('<g class="dg-vote" data-a style="--d:3.6s">' + ''.join(people) + '</g>')

# ---------- Evaluation ----------
group(1988, 12, 1202, 1063, 3.8)
text(2652, 59, 40, 'Evaluation', 'dg-h', d=3.85)

group(2012, 76, 408, 528, 3.9)
text(2217, 153, 36, 'Automated Judge', 'dg-h', d=3.95)
text(2217, 199, 36, 'Classes', 'dg-h', d=3.95)
box(2070, 242, 292, 116, 4.0, glow='e0')
W('<g class="dg-mos" data-a style="--d:4.05s">'
  '<path class="dg-net" d="M2100 276L2125 268M2100 276L2125 284M2100 296L2125 284M2100 296L2125 300M2100 316L2125 300'
  'M2100 316L2125 316M2125 268L2152 288M2125 284L2152 288M2125 300L2152 306M2125 316L2152 306"/>'
  + ''.join(f'<circle class="dg-n1" cx="2100" cy="{y}" r="6"/>' for y in (276, 296, 316))
  + ''.join(f'<circle class="dg-n2" cx="2125" cy="{y}" r="6"/>' for y in (268, 284, 300, 316))
  + ''.join(f'<circle class="dg-n3" cx="2152" cy="{y}" r="6"/>' for y in (288, 306))
  + '</g>')
text(2260, 291, 28, 'MOS', d=4.05)
text(2260, 325, 28, 'Predictors', d=4.05)
box(2070, 410, 292, 116, 4.1, glow='e15')
W('<g class="dg-robot" data-a style="--d:4.15s">'
  '<path class="dg-bub" d="M2110 432h24a4 4 0 0 1 4 4v12a4 4 0 0 1-4 4h-8l-4 6-4-6h-8a4 4 0 0 1-4-4v-12a4 4 0 0 1 4-4z"/>'
  '<rect class="dg-neck" x="2116" y="458" width="10" height="8" rx="2"/>'
  '<rect class="dg-head-r" x="2091" y="466" width="60" height="40" rx="10"/>'
  '<rect class="dg-ear" x="2085" y="476" width="6" height="20" rx="2"/><rect class="dg-ear" x="2151" y="476" width="6" height="20" rx="2"/>'
  '<rect class="dg-screen" x="2099" y="473" width="44" height="26" rx="6"/>'
  '<circle class="dg-eye" cx="2111" cy="486" r="3"/><circle class="dg-eye" cx="2131" cy="486" r="3"/>'
  '<path class="dg-smile" d="M2116 492q5 4 10 0"/></g>')
text(2258, 461, 28, 'Audio LLM', d=4.15)
text(2258, 496, 28, 'Judges', d=4.15)
arrow(2362, 2530, 300, 4.3, 'dg-faint')
arrow(2362, 2530, 468, 4.35, 'dg-faint')

group(2458, 76, 704, 528, 4.2)
text(2810, 152, 35, 'Dimension Level Correlations', 'dg-h', d=4.25)
for x, s, lv in ((2576, 'Word', 'word'), (2809, 'Prosodic', 'prosodic'), (3039, 'Paralinguistic', 'paralinguistic')):
    W(f'<text x="{x}" y="219" font-size="30" text-anchor="middle" class="dg-h dg-lvt" data-level="{lv}" data-a style="--d:4.4s">{s}</text>')
box(2483, 238, 655, 125, 4.45, cls='dg-box dg-row', glow='e30')
box(2483, 406, 655, 125, 4.5, cls='dg-box dg-row', glow='e45')
cells = [(2503, 247, 'r', '0.12'), (2731, 247, 'a', '0.23'), (2955, 247, 'g', '0.4'),
         (2503, 415, 'a', '0.32'), (2730, 415, 'g', '0.36'), (2960, 415, 'r', '0.15')]
for i, (x, y, k, v) in enumerate(cells):
    W(f'<g class="dg-cell dg-{k}" data-a style="--d:{4.6 + i * .08:.2f}s"><rect x="{x}" y="{y}" width="159" height="105" rx="4"/>'
      f'<text x="{x + 80}" y="{y + 64}" font-size="34" text-anchor="middle" data-v="{v}">{v}</text></g>')
W('<g clip-path="url(#dg-clip-r1)"><rect class="dg-shine dg-sh30" x="2323" y="238" width="160" height="125" fill="url(#dg-shine)"/></g>')
W('<g clip-path="url(#dg-clip-r2)"><rect class="dg-shine dg-sh45" x="2323" y="406" width="160" height="125" fill="url(#dg-shine)"/></g>')

group(2012, 644, 704, 408, 4.55)
text(2364, 703, 37, 'Input Conditions', 'dg-h', d=4.6)
box(2037, 733, 303, 120, 4.75)
text(2188, 802, 24, 'Transcript Access', d=4.8)
box(2382, 737, 293, 118, 4.85)
text(2528, 790, 24, 'Dimension description', d=4.9)
text(2528, 820, 24, 'access', d=4.9)
box(2218, 898, 292, 117, 4.95)
text(2364, 966, 24, 'Output granularity', d=5.0)

group(2746, 644, 410, 408, 4.6)
text(2950, 706, 37, 'Metrics', 'dg-h', d=4.65)
box(2808, 745, 293, 117, 4.9)
text(2954, 813, 24, 'Correlations', d=4.95)
box(2808, 902, 293, 118, 5.0)
text(2954, 970, 24, 'AUROC', d=5.05)

# ---------- Flow pulses: start once the figure has built, loop, and never run under reduced motion ----------
W('<g class="dg-pulses" visibility="hidden"><set id="dg-go" attributeName="visibility" to="visible" begin="indefinite"/>')


def pulse(path, a, b, cls, cycle=4.6, start=0):
    """A dot that travels `path` between fractions a and b of a looping cycle."""
    e = .02
    W(f'<circle r="7" class="dg-pulse {cls}" opacity="0">'
      f'<animateMotion path="{path}" dur="{cycle}s" repeatCount="indefinite" begin="dg-go.begin+{start}s" '
      f'keyPoints="0;0;1;1" keyTimes="0;{a};{b};1" calcMode="linear"/>'
      f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{a};{a + e:.3f};{b - e:.3f};{b};1" '
      f'dur="{cycle}s" repeatCount="indefinite" begin="dg-go.begin+{start}s"/></circle>')


rows = [('dg-pw', 394, 441, 512, 831, 898, 'M1250 394H1300V503H1340'),
        ('dg-pp', 610, 441, 518, 830, 906, 'M1260 610H1300V503H1340'),
        ('dg-pa', 846, 447, 523, 831, 906, 'M1258 846H1300V503H1340')]
for cls, y, x1, x2, x3, x4, merge in rows:
    pulse(f'M{x1} {y}H{x2}', 0, .1, cls)
    pulse(f'M{x3} {y}H{x4}', .2, .3, cls)
    pulse(merge, .4, .55, cls)
pulse('M1422 490H1478', .6, .7, 'dg-pacc')
pulse('M1382 542V640', .6, .7, 'dg-palt')
pulse('M1420 686H1478', .72, .8, 'dg-palt')
pulse('M2362 300H2530', 0, .3, 'dg-pacc', cycle=3, start=.4)
pulse('M2362 468H2530', .15, .45, 'dg-pacc', cycle=3, start=.4)
W('</g></svg>')

svg = '\n'.join(out)
path = sys.argv[1]
html = open(path, encoding='utf-8').read()
new, n = re.subn(r'(<div class="diagram-scroll"[^>]*>\n).*?(\n</div>)', lambda m: m.group(1) + svg + m.group(2), html, count=1, flags=re.S)
assert n == 1
open(path, 'w', encoding='utf-8').write(new)
print('ok', len(svg))
