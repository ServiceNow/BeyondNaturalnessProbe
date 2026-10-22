# Beyond Naturalness website

This `/docs` folder is the project website for **Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions**. It is one page whose main job is to let visitors hear the benchmark's clips, organized as level › dimension. GitHub Pages publishes it from the `main` branch's `/docs` folder without GitHub Actions.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8765 --directory docs
```

Then open <http://127.0.0.1:8765>. No build step is required.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | The whole site: hero, Listen (three levels, ten dimensions, reference and error clips), Method, Findings (evaluator-by-dimension grid), Paper |
| `site.css` | All styles, in Apple's design language: SF Pro (Inter elsewhere), glass cards over a mesh gradient, pill controls. Light by default, dark under `prefers-color-scheme: dark` |
| `site.js` | A/B players (Web Audio, decoded on first tap; switching version keeps the playback position), breadcrumb, click-to-centre, grid links, BibTeX copy |
| `assets/audio/<clip-id>.mp3` | 43 curated clips from the 860-utterance split, 64 kbps mono. Durations are baked into `index.html` so nothing loads until a clip is played |
| `assets/audio/make_tracks.py` | Computes each clip's pitch and loudness track with Praat and writes them, with each clip's duration, into the players in `index.html` |
| `assets/paper/make_diagram.py` | Writes the pipeline figure into `index.html` as inline SVG, redrawn component by component on the paper figure's 3200 × 1085 grid, so it follows light/dark mode and builds as it scrolls into view |
| `assets/paper/pipeline.svg` | The paper's pipeline figure, exported from the paper PDF; linked from the caption as "Open the paper's figure" |
| `assets/paper/og.png` | 1200×630 link-preview image referenced by the Open Graph tags, rendered from `assets/paper/og.html` (open it in a browser at 1200×630 and capture the viewport) |
| `assets/paper/fix_svg_icons.py` | Post-export fix so the figure's icons render in Safari (see below) |

## Updating the pipeline figure

The page draws the figure itself. To change a box, label or arrow, edit `make_diagram.py` (each component is one call, at its position in the 3200 × 1085 grid; `--d` is when it appears during the build) and rerun it from the repository root:

```bash
python3 docs/assets/paper/make_diagram.py docs/index.html
```

If the paper's figure itself changes, update `make_diagram.py` to match, then replace `pipeline.svg`, the copy the caption links to.

`pdftocairo -svg` draws each icon through a soft mask that Safari (WebKit) does not render, so the level icons and the Input/Output Sentences icons disappear there. After exporting a new figure, run:

```bash
uv run --with pillow python docs/assets/paper/fix_svg_icons.py exported.svg docs/assets/paper/pipeline.svg
```

It folds each icon's mask into the icon's own alpha channel. Check the result in Safari.

## Editing guidance

1. Keep claims aligned with the paper; the benchmark evaluates evaluators, it does not rank TTS systems.
2. Use **evaluate/evaluation**, not **test**, when describing what the benchmark does.
3. Keep the three levels (word, prosodic, paralinguistic) and their ten dimensions in the paper's order.
4. Keep the official paper title and contributor order unchanged unless the paper metadata changes.
5. Check desktop and mobile widths, light and dark mode, and Safari after editing.

The findings grid in `index.html` is transcribed from the paper's Kendall's τb table (significant positive correlations, naturalness prompt with transcript). Update it by hand if the results change.

## Updating clips or pitch tracks

After adding or replacing a clip, regenerate the tracks drawn under each player (from the repository root; needs `ffmpeg` and Praat 6.4 or later, e.g. `brew install --cask praat`):

```bash
uv run --no-project --with praat-parselmouth --with numpy python docs/assets/audio/make_tracks.py
```

Pitch is Praat's filtered autocorrelation at its standard settings, which the Praat manual recommends for intonation. There is no automatic cleanup; the few octave errors it makes are listed as hand corrections in `CORRECTIONS` at the top of the script, each with its reason. The clips in one example share a time axis and a pitch scale, so a reference and its error line up. Dimensions made only of single-clip examples (expressiveness, prosodic boundaries) share one pitch band across the dimension, so a monotone clip looks flatter than an overacted one. Clips are decoded with ffmpeg rather than Praat's MP3 reader, which keeps the encoder delay and would put the tracks about 37 ms behind the audio.
