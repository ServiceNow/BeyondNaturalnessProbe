<h1 align="center">Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions</h1>

<p align="center">
  <strong>Controlled synthetic speech probes for evaluating TTS quality beyond naturalness across word-level, prosodic, and paralinguistic dimensions.</strong>
</p>

<p align="center">
  <strong>🏆 Accepted to EMNLP 2026</strong>
</p>

<p align="center">
  <a href="https://servicenow.github.io/BeyondNaturalnessProbe/">🌐 Website</a> ·
  <a href="https://huggingface.co/datasets/ServiceNow-AI/Beyond-Naturalness-TTS-Eval">🤗 Dataset</a> ·
  <a href="https://arxiv.org/abs/2608.09930v1">📄 Paper</a>
</p>

<p align="center">
  <img alt="Active pipelines" src="https://img.shields.io/badge/pipelines-9_active-2ea44f?style=for-the-badge">
  <img alt="Dimensions" src="https://img.shields.io/badge/covering-9_dimensions-0969da?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-3776ab?style=for-the-badge&logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-f97316?style=for-the-badge">
</p>

<p align="center">
  <a href="#at-a-glance">At a Glance</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#active-pipelines">Active Pipelines</a> ·
  <a href="#dimension-details">Dimension Details</a> ·
  <a href="#running">Running</a> ·
  <a href="#ipa-format-word-level-dimensions">IPA Format</a> ·
  <a href="#project-structure">Project Structure</a> ·
  <a href="#-citation">Citation</a> ·
  <a href="#license">License</a>
</p>

---

## At a Glance

| 🪜 Graded variants | 🔊 Synthesis | 🎛️ Manipulations | 🧾 Traceable outputs |
|---|---|---|---|
| Emit score-specific variants per dimension | Cartesia and ElevenLabs TTS | Phonemic input edits (segments, lexical stress), pitch-accent and boundary placement via text, F0 contour and tempo via PSOLA, voice splicing, DSP signal artifacts | WAV files plus `manifest.json` records under `results/` |

BeyondNaturalnessProbe generates synthetic speech probes for testing automated TTS evaluators. For each dimension it produces variants of the same utterance at controlled rubric levels, so an evaluator's scores can be checked against a known target.

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for package management. `uv sync` creates `.venv/` and installs everything, including Python 3.14 if needed.

**1. Install dependencies**

```bash
git clone https://github.com/ServiceNow/BeyondNaturalnessProbe.git && cd BeyondNaturalnessProbe
uv sync
```

**2. Configure credentials**

```bash
cp .env.example .env
```

Fill in the values your chosen dimension needs. `.env.example` documents standard OpenAI (`OPENAI_API_KEY`), Azure OpenAI (`AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY`), `CARTESIA_API_KEY`, and `ELEVEN_LABS_API_KEY`. Azure OpenAI takes priority if both OpenAI options are set.

Only the credentials used by the selected dimension are required: `speech_rate`, `intonation`, `expressiveness`, `speaker_identity_consistency`, and `prosodic_boundary` with the default `preset` strategy do not call OpenAI.

**3. Run a smoke test**

```bash
uv run python pipeline.py --dimension lexical_stress --num 1
```

Prefix commands with `uv run` (for example, `uv run python pipeline.py ...`) or activate the virtual environment with `source .venv/bin/activate`.

> [!CAUTION]
> Full runs may call paid third-party APIs (Cartesia, OpenAI, ElevenLabs). Always sanity-check with `--num 1` before running a full batch. `--num` does not apply to two dimensions: `expressiveness` uses `--counts` instead (default `50,20,30`, so 100 items; use `--counts 1,1,1` for one per score), and `prosodic_boundary` with the default `preset` strategy always renders all 60 preset sentences. `uv run python pipeline.py --dimension all` fans out to 7 dimensions concurrently — budget accordingly.

The rest of this README is reference material: the full dimension list, output layout, all CLI flags, and how each pipeline works internally.

## Active Pipelines

| Pipeline | Scores | Engine | Description |
|----------|--------|--------|-------------|
| `lexical_stress` | `0`, `1` | Cartesia sonic-3 | `0` = wrong lexical stress placement<br>`1` = correct lexical stress placement |
| `phonetic_accuracy` | `1`, `2`, `3` | Cartesia sonic-3 | `1` = salient substitution (usually the stressed vowel)<br>`2` = moderate substitution (often a consonant or a one-step vowel shift)<br>`3` = correct pronunciation |
| `prosodic_stress` | `0`, `1` | Cartesia sonic-3 | `0` = misplaced pitch accent (on function words)<br>`1` = natural pitch-accent placement |
| `speech_rate` | `1`, `2`, `3` | Cartesia + PSOLA | `1` = high negative impact from extreme speed (score 2 plus PSOLA)<br>`2` = low negative impact from moderate speed<br>`3` = no impact; normal speech rate |
| `speaker_identity_consistency` | `0`, `1` | Cartesia SSE | `0` = inconsistent voice via vowel crossfade or cross-utterance stitch<br>`1` = consistent voice; same content rendered by a single voice |
| `intonation` | `1`, `2`, `3` | ElevenLabs + Praat | `1` = egregious F0 distortion<br>`2` = attenuated terminal contour<br>`3` = natural question intonation |
| `expressiveness` | `1`, `2`, `3` | Cartesia sonic-3 + PSOLA | `1` = over-expressive; Scott + `emotion="confident"`<br>`2` = under-expressive; Sienna + `emotion="bored"` + PSOLA F0 flattening<br>`3` = natural; Brooke + `speed=0.8` |
| `prosodic_boundary` | `1`, `2`, `3` | Cartesia sonic-3 | `1` = severe boundary error (missing or extra); intended structure unrecoverable<br>`2` = noticeable but recoverable boundary error (missing or extra)<br>`3` = canonical boundaries |
| `human_plausibility` | `0`, `1` | Cartesia sonic-3 + DSP | `0` = 1-2 digital artifacts applied by post-processing<br>`1` = clean reference audio |

The active pipelines are organized around word-level, prosodic, and paralinguistic evaluation dimensions.

`lexical_stress`, `phonetic_accuracy`, `prosodic_stress`, and `human_plausibility` use **OpenAI (GPT-5)** to generate IPA phoneme variants (word-level dimensions), quote-marked text variants (`prosodic_stress`), or artifact recipes (`human_plausibility`), and **Cartesia sonic-3** to synthesize audio. Either standard OpenAI (`OPENAI_API_KEY`) or Azure OpenAI (`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_API_KEY`) credentials work — see [Quickstart](#quickstart).

`speech_rate` skips OpenAI — it generates audio at normal speed via Cartesia, then uses Cartesia's speed parameter and **PSOLA** (Praat, via parselmouth) time-stretching to produce fast/slow variants.

`speaker_identity_consistency` skips the OpenAI phase — it uses Cartesia SSE (with word + phoneme timestamps) directly.

`intonation` synthesizes the sentence verbatim via **ElevenLabs** and applies **Praat/PSOLA** F0 contour manipulations — no OpenAI involvement.

`human_plausibility` uses GPT-5 once per utterance to pick an artifact recipe (type/location/severity — the artifact *type* itself is pre-selected in code, not by the model), synthesizes the clean sentence once via Cartesia, then applies the recipe as post-processing DSP (`shared/artifact_utils.py`) to produce the degraded score-0 variant.


## Dimension Details

<details open>
<summary><big><b>🧬 Word-level dimensions</b> <i>(<code>phonetic_accuracy</code>, <code>lexical_stress</code>)</i></big></summary>

> #### 🧬 `phonetic_accuracy` & `lexical_stress`
>
> **Dimensions:** `phonetic_accuracy`, `lexical_stress`  
> **Engines:** GPT-5 + Cartesia sonic-3  
> **Output:** `results/{dimension}/{dataset}_{id}_{word}/`
>
> These dimensions create controlled word-level pronunciation variants. Each item follows the same two-phase pattern:
>
> | Step | What happens |
> |---|---|
> | **Phase 1** | GPT-5 generates IPA phoneme variants per utterance, concurrently |
> | **Phase 2** | Cartesia sonic-3 synthesizes each variant to WAV, concurrently |
> | **Write outputs** | Each item gets `manifest.json` plus score-specific `.wav` files |
> | **Summarize** | Appends an Excel summary to `results/summary.xlsx` |

</details>

<br>

<details open>
<summary><big><b>📈 Prosodic dimensions</b> <i>(<code>intonation</code>, <code>prosodic_stress</code>, <code>prosodic_boundary</code>, <code>speech_rate</code>)</i></big></summary>

> #### 📈 `intonation`
>
> **Dimension:** `intonation`  
> **Engines:** ElevenLabs + Praat/parselmouth  
> **Output:** `results/intonation/{dataset}_{id}/`
>
> | Score | Generation recipe |
> |---|---|
> | **3** | ElevenLabs synthesizes the question verbatim as the natural anchor |
> | **2** | `subdued_terminal_rise`: pulls the last ~30% of the F0 contour toward the speaker's mean, keeping only 15-30% of its natural excursion |
> | **1** | Applies one of 10 egregious F0 strategies, such as contour reversal, inversion, staircase quantization, or random jumps |
>
> Score 2 works across WH-questions and yes/no questions because it attenuates whatever terminal contour is there (the rise of a yes/no question, the fall of a WH-question) instead of replacing it with a fixed shape.
>
> ---
>
> #### 🗯️ `prosodic_stress`
>
> **Dimension:** `prosodic_stress`  
> **Engines:** GPT-5 + Cartesia sonic-3  
> **Output:** `results/prosodic_stress/{dataset}_{id}_{word}/`
>
> This dimension uses the same GPT-5 → Cartesia pattern as the word-level dimensions, but targets sentence-level stress placement rather than segmental pronunciation. Instead of IPA, it edits the text: the score-0 variant wraps each run of function words in curly quotes, which pushes the pitch accent onto words that would normally be unaccented under broad focus.
>
> | Step | What happens |
> |---|---|
> | **Phase 1** | GPT-5 marks function-word runs with curly quotes to produce the misplaced-stress variant; the original sentence is the correct variant |
> | **Phase 2** | Cartesia sonic-3 renders each score variant |
> | **Write outputs** | Each item gets `manifest.json` plus score-specific `.wav` files |
>
> ---
>
> #### 🚧 `prosodic_boundary`
>
> **Dimension:** `prosodic_boundary`  
> **Engine:** Cartesia sonic-3  
> **Output:** `results/prosodic_boundary/{dataset}_{id}/`
>
> No OpenAI phase — transformations are fully programmatic. The strategy determines how boundary cues are corrupted or augmented:
>
> | Strategy | Score 0 transform | Score 1 |
> |----------|-------------------|---------|
> | `omit_spaces` | All whitespace removed — words run together | Original text |
> | `omit_punctuation` | Internal commas, dashes, semicolons, colons, ellipses stripped | Original text |
> | `extra_punctuation` | Commas inserted between non-boundary words (GPT-5 phase) | Original text |
> | `preset` *(default)* | Fixed 60-sentence dataset with pre-assigned scores 1/2/3 (see below) | — |
>
> The default `preset` strategy uses `data/prosodic_boundary_preset.json` — 60 manually curated sentences with 20 each at scores 1, 2, 3. Scores 1 and 2 are severity levels, each split evenly between missing and extra boundaries; score 3 is natural phrasing. The non-preset strategies use a binary 0/1 scale instead. This avoids any variability in the GPT-5 transform and is the recommended mode for benchmark dataset generation.
>
> This dimension is excluded from `--dimension all`; run it explicitly with `--dimension prosodic_boundary`.
>
> ---
>
> #### ⏱️ `speech_rate`
>
> **Dimension:** `speech_rate`  
> **Engines:** Cartesia sonic-3 + PSOLA  
> **Output:** `results/speech_rate/{id}_{direction}/`
>
> | Score | Generation recipe |
> |---|---|
> | **3** | Cartesia speed `1.0`; normal baseline |
> | **2** | Cartesia speed `0.6` for slow or `1.4` for fast; no PSOLA |
> | **1** | Copies score-2 audio, then applies a PSOLA duration factor of 0.7 (fast) or 1.3 (slow) |
>
> Fast and slow directions alternate across utterances to keep the dataset balanced.

</details>

<br>

<details open>
<summary><big><b>🎭 Paralinguistic dimensions</b> <i>(<code>expressiveness</code>, <code>speaker_identity_consistency</code>, <code>human_plausibility</code>)</i></big></summary>

> #### 🎭 `expressiveness`
>
> **Dimension:** `expressiveness`  
> **Engine:** Cartesia sonic-3, with PSOLA flattening for score 2  
> **Output:** `results/expressiveness/{score}_{id}/`
>
> Each sentence gets **one** render, not three. The score is assigned at selection time:
>
> | Score | Label | Voice | Recipe |
> |-------|-------|-------|--------|
> | 1 | over-expressive | Scott - Sportscaster | Cartesia `emotion="confident"` — theatrical broadcast delivery |
> | 2 | under-expressive | Sienna - Encourager | Cartesia `emotion="bored"` + PSOLA flattening (α=0.5, intensity compression, timing compensation on vowels lengthened to ≥2.5× the median vowel duration) |
> | 3 | natural | Brooke - Big Sister | Cartesia `speed=0.8` — no emotion tag |
>
> Use `--counts N1,N2,N3` to control how many utterances get each score; for example, `--counts 50,20,30` generates 50 over-expressive, 20 under-expressive, and 30 natural items from Harvard.
>
> This dimension is excluded from `--dimension all`; run it explicitly with `--dimension expressiveness`.
>
> **Dependencies:** score-2 items require `pyworld` and `soundfile` (`uv sync` installs them). The lazy import in `shared/expressiveness.py` means missing deps only cause a runtime error on score-2 items, not at startup.
>
> ---
>
> #### 🗣️ `speaker_identity_consistency`
>
> **Dimension:** `speaker_identity_consistency`  
> **Engine:** Cartesia SSE with word + phoneme timestamps  
> **Output:** `results/speaker_identity_consistency/...`
>
> Each sample is paired:
>
> | Score | Construction |
> |---|---|
> | **0** | Inconsistent voice: vowel-crossfade within one sentence, or stitch of two utterances spoken by different voices |
> | **1** | Consistent voice: the same sentence or utterance pair rendered by a single voice |
>
> Samples follow a balanced 2×2×2 schedule over mode, gender, and voice order. Every 8 consecutive samples visit all combinations exactly once:
>
> | # | mode | gender | order |
> |---|------|--------|-------|
> | 0 | crossfade | feminine | Brooke → Emma |
> | 1 | stitch | feminine | Brooke → Emma |
> | 2 | crossfade | masculine | Ronald → Henry |
> | 3 | stitch | masculine | Ronald → Henry |
> | 4 | crossfade | feminine | Emma → Brooke |
> | 5 | stitch | feminine | Emma → Brooke |
> | 6 | crossfade | masculine | Henry → Ronald |
> | 7 | stitch | masculine | Henry → Ronald |
>
> **Crossfade mode:** the fade region is a full vowel of the longest word in the middle 50% of the sentence. Vowel choice prefers non-schwa, then length-marked vowels, then the vowel nearest the word's midpoint, so both voices independently pick the same nucleus. Before fading, Praat/parselmouth measures both voices' F0 at the target vowel; candidate words are skipped if the mismatch exceeds 3 semitones, and voice B is pitch-shifted with PSOLA when the residual exceeds 0.5 semitones.
>
> **Stitch mode:** utterances are joined with a 400 ms silence gap; no pitch matching is applied.
>
> ---
>
> #### 🧪 `human_plausibility`
>
> **Dimension:** `human_plausibility`  
> **Engines:** GPT-5 + Cartesia sonic-3 + DSP  
> **Output:** `results/human_plausibility/{id}_{word}/`
>
> | Step | What happens |
> |---|---|
> | **Phase 1** | GPT-5 picks artifact location and severity for score 0. The artifact type is pre-selected in code from 8 categories, so the model cannot substitute a different one. |
> | **Phase 2** | Cartesia synthesizes the clean sentence once |
> | **Phase 3** | `shared/artifact_utils.py` applies the recipe to the clean WAV to create score 0 |
> | **Write outputs** | Writes `manifest.json`, clean score-1 audio, and degraded score-0 audio |
>
> Both versions keep identical `sentence` fields in the manifest; only the audio differs. Artifact types include clipping, noise, glitches, spectral smearing, and related post-processing degradations.

</details>

## Running

```bash
# Single dimension (default dataset: Mozilla)
uv run python pipeline.py --dimension lexical_stress
uv run python pipeline.py --dimension phonetic_accuracy
uv run python pipeline.py --dimension prosodic_stress
uv run python pipeline.py --dimension speech_rate
uv run python pipeline.py --dimension speaker_identity_consistency

# Intonation — uses EmergentTTS-Eval Questions category from HuggingFace
uv run python pipeline.py --dimension intonation --dataset emergent --category questions

# Expressiveness — Harvard sentences (excluded from --dimension all; run separately)
uv run python pipeline.py --dimension expressiveness --counts 50,20,30

# Prosodic boundary — preset dataset (excluded from --dimension all; run separately)
uv run python pipeline.py --dimension prosodic_boundary
uv run python pipeline.py --dimension prosodic_boundary --strategy omit_punctuation

# All other dimensions concurrently (excludes expressiveness and prosodic_boundary — run those separately)
uv run python pipeline.py --dimension all

# Use Harvard Sentences dataset (word-level dimensions)
uv run python pipeline.py --dimension lexical_stress --dataset harvard
uv run python pipeline.py --dimension all --dataset harvard

# Use a custom utterances file
uv run python pipeline.py --dimension lexical_stress --utterances path/to/custom.json
```

**Datasets:**

| Name | Flag | Sentences | Source |
|------|------|-----------|--------|
| Mozilla Common Voice | `--dataset mozilla` *(default)* | 100 | CC-0 |
| Harvard Sentences | `--dataset harvard` | 720 | IEEE, public domain |
| EmergentTTS-Eval | `--dataset emergent` | varies by category | [bosonai/EmergentTTS-Eval](https://huggingface.co/datasets/bosonai/EmergentTTS-Eval) |

**Common options:**

```bash
# Generate multiple independent batches (each picks fresh utterances)
python pipeline.py --dimension lexical_stress --samples 3

# Fix a reproducible utterance selection
python pipeline.py --dimension lexical_stress --num 32 --seed 42

# Override voice or model (word-level dimensions only)
python pipeline.py --dimension phonetic_accuracy --voice <uuid> --model sonic-2

# Speech rate: PSOLA-only mode (single Cartesia gen, all speed changes via PSOLA)
python pipeline.py --dimension speech_rate --psola-only

# Intonation: limit sentence length and generate only specific scores
python pipeline.py --dimension intonation --dataset emergent --category questions \
    --num 20 --max-words 15 --only-scores 1,2

# Generate only one score, for any dimension
python pipeline.py --dimension speech_rate --only-scores 1
```

### Generating a specific score

`--only-scores N[,N]` restricts which score(s) are delivered, for any dimension — not just the ones with a naturally single-score-per-item model. Valid values are dimension-specific (see the pipeline table above; `prosodic_boundary`'s scale also depends on `--strategy`). With `--dimension all`, a dimension whose scale doesn't include any requested value is skipped with a log line rather than erroring, since different dimensions use different scales.

Most dimensions render only what you asked for with no extra cost. A few have an internal "anchor" dependency and still pay part of the generation cost even when you only want a derived score — the unwanted intermediate is generated, then discarded (not written to the output folder or the manifest):

| Dimension | Anchor cost paid when filtering |
|---|---|
| `speech_rate` | Score 1 is derived from score 2's audio, which is derived from score 3's duration — requesting score 1 alone still generates 2 and 3 internally |
| `human_plausibility` | Score 0 needs the clean (score 1) audio as its DSP input; requesting score 1 alone skips the OpenAI recipe call entirely |
| `speaker_identity_consistency` | Score 0 needs both voices rendered; requesting only score 1 renders voice A alone (skips voice B and the crossfade/stitch computation) |

All other dimensions (`lexical_stress`, `phonetic_accuracy`, `prosodic_stress`, `intonation`, `expressiveness`, `prosodic_boundary`) either have no anchor dependency or already generate one score per item.

**Cache management:**

```bash
# Rebuild cache from WAV files on disk (fixes a stale or missing cache)
python pipeline.py --dimension lexical_stress --rebuild-cache

# Reset cache to allow re-using already-seen utterances
python pipeline.py --dimension lexical_stress --reset-cache
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dimension` / `-d` | *(required)* | `lexical_stress`, `phonetic_accuracy`, `prosodic_stress`, `speech_rate`, `speaker_identity_consistency`, `intonation`, `human_plausibility`, `expressiveness`, `prosodic_boundary`, or `all` (excludes `expressiveness` and `prosodic_boundary`) |
| `--dataset` | `mozilla` | Input utterances: `mozilla`, `harvard`, or `emergent` |
| `--category` | — | `emergent` only — filter by category (e.g. `questions`) |
| `--utterances` | — | Path to a custom utterances JSON file (overrides `--dataset`) |
| `--samples` / `-s` | `1` | Independent batches; each picks fresh utterances appended to the same output dir |
| `--num` / `-n` | per-dimension | Utterances per batch. For `speaker_identity_consistency`: number of **paired** samples (each = 1 score-0 + 1 score-1 WAV); default 8 = one full alternation cycle |
| `--voice` | per-dimension | Cartesia voice UUID (ignored for `speaker_identity_consistency`, `intonation`, and `--dimension all`) |
| `--model` | `sonic-3` | Cartesia model ID |
| `--deployment` | `gpt-5` | Azure OpenAI deployment name |
| `--output` | `results/` | Base output dir; writes to `{dir}/{dimension}/` |
| `--openai-concurrency` | `5` | Max parallel OpenAI requests |
| `--tts-concurrency` | `10` | Max parallel TTS requests (Cartesia or ElevenLabs) |
| `--seed` | — | Random seed for reproducible utterance selection |
| `--max-words` | — | `intonation` only — discard sentences longer than N words |
| `--min-words` | — | `intonation` only — discard sentences shorter than N words |
| `--only-scores` | — | Comma-separated scores to generate for **any** dimension, e.g. `1,2`. See [Generating a specific score](#generating-a-specific-score) below |
| `--stack-rate` | `0.0` | `intonation` only — probability of composing two score-1 F0 strategies in sequence |
| `--counts` | `50,20,30` | `expressiveness` only — comma-separated item counts for score 1, 2, 3 (e.g. `50,20,30`); zeroed automatically for scores excluded by `--only-scores` |
| `--strategy` | `preset` | `prosodic_boundary` only — `omit_spaces`, `omit_punctuation`, `extra_punctuation`, or `preset` (uses fixed 60-sentence dataset) |
| `--rebuild-cache` | — | Rebuild utterance cache from WAV files on disk |
| `--reset-cache` | — | Clear utterance cache for the selected dimension(s) |
| `--psola-only` | — | (speech_rate only) All speed variants via PSOLA from a single Cartesia generation |

## IPA Format (word-level dimensions)

Phonemes use Cartesia's pipe-separated format. For example, *banana*:

```
<<b|ə|ˈ|n|æ|n|ə>>
```

- `ˈ` precedes the stressed syllable

## Project Structure

```
BeyondNaturalnessProbe/
├── pipeline.py                  # Single entry point for all dimensions
├── prompts/                     # One GPT-5 prompt per GPT-driven dimension/strategy
│   ├── lexical_stress.txt
│   ├── phonetic_accuracy.txt
│   ├── prosodic_stress.txt
│   ├── human_plausibility.txt
│   └── prosodic_boundary_extra_punctuation.txt
├── data/
│   ├── utterances.json          # 100 English sentences (Mozilla Common Voice, CC-0)
│   ├── harvard_sentences.json   # 720 Harvard Sentences (IEEE, public domain)
│   └── prosodic_boundary_preset*.json  # Curated 60-sentence preset for prosodic_boundary
├── shared/                      # Shared async utilities
│   ├── clients.py               # API client factories + env validation
│   ├── emergent_dataset.py      # Loader for bosonai/EmergentTTS-Eval (HuggingFace)
│   ├── intonation.py            # F0 strategies + Praat pipeline for intonation
│   ├── openai_utils.py          # GPT prompt building & response parsing
│   ├── pipeline.py              # Two-phase pipeline runner + speech rate pipeline
│   ├── tts_utils.py             # Cartesia + ElevenLabs TTS wrappers
│   ├── audio_utils.py           # WAV duration measurement, PSOLA time-stretching
│   ├── speaker_identity.py      # Speaker-identity audio primitives + runners
│   ├── artifact_pipeline.py     # human_plausibility: GPT recipe + Cartesia + DSP orchestration
│   ├── artifact_utils.py        # human_plausibility: DSP artifact implementations
│   ├── prosodic_strategies.py   # Strategy dispatch for prosodic_boundary
│   ├── expressiveness.py        # Cartesia synthesis for expressiveness
│   ├── flatten_expressiveness.py # PSOLA F0/intensity flattening (expressiveness score 2)
│   └── utterances.py            # Utterance loading, candidate filtering, output naming
├── tests/                       # pytest — one runner (mocked TTS/OpenAI clients) per dimension, plus dispatch/env logic
└── results/                     # Generated audio (not committed)
    ├── lexical_stress/
    │   └── {dataset}_{id:03d}_{word}/
    │       ├── manifest.json
    │       ├── {dataset}_{id:03d}_{word}_0.wav
    │       └── {dataset}_{id:03d}_{word}_1.wav
    ├── phonetic_accuracy/
    │   └── {dataset}_{id:03d}_{word}/
    │       ├── manifest.json
    │       ├── {dataset}_{id:03d}_{word}_1.wav
    │       ├── {dataset}_{id:03d}_{word}_2.wav
    │       └── {dataset}_{id:03d}_{word}_3.wav
    ├── prosodic_stress/
    │   └── {dataset}_{id:03d}_stress/            # "stress" is a fixed placeholder word
    │       ├── manifest.json
    │       ├── {dataset}_{id:03d}_stress_0.wav   (function words quoted)
    │       └── {dataset}_{id:03d}_stress_1.wav   (original sentence)
    ├── speech_rate/
    │   └── {id:03d}_{fast|slow}/
    │       ├── manifest.json
    │       ├── score_1.wav
    │       ├── score_2.wav
    │       └── score_3.wav
    ├── speaker_identity_consistency/
    │   ├── {id:03d}_{gender}_{order}/               # crossfade sample
    │   │   ├── manifest.json                          # mode: "crossfade"
    │   │   ├── 0.wav   # inconsistent (voice_a→voice_b vowel crossfade, pitch-matched)
    │   │   └── 1.wav   # consistent (voice_a only)
    │   └── {id_a:03d}_{id_b:03d}_{gender}_{order}/  # stitch sample
    │       ├── manifest.json                          # mode: "stitch"
    │       ├── 0.wav   # inconsistent (voice_a U_a + voice_b U_b)
    │       └── 1.wav   # consistent (voice_a speaks both U_a + U_b)
    ├── intonation/
    │   └── {dataset}_{id:03d}/
    │       ├── manifest.json
    │       ├── {dataset}_{id:03d}_1.wav   (F0 distorted — egregious)
    │       ├── {dataset}_{id:03d}_2.wav   (subdued terminal rise)
    │       └── {dataset}_{id:03d}_3.wav   (verbatim ElevenLabs — anchor)
    ├── expressiveness/
    │   └── {score}_{id:03d}/
    │       ├── manifest.json
    │       └── {score}_{id:03d}.wav       (single render per sentence)
    ├── prosodic_boundary/
    │   └── {dataset}_{id:03d}/
    │       ├── manifest.json
    │       ├── {dataset}_{id:03d}_1.wav   (boundary omission)
    │       ├── {dataset}_{id:03d}_2.wav   (boundary addition)
    │       └── {dataset}_{id:03d}_3.wav   (canonical)
    └── human_plausibility/
        └── {id:03d}_{word}/
            ├── manifest.json
            ├── {word}_0.wav   (1-2 digital artifacts applied)
            └── {word}_1.wav   (clean reference)
```

Folder naming distinguishes the two modes by the number of utterance IDs in the prefix (single id → crossfade, two ids → stitch); `mode` is also stored in each manifest for unambiguous aggregation.

## 📝 Citation

If you use our benchmark, code or artifacts in your research, please cite our work:

```bibtex
@article{bamgbose2026beyond,
  title={Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions},
  author={Bamgbose, Oluwanifemi and Rosen, Simon and Shah, Jash and Brin, Lindsay Devon and Nguyen, Hoang H and Koelzer, Anke and Hansen, Rachel and Bogavelli, Tara and Riols, Fanny},
  journal={arXiv preprint arXiv:2608.09930},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
