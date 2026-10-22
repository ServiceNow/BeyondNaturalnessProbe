"""
Intonation pipeline (F0 contour manipulation via Praat/PSOLA).

For each utterance:
  Phase 1 — synthesize the input verbatim via ElevenLabs → score-3 anchor WAV.
  Phase 2 — load that base through Parselmouth and apply F0 manipulations
            to produce the score-1 (egregious) and score-2 (mild) variants.

Output layout (folder and file prefix match — see ``output_names``):

  output_dir/{dataset}_{id:03d}/
      {dataset}_{id:03d}_1.wav   (F0-manipulated, egregious)
      {dataset}_{id:03d}_2.wav   (F0-manipulated, subdued terminal rise)
      {dataset}_{id:03d}_3.wav   (clean ElevenLabs synthesis — the anchor)
      manifest.json

All three scores share the same verbatim ElevenLabs base, so the only
variable is the Praat F0 manipulation applied to it.

Severity ordering:

  Score 3 — verbatim ElevenLabs output: natural, correctly rising question.

  Score 2 — ``_subdued_terminal_rise``: the terminal rise (last ~30%) is
  compressed toward the mean to 15–30% of its original excursion. The rise
  is audibly present but insufficient — the listener hears "almost a question."
  Works for all question types including WH-questions (which have a natural
  fall that is also suppressed).

  Score 1 (egregious) — strategies from ``F0_SCORE_1_STRATEGIES``: wide-range
  distortions (mean ±70–130 Hz) that are plainly wrong regardless of the
  original contour. Inversion and reversal include amplification so they
  don't collapse to near-no-ops on prosody-poor bases.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
from elevenlabs.client import AsyncElevenLabs

from .tts_utils import generate_tts_elevenlabs
from .utterances import output_names


# ─── Praat constants ─────────────────────────────────────────────────────────
F0_FALLBACK_HZ       = 120.0
F0_CLIP_MIN_HZ       = 60.0
F0_CLIP_MAX_HZ       = 320.0
N_CONTROL_POINTS     = 45
_PITCH_ANALYSIS_FLOOR_HZ = 75.0   # higher than manipulation floor to avoid octave errors in tracking
_PITCH_MANIP_FLOOR_HZ    = 50.0   # lower bound for PSOLA resynthesis control points
_PITCH_CEIL_HZ       = 600.0
_MANIP_TIME_STEP     = 0.01


def _sample_original_f0(sound: parselmouth.Sound, n_points: int):
    duration = sound.get_total_duration()
    times = np.linspace(0.05, duration - 0.05, n_points)
    pitch = call(sound, "To Pitch", 0.0, _PITCH_ANALYSIS_FLOOR_HZ, _PITCH_CEIL_HZ)
    f0 = np.array([
        call(pitch, "Get value at time", float(t), "Hertz", "Linear")
        for t in times
    ])
    f0 = np.where(np.isnan(f0), F0_FALLBACK_HZ, f0)
    return times, f0, duration


# =============================================================================
# Score 1 — HIGH-IMPACT strategies
# Taxonomy from the wonky_intonation.py reference; magnitudes re-tuned for a
# reliable severity floor while staying in-register (see module docstring).
# =============================================================================

def _random_walk(f0, mean_f0, times, duration, n):
    # σ=35. Cumulative peaks ~150–250 Hz — assertive drunken drift within the
    # 320 Hz ceiling on a male voice.
    walk = np.cumsum(np.random.randn(n) * 35)
    return f0 + walk


def _invert(f0, mean_f0, times, duration, n):
    # Inversion + ×1.3–1.6 amplification so the effect is audible even when
    # the original contour is relatively flat. Still in register.
    new_f0 = 2 * mean_f0 - f0
    new_f0 = mean_f0 + (new_f0 - mean_f0) * random.uniform(1.3, 1.6)
    new_f0 += np.random.randn(n) * 25
    return new_f0


def _exaggerate(f0, mean_f0, times, duration, n):
    # 3.5–5.0× — clearly separated from the score-2 mild_exaggerate range so
    # severity ordering is unambiguous.
    scale = random.uniform(3.5, 5.0)
    return mean_f0 + (f0 - mean_f0) * scale


def _flatten_then_spike(f0, mean_f0, times, duration, n):
    # 0.7× mean baseline with 3–5 spikes at 0.5–1.6× mean — baseline dip and
    # spike excursions both ~±40 Hz for a male voice, plainly audible.
    new_f0 = np.full(n, mean_f0 * 0.7)
    for _ in range(random.randint(3, 5)):
        idx = random.randint(0, n - 1)
        new_f0[idx] = mean_f0 * random.uniform(0.5, 1.6)
    return new_f0


def _sine_wave(f0, mean_f0, times, duration, n):
    # 3–5 Hz modulation at ±40–65 Hz — clearly audible wobble on the base.
    freq = random.uniform(3, 5)
    amplitude = random.uniform(40, 65)
    phase = random.uniform(0, 2 * np.pi)
    return f0 + amplitude * np.sin(2 * np.pi * freq * times / duration + phase)


def _reverse(f0, mean_f0, times, duration, n):
    # Reversal + ×1.2–1.5 amplification so wrong-direction motion is plainly
    # audible even on flat originals.
    new_f0 = f0[::-1].copy()
    new_f0 = mean_f0 + (new_f0 - mean_f0) * random.uniform(1.2, 1.5)
    new_f0 += np.random.randn(n) * 25
    return new_f0


def _staircase(f0, mean_f0, times, duration, n):
    # 3–5 levels within mean ±70 Hz. Endpoints anchor to the extremes and the
    # remaining level(s) are sampled uniformly in between — guarantees every
    # draw spans the full range (a pure-random draw can cluster near the mean).
    n_steps = random.randint(3, 5)
    low, high = mean_f0 - 70, mean_f0 + 70
    middle = list(np.random.uniform(low, high, n_steps - 2))
    anchors = [low, high] + middle
    random.shuffle(anchors)
    levels = np.array(anchors)
    step_size = max(n // n_steps, 1)
    new_f0 = np.zeros(n)
    for i in range(n):
        new_f0[i] = levels[min(i // step_size, n_steps - 1)]
    return new_f0


def _random_jumps(f0, mean_f0, times, duration, n):
    # 50% of frames replaced within mean ±90 Hz — dense enough that the chaos
    # is unmistakable regardless of original contour.
    new_f0 = f0.copy()
    for i in range(n):
        if random.random() < 0.50:
            new_f0[i] = random.uniform(mean_f0 - 90, mean_f0 + 90)
    return new_f0


def _question_everything(f0, mean_f0, times, duration, n):
    # Wide in-register sweep: ~mean−40 → mean+130 Hz. The 320 Hz ceiling will
    # clip the top for high-mean voices — that clipping leaves a plausibly
    # emphatic rising contour.
    start = mean_f0 + random.uniform(-50, -25)
    end   = mean_f0 + random.uniform(80, 130)
    new_f0 = np.linspace(start, end, n)
    new_f0 += np.random.randn(n) * 15
    return new_f0


def _monotone_drift(f0, mean_f0, times, duration, n):
    # Wide drift range and heavier wobble for an audibly wrong slope.
    start = mean_f0 + random.uniform(-60, -30)
    end   = mean_f0 + random.uniform(70, 120)
    if random.random() < 0.5:
        start, end = end, start
    new_f0 = np.linspace(start, end, n)
    new_f0 += np.random.randn(n) * 30
    return new_f0


F0_SCORE_1_STRATEGIES: list[dict] = [
    {"id": "reverse",             "fn": _reverse,             "description": "Contour time-reversed, scaled 1.2–1.5× and perturbed by ±25 Hz noise — prosodic motion going the wrong way and amplified."},
    {"id": "invert",              "fn": _invert,              "description": "Contour inverted around the mean and scaled 1.3–1.6× with ±25 Hz noise — motion reversed and amplified."},
    {"id": "question_everything", "fn": _question_everything, "description": "Linear rise from roughly mean−40 Hz to mean+130 Hz across the utterance — pronounced in-register rising sweep."},
    {"id": "exaggerate",          "fn": _exaggerate,          "description": "Pitch range exaggerated 3.5–5.0×, producing theatrical over-expressive prosody (ceiling clips extreme peaks)."},
    {"id": "staircase",           "fn": _staircase,           "description": "3–5 distinct pitch levels within mean ±70 Hz — plainly quantized, jumpy contour."},
    {"id": "random_jumps",        "fn": _random_jumps,        "description": "~50% of frames replaced with a random pitch within mean ±90 Hz — constant, audible wrong-note chaos."},
    {"id": "flatten_then_spike",  "fn": _flatten_then_spike,  "description": "Pronounced flat baseline (0.7× mean F0) punctuated by 3–5 emphasis bumps at 0.5–1.6× mean — clearly robotic with jarring accents."},
    {"id": "sine_wave",           "fn": _sine_wave,           "description": "Slow sinusoidal wobble (3–5 Hz, ±40–65 Hz amplitude) overlaid on the original — audibly unsteady, seasick contour."},
    {"id": "monotone_drift",      "fn": _monotone_drift,      "description": "Gradual linear drift across ~130–180 Hz within the voice's range (direction randomized), with ±30 Hz wobble."},
    {"id": "random_walk",         "fn": _random_walk,         "description": "Cumulative random walk (σ≈35 Hz) superimposed on the original contour — strong, drunken pitch drift."},
]


# =============================================================================
# Score 2 — LOW-IMPACT strategies
# Taxonomy from the wonky_intonation_low.py reference; magnitudes adjusted to
# stay clearly audible while remaining below the score-1 floor.
# =============================================================================

def _subdued_terminal_rise(f0, mean_f0, times, duration, n):
    # Compress only the terminal portion (last ~30%) toward the mean.
    # The question rise is present but peaks at only 15–30% of its natural
    # excursion — listener hears "almost a question" without full commitment.
    # Works for all question types: WH-questions (natural fall also suppressed),
    # and YN-questions (rise flattened but not reversed).
    out = f0.copy()
    start = int(n * random.uniform(0.65, 0.75))
    attenuation = random.uniform(0.15, 0.30)
    out[start:] = mean_f0 + (f0[start:] - mean_f0) * attenuation
    return out


F0_SCORE_2_STRATEGY: dict = {
    "id": "subdued_terminal_rise",
    "fn": _subdued_terminal_rise,
    "description": "Terminal rise compressed to 15–30% of its natural excursion — the question intonation is present but insufficiently high.",
}


# ─── Praat resynthesis entry point ───────────────────────────────────────────

def apply_f0_manipulation(sound: parselmouth.Sound, strategy: dict) -> parselmouth.Sound:
    """Rewrite the F0 contour of *sound* per *strategy* and return the PSOLA resynthesis."""
    manipulation = call(sound, "To Manipulation", _MANIP_TIME_STEP, _PITCH_MANIP_FLOOR_HZ, _PITCH_CEIL_HZ)
    pitch_tier = call(manipulation, "Extract pitch tier")
    duration = sound.get_total_duration()
    call(pitch_tier, "Remove points between", 0, duration)

    times, original_f0, _ = _sample_original_f0(sound, N_CONTROL_POINTS)
    mean_f0 = float(np.mean(original_f0))
    new_f0 = strategy["fn"](original_f0, mean_f0, times, duration, N_CONTROL_POINTS)
    new_f0 = np.clip(new_f0, F0_CLIP_MIN_HZ, F0_CLIP_MAX_HZ)

    for t, f in zip(times, new_f0):
        call(pitch_tier, "Add point", float(t), float(f))

    call([pitch_tier, manipulation], "Replace pitch tier")
    return call(manipulation, "Get resynthesis (overlap-add)")


# ─── Phase 1 — base ElevenLabs synthesis ─────────────────────────────────────

async def _synthesize_base(
    client: AsyncElevenLabs,
    item: dict,
    folder: Path,
    file_prefix: str,
    voice_id: str,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    *,
    score3_requested: bool = True,
) -> Path | None:
    async with semaphore:
        folder.mkdir(parents=True, exist_ok=True)
        # Use _3.wav when score 3 is part of the run (it IS the deliverable).
        # Use _base.wav as a scratch file when score 3 is not requested — Praat
        # still needs a base to manipulate but we shouldn't leave a _3.wav on
        # disk that won't appear in any manifest.
        wav_name = f"{file_prefix}_3.wav" if score3_requested else f"{file_prefix}_base.wav"
        wav_path = folder / wav_name

        if wav_path.exists():
            print(f"[{index}/{total}] Utt {item['id']:>3}: base exists, skipping synthesis", flush=True)
            return wav_path

        t0 = time.time()
        try:
            await generate_tts_elevenlabs(client, item["text"], wav_path, voice_id=voice_id)
            print(
                f"[{index}/{total}] Utt {item['id']:>3}: base synthesized ({time.time()-t0:.1f}s) -> {wav_name}",
                flush=True,
            )
            return wav_path
        except Exception as exc:
            print(f"[{index}/{total}] Utt {item['id']:>3}: BASE SYNTH FAILED: {exc}", flush=True)
            return None


# ─── Phase 2 — F0 manipulation (synchronous; Praat is CPU-bound) ─────────────

def _apply_and_save(
    base_path: Path,
    score: int,
    strategies: list[dict],
    folder: Path,
    file_prefix: str,
) -> str | None:
    """Apply *strategies* in order (composing their effects) and save."""
    wav_name = f"{file_prefix}_{score}.wav"
    wav_path = folder / wav_name
    if wav_path.exists():
        return wav_name
    ids = "+".join(s["id"] for s in strategies)
    try:
        sound = parselmouth.Sound(str(base_path))
        for strat in strategies:
            sound = apply_f0_manipulation(sound, strat)
        sound.save(str(wav_path), "WAV")
        return wav_name
    except Exception as exc:
        print(f"    Score {score} [{ids}] FAILED: {exc}", flush=True)
        return None


def _draw_score_1_strategies(stack_rate: float) -> list[dict]:
    """
    Pick strategies for a score-1 sample.

    With probability *stack_rate*, return two distinct F0 strategies to
    compose in sequence. Otherwise return a single strategy. Stacking is
    off by default (stack_rate=0.0) because it widens score-1 variance —
    some stacked samples end up markedly more severe than singletons,
    which breaks consistency across the score-1 bucket.
    """
    primary = random.choice(F0_SCORE_1_STRATEGIES)
    if random.random() >= stack_rate:
        return [primary]
    remaining = [s for s in F0_SCORE_1_STRATEGIES if s["id"] != primary["id"]]
    if not remaining:
        return [primary]
    return [primary, random.choice(remaining)]


# ─── Public entry point ──────────────────────────────────────────────────────

async def run_intonation_pipeline(
    dimension_name: str,
    items: list[dict],
    output_dir: Path,
    elevenlabs_client: AsyncElevenLabs,
    voice_id: str,
    *,
    tts_concurrency: int = 10,
    stack_rate: float = 0.0,
    dataset_tag: str = "",
    only_scores: set[int] | None = None,
) -> list[int]:
    """
    Run the intonation pipeline over *items* and return the list of utterance
    IDs that were produced successfully.

    ``only_scores`` restricts which score WAVs are generated (e.g. ``{2}``
    skips score-1 Praat manipulation and score-3 verbatim synthesis entirely).

    All three scores derive from the same verbatim ElevenLabs base (score 3):
      Score 3 — verbatim ElevenLabs synthesis (the anchor).
      Score 2 — Praat subdued_terminal_rise on score-3 base: rise present
                but compressed to 15–30% of natural excursion.
      Score 1 — Praat egregious F0 distortion on score-3 base.
    """
    sep = "=" * 60
    want = only_scores or {1, 2, 3}
    output_dir.mkdir(parents=True, exist_ok=True)
    layouts: dict[int, tuple[Path, str]] = {}
    for item in items:
        folder_name, file_prefix = output_names(item["id"], item.get("target_word", ""), dataset_tag)
        layouts[item["id"]] = (output_dir / folder_name, file_prefix)

    tts_sem = asyncio.Semaphore(tts_concurrency)

    # Phase 1 — verbatim synthesis (score-3 anchor + Praat base for 1 and 2).
    print(f"\n{sep}")
    print(f"Phase 1: Synthesizing base audio for {dimension_name} via ElevenLabs")
    print(sep)
    base_paths = list(await asyncio.gather(*(
        _synthesize_base(
            elevenlabs_client, item, layouts[item["id"]][0], layouts[item["id"]][1],
            voice_id, i, len(items), tts_sem,
            score3_requested=(3 in want),
        )
        for i, item in enumerate(items, 1)
    )))

    # Phase 2 — per-utterance Praat manipulations (scores 1 and 2).
    if want & {1, 2}:
        print(f"\n{sep}")
        print("Phase 2: Applying Praat F0 manipulations")
        print(sep)

    s2_strategy = F0_SCORE_2_STRATEGY

    successful_ids: list[int] = []

    for item, base_path in zip(items, base_paths):
        folder, file_prefix = layouts[item["id"]]
        print(f"\n[{item['id']:>3}] \"{item['text']}\"")

        # Score 3 — verbatim
        wav_3 = base_path.name if (3 in want and base_path and base_path.exists()) else None
        if 3 in want and wav_3 is None:
            print(f"  Skipping score 3 — no base audio.")

        # Score 2 — subdued terminal rise via Praat on verbatim base
        wav_2: str | None = None
        s2_description = s2_strategy["description"]
        if 2 in want:
            if base_path and base_path.exists():
                t0 = time.time()
                wav_2 = _apply_and_save(base_path, 2, [s2_strategy], folder, file_prefix)
                print(f"  Score 2 [subdued_terminal_rise] -> {wav_2 or '(failed)'}  ({time.time()-t0:.1f}s)")
            else:
                print(f"  Score 2 — no base audio, skipping.")

        # Score 1 — egregious Praat distortion on verbatim base
        wav_1: str | None = None
        s1_ids: list[str] = []
        s1_description = ""
        if 1 in want:
            if base_path and base_path.exists():
                s1_list = _draw_score_1_strategies(stack_rate)
                s1_ids = [s["id"] for s in s1_list]
                s1_description = " → then → ".join(s["description"] for s in s1_list)
                t0 = time.time()
                wav_1 = _apply_and_save(base_path, 1, s1_list, folder, file_prefix)
                s1_tag = "+".join(s1_ids) + (" (stacked)" if len(s1_ids) > 1 else "")
                print(f"  Score 1 [{s1_tag}] -> {wav_1 or '(failed)'}  ({time.time()-t0:.1f}s)")
            else:
                print(f"  Score 1 — no base audio, skipping.")

        if 3 in want:
            print(f"  Score 3 (verbatim)    -> {wav_3 or '(skipped)'}")

        versions = []
        if 1 in want and wav_1:
            versions.append({
                "score": 1,
                "strategies": s1_ids,
                "stacked": len(s1_ids) > 1,
                "error_description": s1_description,
                "transcription": "",
                "original_sentence": item["text"],
                "sentence": item["text"],
                "audio_file": wav_1,
            })
        if 2 in want and wav_2:
            versions.append({
                "score": 2,
                "strategies": ["subdued_terminal_rise"],
                "stacked": False,
                "error_description": s2_description,
                "transcription": "",
                "original_sentence": item["text"],
                "sentence": item["text"],
                "audio_file": wav_2,
            })
        if 3 in want and wav_3:
            versions.append({
                "score": 3,
                "strategies": [],
                "stacked": False,
                "error_description": "Verbatim ElevenLabs output. Canonical intonation.",
                "transcription": "",
                "original_sentence": item["text"],
                "sentence": item["text"],
                "audio_file": wav_3,
            })

        entry = {
            "utterance_id": item["id"],
            "original_text": item["text"],
            "target_word": item.get("target_word", ""),
            "generation_method": "f0_praat",
            "versions": versions,
            "output_dir": str(folder.relative_to(output_dir)),
        }
        (folder / "manifest.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False))

        produced = {s for s, w in [(1, wav_1), (2, wav_2), (3, wav_3)] if w}
        if want.issubset(produced):
            successful_ids.append(item["id"])

    print(f"\n{sep}")
    print(f"Done!  Output directory: {output_dir}")
    print(f"Produced: {len(successful_ids)}/{len(items)} utterances ({len(successful_ids) * len(want)} audio files)")
    return successful_ids
