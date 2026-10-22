"""
Two-phase pipeline runner shared by the word-level dimensions.

Phase 1 — OpenAI variant generation (concurrent, semaphore-limited)
Phase 2 — TTS synthesis               (concurrent, semaphore-limited)

Phase 2 routes to Cartesia by default; pass ``elevenlabs_client`` to use
ElevenLabs instead (mutually exclusive with ``cartesia_client``). Each
dimension calls run_two_phase_pipeline() with its own config and a pre-built
list of items ({id, text, target_word}[, prompt_vars]).

The intonation dimension bypasses this runner entirely — see
``shared.intonation.run_intonation_pipeline``.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

from openai import AsyncAzureOpenAI, AsyncOpenAI
from cartesia import AsyncCartesia
from elevenlabs.client import AsyncElevenLabs

from .openai_utils import build_prompt, call_openai
from .tts_utils import generate_tts, generate_tts_elevenlabs, DEFAULT_OUTPUT_FORMAT
from .audio_utils import get_wav_duration, psola_stretch
from .utterances import output_names


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

async def _process_openai_item(
    openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment: str,
    prompt_template: str,
    item: dict,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    dataset_tag: str,
) -> dict | None:
    """
    Generate phoneme variants for one utterance via OpenAI (standard or Azure).

    If a manifest.json already exists for this utterance (from a prior
    interrupted run), it is loaded directly and the OpenAI call is skipped.

    Returns a manifest entry dict, or None if the call fails or the
    model returns an empty versions list (e.g. monosyllabic target).
    """
    folder_name, _ = output_names(item["id"], item["target_word"], dataset_tag)
    manifest_path = output_dir / folder_name / "manifest.json"

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("versions"):
            print(
                f"\n[{index}/{total}] Utterance {item['id']}: \"{item['target_word']}\" "
                "— skipped Phase 1 (manifest already exists)",
                flush=True,
            )
            return existing

    async with semaphore:
        prompt_vars = item.get("prompt_vars", {})
        prompt = build_prompt(prompt_template, item["text"], item["target_word"], **prompt_vars)
        t0 = time.time()
        lines = [f"[{index}/{total}] Utterance {item['id']}: \"{item['target_word']}\""]

        try:
            result = await call_openai(openai_client, deployment, prompt)
        except Exception as exc:
            elapsed = time.time() - t0
            lines += [
                f"  OpenAI ({deployment})... FAILED ({elapsed:.1f}s)",
                f"  ERROR: {exc}",
            ]
            print("\n" + "\n".join(lines), flush=True)
            return None

        elapsed = time.time() - t0
        versions = result.get("versions", [])

        if not versions:
            lines.append(
                f"  OpenAI ({deployment})... done ({elapsed:.1f}s) — "
                "skipped (empty versions; target may be monosyllabic or unsuitable)"
            )
            print("\n" + "\n".join(lines), flush=True)
            return None

        lines.append(f"  OpenAI ({deployment})... done ({elapsed:.1f}s)")
        for v in versions:
            lines.append(f"  Score {v['score']}: {v['error_description'][:80]}")

        print("\n" + "\n".join(lines), flush=True)
        return {
            "utterance_id": item["id"],
            "original_text": item["text"],
            "target_word": item["target_word"],
            "versions": versions,
        }


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

async def _process_tts_entry(
    cartesia_client: AsyncCartesia | None,
    entry: dict,
    output_dir: Path,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    *,
    model_id: str,
    voice_id: str,
    output_format: dict,
    dataset_tag: str,
    elevenlabs_client: AsyncElevenLabs | None = None,
    elevenlabs_voice_id: str | None = None,
    elevenlabs_model: str = "eleven_multilingual_v2",
) -> None:
    """
    Synthesize every score variant for one utterance and write the manifest.

    Routes to ElevenLabs when *elevenlabs_client* is provided; otherwise Cartesia.

    Output layout (folder and file prefix match — see ``output_names``):
        output_dir/{dataset}_{uid:03d}[_{word}]/
            {dataset}_{uid:03d}[_{word}]_{score}.wav   (one per version)
            manifest.json
    """
    async with semaphore:
        uid = entry["utterance_id"]
        folder_name, file_prefix = output_names(uid, entry["target_word"], dataset_tag)
        folder = output_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        entry["output_dir"] = str(folder.relative_to(output_dir))
        label = entry["target_word"] or file_prefix
        lines = [f"[{index}/{total}] Utterance {uid} — \"{label}\""]

        for v in entry["versions"]:
            score = v["score"]
            wav_name = f"{file_prefix}_{score}.wav"
            wav_path = folder / wav_name
            t0 = time.time()

            if wav_path.exists():
                v["audio_file"] = wav_name
                lines.append(f"  Score {score}... skipped (already exists) -> {wav_name}")
                continue

            try:
                if elevenlabs_client is not None:
                    await generate_tts_elevenlabs(
                        elevenlabs_client,
                        v["sentence"],
                        wav_path,
                        voice_id=elevenlabs_voice_id,
                        model_id=elevenlabs_model,
                    )
                else:
                    await generate_tts(
                        cartesia_client,
                        v["sentence"],
                        wav_path,
                        model_id=model_id,
                        voice_id=voice_id,
                        output_format=output_format,
                    )
                v["audio_file"] = wav_name
                lines.append(f"  Score {score}... done ({time.time() - t0:.1f}s) -> {wav_name}")
            except Exception as exc:
                lines.append(f"  Score {score}... FAILED ({time.time() - t0:.1f}s)")
                lines.append(f"    ERROR: {exc}")
                v["audio_file"] = None

        print("\n" + "\n".join(lines), flush=True)

        (folder / "manifest.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False)
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_two_phase_pipeline(
    dimension_name: str,
    items: list[dict],
    prompt_template: str,
    output_dir: Path,
    openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment: str,
    cartesia_client: AsyncCartesia | None = None,
    model_id: str = "sonic-3",
    voice_id: str = "",
    output_format: dict = DEFAULT_OUTPUT_FORMAT,
    *,
    elevenlabs_client: AsyncElevenLabs | None = None,
    elevenlabs_voice_id: str | None = None,
    elevenlabs_model: str = "eleven_multilingual_v2",
    openai_concurrency: int = 5,
    tts_concurrency: int = 10,
    dataset_tag: str = "",
    only_scores: set[int] | None = None,
) -> list[int]:
    """
    Run the full two-phase TTS generation pipeline.

    Exactly one of *cartesia_client* or *elevenlabs_client* must be provided.

    ``only_scores`` restricts which score WAVs are synthesized and kept in the
    manifest (e.g. ``{1}`` skips rendering every other version). Phase 1 (the
    OpenAI call) always generates the full variant set regardless — it's a
    single call per utterance either way, so there's no cost to scope down.
    """
    sep = "=" * 60

    # ── Phase 1: OpenAI ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"Phase 1: Generating {dimension_name} variants via OpenAI")
    print(sep)

    openai_sem = asyncio.Semaphore(openai_concurrency)
    results = await asyncio.gather(*(
        _process_openai_item(
            openai_client, deployment, prompt_template,
            item, i, len(items), openai_sem, output_dir, dataset_tag,
        )
        for i, item in enumerate(items, 1)
    ))
    manifest = [r for r in results if r is not None]

    if only_scores is not None:
        for entry in manifest:
            entry["versions"] = [v for v in entry["versions"] if v["score"] in only_scores]
        manifest = [e for e in manifest if e["versions"]]

    if not manifest:
        print("\nNo variants generated — nothing to synthesize.")
        return []

    # ── Phase 2: TTS synthesis ───────────────────────────────────────────────
    engine_name = "ElevenLabs" if elevenlabs_client is not None else "Cartesia"
    print(f"\n{sep}")
    print(f"Phase 2: Generating TTS audio via {engine_name}")
    print(sep)

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_sem = asyncio.Semaphore(tts_concurrency)
    await asyncio.gather(*(
        _process_tts_entry(
            cartesia_client, entry, output_dir,
            i, len(manifest), tts_sem,
            model_id=model_id,
            voice_id=voice_id,
            output_format=output_format,
            dataset_tag=dataset_tag,
            elevenlabs_client=elevenlabs_client,
            elevenlabs_voice_id=elevenlabs_voice_id,
            elevenlabs_model=elevenlabs_model,
        )
        for i, entry in enumerate(manifest, 1)
    ))

    # ── Summary ──────────────────────────────────────────────────────────────
    successful_ids = [
        e["utterance_id"]
        for e in manifest
        if any(v.get("audio_file") for v in e["versions"])
    ]
    total_files = sum(
        sum(1 for v in e["versions"] if v.get("audio_file"))
        for e in manifest
    )
    print(f"\n{sep}")
    print(f"Done!  Output directory: {output_dir}")
    print(f"\nStructure:")
    print(f"  {output_dir.name}/")
    for entry in manifest:
        folder_name, _ = output_names(entry["utterance_id"], entry["target_word"], dataset_tag)
        print(f"    {folder_name}/")
        print(f"      manifest.json")
        for v in entry["versions"]:
            print(f"      {v.get('audio_file', '(failed)')}")
    print(f"\nTotal audio files generated: {total_files}")
    return successful_ids


# ---------------------------------------------------------------------------
# Speech-rate pipeline
# ---------------------------------------------------------------------------

# Speech-rate generation strategy (hybrid mode):
#   Score 3 — Cartesia speed 1.0 (normal baseline)
#   Score 2 — Cartesia at max deviation (0.6 slow / 1.4 fast), no PSOLA
#   Score 1 — copy of score 2, then PSOLA ±0.3x to push further
#
# Only 2 TTS generations per utterance.  Score 1 reuses score 2's
# audio so the prosody/phrasing is identical — the only difference
# is an additional PSOLA time-stretch.

_DIRECTION_CFG: dict[str, dict] = {
    "fast": {
        "cartesia_speed": 1.4,
        "psola_factor": 0.7,   # compress score 2 by 30% more
        "score2_desc": "Speech rate fast — low negative impact (Cartesia speed 1.4).",
        "score1_desc": "Speech rate very fast — high negative impact (Cartesia 1.4 + PSOLA 0.7x).",
    },
    "slow": {
        "cartesia_speed": 0.6,
        "psola_factor": 1.3,   # stretch score 2 by 30% more
        "score2_desc": "Speech rate slow — low negative impact (Cartesia speed 0.6).",
        "score1_desc": "Speech rate very slow — high negative impact (Cartesia 0.6 + PSOLA 1.3x).",
    },
}


async def _generate_one_wav(
    cartesia_client: AsyncCartesia,
    transcript: str,
    wav_path: Path,
    *,
    model_id: str,
    voice_id: str,
    output_format: dict,
    speed: float,
) -> float | None:
    """Synthesize one WAV and return its duration, or None on failure."""
    await generate_tts(
        cartesia_client, transcript, wav_path,
        model_id=model_id, voice_id=voice_id,
        output_format=output_format, speed=speed,
    )
    dur = get_wav_duration(wav_path)
    if dur < 0.1:
        wav_path.unlink(missing_ok=True)
        return None
    return dur


def _finalize_versions(
    versions: list[dict | None], only_scores: set[int] | None, folder: Path,
) -> list[dict]:
    """Drop None entries; if only_scores is set, drop+delete anything not requested.

    Scores generated only as a prerequisite for a requested score (e.g. score 2
    generated solely so score 1 could be derived from it) are scratch —
    not part of what the caller asked for — so their WAV is removed here.
    """
    versions = [v for v in versions if v is not None]
    if only_scores is None:
        return versions
    kept, dropped = [], []
    for v in versions:
        (kept if v["score"] in only_scores else dropped).append(v)
    for v in dropped:
        wav_name = v.get("audio_file")
        if wav_name:
            (folder / wav_name).unlink(missing_ok=True)
    return kept


async def _process_speech_rate_entry(
    cartesia_client: AsyncCartesia,
    item: dict,
    direction: str,
    output_dir: Path,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    *,
    model_id: str,
    voice_id: str,
    output_format: dict,
    psola_only: bool = False,
    only_scores: set[int] | None = None,
) -> dict:
    """
    Generate speed variants for one utterance (2 TTS calls).

    ``only_scores`` restricts which scores are *kept* in the output/manifest.
    The 3→2→1 chain (score 2 needs score 3's duration as a ratio baseline;
    score 1 is a copy of score 2 + extra PSOLA) is unavoidable internally —
    any score not in ``only_scores`` is still generated as a prerequisite but
    deleted from disk and omitted from the manifest afterward.

    1. **Score 3** — Cartesia speed 1.0 (normal baseline).
    2. **Score 2** — Cartesia at max deviation (0.6 or 1.4), no PSOLA.
       In *psola_only* mode this is a copy of score 3 with PSOLA instead.
    3. **Score 1** — copy of score 2's audio, PSOLA-stretched ±0.3x further.
    """
    cfg = _DIRECTION_CFG[direction]
    want = only_scores if only_scores is not None else {1, 2, 3}
    needs_score2 = bool(want & {1, 2})  # score 1 also needs score 2 as an input
    needs_score1 = 1 in want

    async with semaphore:
        uid = item["id"]
        folder = output_dir / f"{uid:03d}_{direction}"
        folder.mkdir(parents=True, exist_ok=True)

        lines = [
            f"[{index}/{total}] Utterance {uid} — "
            f"\"{item['text'][:50]}\"  (direction: {direction})"
        ]

        tts_kwargs = dict(
            model_id=model_id, voice_id=voice_id, output_format=output_format,
        )

        # ── Score 3: normal (baseline) ──────────────────────────────
        s3_wav = folder / "score_3.wav"
        s3: dict = {
            "score": 3, "direction": "normal",
            "error_description": "Normal speech rate — no negative impact.",
            "sentence": item["text"], "cartesia_speed": 1.0,
        }

        if s3_wav.exists():
            baseline_dur = get_wav_duration(s3_wav)
            s3.update(audio_file="score_3.wav", duration=round(baseline_dur, 3),
                      duration_ratio=1.0, psola_applied=False)
            lines.append(f"  Score 3 (normal)... skipped -> score_3.wav  ({baseline_dur:.2f}s)")
        else:
            t0 = time.time()
            try:
                baseline_dur = await _generate_one_wav(
                    cartesia_client, item["text"], s3_wav, speed=1.0, **tts_kwargs,
                )
            except Exception as exc:
                baseline_dur = None
                lines.append(f"  Score 3 (normal)... FAILED ({time.time()-t0:.1f}s): {exc}")

            if baseline_dur is None:
                s3.update(audio_file=None, duration=None,
                          duration_ratio=None, psola_applied=False)
                print("\n" + "\n".join(lines), flush=True)
                entry = {
                    "utterance_id": uid, "original_text": item["text"],
                    "direction": direction,
                    "output_dir": str(folder.relative_to(output_dir)),
                    "versions": _finalize_versions([s3], only_scores, folder),
                }
                (folder / "manifest.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False))
                return entry

            s3.update(audio_file="score_3.wav", duration=round(baseline_dur, 3),
                      duration_ratio=1.0, psola_applied=False)
            lines.append(
                f"  Score 3 (normal)... done ({time.time()-t0:.1f}s) "
                f"-> score_3.wav  ({baseline_dur:.2f}s baseline)")

        # ── Score 2: Cartesia at max deviation, no PSOLA ───────────
        # Skipped entirely if neither score 1 nor score 2 was requested —
        # score 3 alone needs nothing further.
        s2_wav = folder / "score_2.wav"
        s2: dict | None = None

        if needs_score2:
            s2 = {
                "score": 2, "direction": direction,
                "error_description": cfg["score2_desc"],
                "sentence": item["text"],
                "cartesia_speed": 1.0 if psola_only else cfg["cartesia_speed"],
            }

            if s2_wav.exists():
                s2_dur = get_wav_duration(s2_wav)
                s2.update(audio_file="score_2.wav", duration=round(s2_dur, 3),
                          duration_ratio=round(s2_dur / baseline_dur, 3),
                          psola_applied=False)
                lines.append(f"  Score 2... skipped -> score_2.wav  ({s2_dur:.2f}s)")
            else:
                t0 = time.time()
                if psola_only:
                    # Copy baseline, PSOLA to match Cartesia-equivalent ratio
                    shutil.copy2(s3_wav, s2_wav)
                    # Approximate the Cartesia speed effect: speed 1.4 ≈ ratio 0.8,
                    # speed 0.6 ≈ ratio 1.9.  Use 1/speed as a rough proxy.
                    target_factor = 1.0 / cfg["cartesia_speed"]
                    psola_stretch(s2_wav, s2_wav, target_factor)
                    s2_dur = get_wav_duration(s2_wav)
                    s2.update(audio_file="score_2.wav", duration=round(s2_dur, 3),
                              duration_ratio=round(s2_dur / baseline_dur, 3),
                              psola_applied=True)
                    lines.append(
                        f"  Score 2... done ({time.time()-t0:.1f}s) "
                        f"-> score_2.wav  ({s2_dur:.2f}s, PSOLA-only)")
                else:
                    try:
                        s2_dur = await _generate_one_wav(
                            cartesia_client, item["text"], s2_wav,
                            speed=cfg["cartesia_speed"], **tts_kwargs,
                        )
                    except Exception as exc:
                        s2_dur = None
                        lines.append(f"  Score 2... FAILED ({time.time()-t0:.1f}s): {exc}")

                    if s2_dur is None:
                        s2.update(audio_file=None, duration=None,
                                  duration_ratio=None, psola_applied=False)
                        # Can't derive score 1 without score 2
                        print("\n" + "\n".join(lines), flush=True)
                        entry = {
                            "utterance_id": uid, "original_text": item["text"],
                            "direction": direction,
                            "output_dir": str(folder.relative_to(output_dir)),
                            "versions": _finalize_versions([s2, s3], only_scores, folder),
                        }
                        (folder / "manifest.json").write_text(
                            json.dumps(entry, indent=2, ensure_ascii=False))
                        return entry

                    s2.update(audio_file="score_2.wav", duration=round(s2_dur, 3),
                              duration_ratio=round(s2_dur / baseline_dur, 3),
                              psola_applied=False)
                    lines.append(
                        f"  Score 2... done ({time.time()-t0:.1f}s) "
                        f"-> score_2.wav  ({s2_dur:.2f}s, ratio {s2_dur/baseline_dur:.2f})")

        # ── Score 1: copy score 2 + PSOLA ±0.3x ───────────────────
        # Skipped entirely if score 1 wasn't requested.
        s1_wav = folder / "score_1.wav"
        s1: dict | None = None

        if needs_score1:
            s1 = {
                "score": 1, "direction": direction,
                "error_description": cfg["score1_desc"],
                "sentence": item["text"],
                "cartesia_speed": s2.get("cartesia_speed"),
                "psola_factor": cfg["psola_factor"],
            }

            if s1_wav.exists():
                s1_dur = get_wav_duration(s1_wav)
                s1.update(audio_file="score_1.wav", duration=round(s1_dur, 3),
                          duration_ratio=round(s1_dur / baseline_dur, 3),
                          psola_applied=True)
                lines.append(f"  Score 1... skipped -> score_1.wav  ({s1_dur:.2f}s)")
            else:
                t0 = time.time()
                shutil.copy2(s2_wav, s1_wav)
                psola_stretch(s1_wav, s1_wav, cfg["psola_factor"])
                s1_dur = get_wav_duration(s1_wav)
                s1.update(audio_file="score_1.wav", duration=round(s1_dur, 3),
                          duration_ratio=round(s1_dur / baseline_dur, 3),
                          psola_applied=True)
                lines.append(
                    f"  Score 1... done ({time.time()-t0:.1f}s) "
                    f"-> score_1.wav  ({s1_dur:.2f}s, ratio {s1_dur/baseline_dur:.2f}, "
                    f"PSOLA {cfg['psola_factor']}x on score 2)")

        print("\n" + "\n".join(lines), flush=True)

        entry = {
            "utterance_id": uid,
            "original_text": item["text"],
            "direction": direction,
            "output_dir": str(folder.relative_to(output_dir)),
            "versions": _finalize_versions([s1, s2, s3], only_scores, folder),
        }
        (folder / "manifest.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False)
        )
        return entry


async def run_speech_rate_pipeline(
    items: list[dict],
    output_dir: Path,
    cartesia_client: AsyncCartesia,
    model_id: str,
    voice_id: str,
    output_format: dict = DEFAULT_OUTPUT_FORMAT,
    *,
    tts_concurrency: int = 10,
    psola_only: bool = False,
    only_scores: set[int] | None = None,
) -> list[int]:
    """
    Generate speech-rate evaluation audio (no OpenAI phase needed).

    For each utterance the same sentence is synthesized at three levels:
      Score 1 — high negative impact  (severe speed deviation)
      Score 2 — low negative impact   (moderate speed deviation)
      Score 3 — no negative impact    (normal rate, used as baseline)

    Args:
        psola_only: When True, only one Cartesia generation (speed 1.0)
            is done per utterance; scores 1 & 2 are produced purely via
            PSOLA time-stretching.  This eliminates pause/phrasing
            variance between versions.
        only_scores: Restrict which scores are delivered. Score 3 is the
            anchor for score 2's duration ratio, and score 2 is the input
            score 1 is derived from — requesting a lower score still pays
            for the higher ones internally, but only the requested scores'
            WAVs are kept and only they appear in the manifest.
    """
    mode = "PSOLA-only" if psola_only else "hybrid (Cartesia speed + PSOLA)"
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Generating speech rate variants — mode: {mode}")
    print(sep)

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_sem = asyncio.Semaphore(tts_concurrency)

    entries = await asyncio.gather(*(
        _process_speech_rate_entry(
            cartesia_client,
            item,
            "fast" if i % 2 == 1 else "slow",
            output_dir,
            i, len(items), tts_sem,
            model_id=model_id,
            voice_id=voice_id,
            output_format=output_format,
            psola_only=psola_only,
            only_scores=only_scores,
        )
        for i, item in enumerate(items, 1)
    ))

    successful_ids = [
        e["utterance_id"]
        for e in entries
        if any(v.get("audio_file") for v in e["versions"])
    ]
    total_files = sum(
        sum(1 for v in e["versions"] if v.get("audio_file"))
        for e in entries
    )

    n_fast = sum(1 for e in entries if e["direction"] == "fast")
    n_slow = sum(1 for e in entries if e["direction"] == "slow")

    print(f"\n{sep}")
    print(f"Done!  Output directory: {output_dir}")
    print(f"  Directions: {n_fast} fast, {n_slow} slow")
    print(f"\nStructure:")
    print(f"  {output_dir.name}/")
    for entry in entries:
        folder_name = f"{entry['utterance_id']:03d}_{entry['direction']}"
        print(f"    {folder_name}/")
        print(f"      manifest.json")
        for v in entry["versions"]:
            ratio = v.get("duration_ratio")
            ratio_str = f"  (ratio {ratio:.2f})" if ratio else ""
            print(f"      {v.get('audio_file', '(failed)')}{ratio_str}")
    print(f"\nTotal audio files generated: {total_files}")
    return successful_ids
