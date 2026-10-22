"""
Three-phase pipeline for the human_plausibility dimension.

Phase 1 — OpenAI generates an artifact recipe (same as phoneme pipelines).
Phase 2 — Cartesia synthesizes the clean reference audio ONCE per utterance.
Phase 3 — artifact_utils applies the recipe to produce the degraded variant.

This avoids redundant Cartesia calls: all versions share the same spoken
sentence, so only one TTS synthesis is needed per utterance.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path

from openai import AsyncAzureOpenAI, AsyncOpenAI
from cartesia import AsyncCartesia

from .openai_utils import build_prompt, call_openai
from .tts_utils import generate_tts, DEFAULT_OUTPUT_FORMAT
from .artifact_utils import apply_artifacts_to_wav


# ---------------------------------------------------------------------------
# Random artifact selection
# ---------------------------------------------------------------------------

# Types grouped by perceptual category so two-artifact picks are always varied.
_ARTIFACT_CATEGORIES: dict[str, list[str]] = {
    "compression":    ["mp3_compression", "quantization_noise"],
    "clipping":       ["hard_clipping", "soft_clipping", "bit_crushing"],
    "glitch":         ["buffer_dropout", "stutter_repeat", "granular_fragmentation"],
    "noise":          ["white_noise", "pink_noise", "vinyl_crackle", "em_hum_50hz", "em_hum_60hz"],
    "time_pitch":     ["time_stretch_artifact", "pitch_correction_artifact", "phase_cancellation"],
    "digital_errors": ["bit_errors", "sample_rate_mismatch", "buffer_underrun"],
    "delay":          ["echo_feedback", "resonant_comb_filter", "feedback_squeal"],
    "spectral":       ["fft_smearing", "phase_distortion", "aliasing"],
}


def _pick_artifact_types() -> list[str]:
    """Return 1 or 2 randomly selected artifact type strings (always from different categories)."""
    n = random.choice([1, 2])
    categories = random.sample(list(_ARTIFACT_CATEGORIES.keys()), k=n)
    return [random.choice(_ARTIFACT_CATEGORIES[cat]) for cat in categories]


def _format_artifact_selection(types: list[str]) -> str:
    if len(types) == 1:
        return f"1 artifact — type must be `{types[0]}`"
    return f"2 artifacts — types must be `{types[0]}` and `{types[1]}` (in that order in the JSON)"


# ---------------------------------------------------------------------------
# Phase 1 — OpenAI
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
) -> dict | None:
    word = re.sub(r"[^a-zA-Z0-9]", "", item["target_word"].lower())
    manifest_path = output_dir / f"{item['id']:03d}_{word}" / "manifest.json"

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
        artifact_types = _pick_artifact_types()
        artifact_selection = _format_artifact_selection(artifact_types)
        prompt = build_prompt(prompt_template, item["text"], item["target_word"])
        prompt = prompt.replace("{{ARTIFACT_SELECTION}}", artifact_selection)
        t0 = time.time()
        lines = [f"[{index}/{total}] Utterance {item['id']}: \"{item['target_word']}\""]

        try:
            result = await call_openai(openai_client, deployment, prompt)
        except Exception as exc:
            elapsed = time.time() - t0
            lines += [f"  OpenAI ({deployment})... FAILED ({elapsed:.1f}s)", f"  ERROR: {exc}"]
            print("\n" + "\n".join(lines), flush=True)
            return None

        elapsed = time.time() - t0
        versions = result.get("versions", [])

        if not versions:
            lines.append(
                f"  OpenAI ({deployment})... done ({elapsed:.1f}s) — "
                "skipped (empty versions)"
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
# Phase 2+3 — Cartesia (once) + artifact application
# ---------------------------------------------------------------------------

async def _process_artifact_entry(
    cartesia_client: AsyncCartesia,
    entry: dict,
    output_dir: Path,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    *,
    model_id: str,
    voice_id: str,
    output_format: dict,
    only_scores: set[int] | None = None,
) -> None:
    async with semaphore:
        uid = entry["utterance_id"]
        word = re.sub(r"[^a-zA-Z0-9]", "", entry["target_word"].lower())
        folder = output_dir / f"{uid:03d}_{word}"
        folder.mkdir(parents=True, exist_ok=True)
        entry["output_dir"] = str(folder.relative_to(output_dir))

        lines = [f"[{index}/{total}] Utterance {uid} — \"{entry['target_word']}\""]

        # Identify clean version (no artifacts) and artifact version(s)
        clean_v = next(
            (v for v in entry["versions"] if not v.get("artifacts")),
            entry["versions"][-1],
        )
        artifact_versions = [v for v in entry["versions"] if v.get("artifacts")]

        # ── Phase 2: synthesize clean audio once ─────────────────────────────
        clean_wav = folder / f"{word}_{clean_v['score']}.wav"
        if clean_wav.exists():
            clean_v["audio_file"] = clean_wav.name
            lines.append(f"  Score {clean_v['score']}... skipped (already exists) -> {clean_wav.name}")
        else:
            t0 = time.time()
            try:
                await generate_tts(
                    cartesia_client,
                    clean_v["sentence"],
                    clean_wav,
                    model_id=model_id,
                    voice_id=voice_id,
                    output_format=output_format,
                )
                clean_v["audio_file"] = clean_wav.name
                lines.append(
                    f"  Score {clean_v['score']}... done ({time.time() - t0:.1f}s) -> {clean_wav.name}"
                )
            except Exception as exc:
                lines.append(f"  Score {clean_v['score']}... FAILED ({time.time() - t0:.1f}s)")
                lines.append(f"    ERROR: {exc}")
                clean_v["audio_file"] = None
                print("\n" + "\n".join(lines), flush=True)
                (folder / "manifest.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False)
                )
                return

        # ── Phase 3: apply artifacts ──────────────────────────────────────────
        for v in artifact_versions:
            score = v["score"]
            wav_name = f"{word}_{score}.wav"
            wav_path = folder / wav_name

            if wav_path.exists():
                v["audio_file"] = wav_name
                lines.append(f"  Score {score}... skipped (already exists) -> {wav_name}")
                continue

            t0 = time.time()
            try:
                apply_artifacts_to_wav(
                    clean_wav,
                    wav_path,
                    v["artifacts"],
                    sentence=entry.get("original_text", ""),
                    target_word=entry.get("target_word", ""),
                )
                v["audio_file"] = wav_name
                lines.append(f"  Score {score}... done ({time.time() - t0:.1f}s) -> {wav_name}")
            except Exception as exc:
                lines.append(f"  Score {score}... FAILED ({time.time() - t0:.1f}s)")
                lines.append(f"    ERROR: {exc}")
                v["audio_file"] = None

        if only_scores is not None:
            keep, drop = [], []
            for v in entry["versions"]:
                (keep if v["score"] in only_scores else drop).append(v)
            for v in drop:
                wav_name = v.get("audio_file")
                if wav_name:
                    (folder / wav_name).unlink(missing_ok=True)
            entry["versions"] = keep

        print("\n" + "\n".join(lines), flush=True)
        (folder / "manifest.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False)
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_artifact_pipeline(
    dimension_name: str,
    items: list[dict],
    prompt_template: str,
    output_dir: Path,
    openai_client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment: str,
    cartesia_client: AsyncCartesia,
    model_id: str,
    voice_id: str,
    output_format: dict = DEFAULT_OUTPUT_FORMAT,
    *,
    openai_concurrency: int = 5,
    tts_concurrency: int = 10,
    only_scores: set[int] | None = None,
) -> list[int]:
    """
    Matches the signature of run_two_phase_pipeline for drop-in use in pipeline.py.

    ``only_scores`` restricts which scores are delivered. Score 1 (clean) never
    depends on the GPT-picked artifact recipe — its ``sentence`` is always the
    unmodified input text — so when only score 1 is requested, Phase 1 (OpenAI)
    is skipped entirely rather than discarding an unused recipe.
    """
    sep = "=" * 60
    want = only_scores if only_scores is not None else {0, 1}

    if want == {1}:
        print(f"\n{sep}")
        print(f"Phase 1: skipped — only the clean (score 1) reference was requested")
        print(sep)
        manifest = [
            {
                "utterance_id": item["id"],
                "original_text": item["text"],
                "target_word": item["target_word"],
                "versions": [{
                    "score": 1,
                    "sentence": item["text"],
                    "error_description": "Clean reference — no artifacts applied.",
                    "artifacts": [],
                }],
            }
            for item in items
        ]
    else:
        # ── Phase 1: OpenAI ──────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"Phase 1: Generating {dimension_name} artifact recipes via OpenAI")
        print(sep)

        openai_sem = asyncio.Semaphore(openai_concurrency)
        results = await asyncio.gather(*(
            _process_openai_item(
                openai_client, deployment, prompt_template,
                item, i, len(items), openai_sem, output_dir,
            )
            for i, item in enumerate(items, 1)
        ))
        manifest = [r for r in results if r is not None]

    if not manifest:
        print("\nNo recipes generated — nothing to synthesize.")
        return []

    # ── Phase 2+3: Cartesia + artifact processing ────────────────────────────
    print(f"\n{sep}")
    print("Phase 2+3: Synthesizing clean audio and applying artifacts via Cartesia + post-processing")
    print(sep)

    output_dir.mkdir(parents=True, exist_ok=True)
    tts_sem = asyncio.Semaphore(tts_concurrency)
    await asyncio.gather(*(
        _process_artifact_entry(
            cartesia_client, entry, output_dir,
            i, len(manifest), tts_sem,
            model_id=model_id,
            voice_id=voice_id,
            output_format=output_format,
            only_scores=only_scores,
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
        w = re.sub(r"[^a-zA-Z0-9]", "", entry["target_word"].lower())
        folder_name = f"{entry['utterance_id']:03d}_{w}"
        print(f"    {folder_name}/")
        print(f"      manifest.json")
        for v in entry["versions"]:
            print(f"      {v.get('audio_file', '(failed)')}")
    print(f"\nTotal audio files generated: {total_files}")
    return successful_ids
