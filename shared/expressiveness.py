"""
Expressiveness pipeline.

Each input sentence gets ONE delivery; the score determines the recipe:

  Score 1 = over-expressive   — Scott - Sportscaster + emotion="confident"
  Score 2 = under-expressive  — Sienna - Encourager  + emotion="bored"
                                + PSOLA flattening (alpha=0.5, intensity
                                  comp, timing comp on egregious vowels)
  Score 3 = natural           — Brooke - Big Sister + speed=0.8

Voice, emotion, speed, and the PSOLA-flatten flag are all per-version —
the caller (pipeline.py) pre-populates them from SCORE_META and we just
execute. No tag pools, no direction logic, no eSpeak: every render goes
through Cartesia sonic-3, and score 2 gets an extra parselmouth pass via
experiments/flatten_expressiveness.py to take Cartesia's residual F0
variation out of the bored take.

Source text is Harvard sentences — emotionally neutral, phonetically
balanced — so all perceived affect comes from the recipe.

There's no Phase 1 LLM call; pipeline.py builds the items
deterministically from SCORE_META.

Output layout:
  output_dir/{score}_{id:03d}/
      {score}_{id:03d}.wav     # e.g. 1_047.wav, 2_708.wav, 3_293.wav
      manifest.json
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from cartesia import AsyncCartesia

from .tts_utils import generate_tts, DEFAULT_OUTPUT_FORMAT

# Canonical PSOLA parameters for the under-expressive anchor, lifted
# verbatim from shared/flatten_expressiveness.py's docstring ("Canonical
# defaults validated by listening on the 10 Harvard sentences under
# experiments/expressiveness_pairs/*_under/").
PSOLA_CONFIG = dict(
    alpha            = 0.5,
    skip_intensity   = False,     # canonical — intensity compression ON
    flatten_timing   = True,
    backend          = "praat",
    timing_threshold = 2.5,
    timing_floor     = 0.5,
)


def _folder_name(score: int, uid: int) -> str:
    return f"{score}_{uid:03d}"


async def _process_tts_entry(
    cartesia_client: AsyncCartesia,
    entry: dict,
    output_dir: Path,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    *,
    model_id: str,
    output_format: dict,
) -> None:
    async with semaphore:
        uid       = entry["utterance_id"]
        score     = entry["score"]
        label     = entry["label"]
        voice_id  = entry["voice_id"]
        voice_nm  = entry.get("voice_name") or "?"

        folder = output_dir / _folder_name(score, uid)
        folder.mkdir(parents=True, exist_ok=True)
        entry["output_dir"] = str(folder.relative_to(output_dir))

        lines = [
            f"[{index}/{total}] Item {uid} — score={score} label=\"{label}\" "
            f"voice=\"{voice_nm}\"  text={entry['original_text']!r}"
        ]

        for v in entry["versions"]:
            tag       = v.get("tts_emotion")          # None for natural
            speed     = v.get("speed")                # None or float
            do_psola  = bool(v.get("psola_flatten"))
            wav_name  = f"{v['score']}_{uid:03d}.wav"
            wav_path  = folder / wav_name
            t0        = time.time()

            # Backfill voice info for downstream Excel export — pipeline
            # already sets these per-version, but be defensive in case a
            # resumed manifest predates the per-version voice fields.
            v.setdefault("voice_id",   voice_id)
            v.setdefault("voice_name", voice_nm)

            if wav_path.exists():
                v["audio_file"] = wav_name
                lines.append(f"  {v['label']}... skipped (already exists) -> {wav_name}")
                continue

            tag_display   = tag if tag else "(default)"
            speed_display = f"{speed:.2f}" if speed is not None else "(default)"
            psola_display = "yes" if do_psola else "no"

            try:
                generation_config: dict = {}
                if tag is not None:
                    generation_config["emotion"] = tag
                if speed is not None:
                    generation_config["speed"] = speed

                await generate_tts(
                    cartesia_client,
                    v["sentence"],
                    wav_path,
                    model_id=model_id,
                    voice_id=voice_id,
                    output_format=output_format,
                    generation_config=generation_config or None,
                )

                if do_psola:
                    # Lazy import — pyworld/soundfile may not be installed
                    # in all environments; we only need them for score-2 PSOLA.
                    from .flatten_expressiveness import flatten_one  # noqa: PLC0415
                    # PSOLA is sync + CPU-bound; off-load to a thread so
                    # we don't block the TTS event loop. Write the
                    # flattened audio back over the Cartesia render.
                    await asyncio.to_thread(
                        flatten_one, wav_path, wav_path, **PSOLA_CONFIG,
                    )

                v["audio_file"] = wav_name
                lines.append(
                    f"  {v['label']} [cartesia tag={tag_display} "
                    f"speed={speed_display} psola={psola_display}]... done "
                    f"({time.time() - t0:.1f}s) -> {wav_name}"
                )
            except Exception as exc:
                lines.append(
                    f"  {v['label']} [cartesia tag={tag_display} "
                    f"speed={speed_display} psola={psola_display}]... "
                    f"FAILED ({time.time() - t0:.1f}s)"
                )
                lines.append(f"    ERROR: {exc}")
                v["audio_file"] = None

        print("\n" + "\n".join(lines), flush=True)
        (folder / "manifest.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False)
        )


async def run_expressiveness_pipeline(
    items: list[dict],
    output_dir: Path,
    cartesia_client: AsyncCartesia,
    model_id: str,
    output_format: dict = DEFAULT_OUTPUT_FORMAT,
    *,
    tts_concurrency: int = 10,
) -> list[int]:
    """
    Render each item once via Cartesia (+ optional PSOLA for score 2).

    Items shape (caller pre-populates the single version; no Phase 1):
        [{
            "id":         int,
            "text":       str,
            "score":      1 | 2 | 3,
            "label":      "over" | "under" | "natural",
            "voice_id":   str,
            "voice_name": str,
            "versions":   [   # exactly one entry per item
                {"score": int, "label": str,
                 "tts_emotion": str | None, "speed": float | None,
                 "psola_flatten": bool, "voice_id": str, "voice_name": str,
                 "sentence": str, "error_description": str},
            ],
         }, ...]

    Resume: if an item's manifest.json already exists with versions, we
    keep the existing entry as-is and only re-render WAVs that are
    missing on disk (handled per-version in _process_tts_entry).

    Returns: list of utterance_ids that produced at least one WAV.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("Phase 1 (skipped): expressiveness uses pre-existing utterances")
    print(f"{sep}\nPhase 2: Generating TTS audio via Cartesia "
          f"(+ PSOLA flatten on score-2 items)")
    print(sep)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-item manifest dicts. We may have a manifest on disk from
    # a prior interrupted run — keep its versions (they hold the chosen
    # parameters + any already-rendered audio_file references).
    manifest: list[dict] = []
    for it in items:
        manifest_path = (
            output_dir / _folder_name(it["score"], it["id"]) / "manifest.json"
        )
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("versions"):
                manifest.append(existing)
                continue
        manifest.append({
            "utterance_id":  it["id"],
            "original_text": it["text"],
            "target_word":   it["label"],     # surfaces in Excel "Target Word" column
            "score":         it["score"],
            "label":         it["label"],
            "voice_id":      it["voice_id"],
            "voice_name":    it["voice_name"],
            "versions":      it["versions"],
        })

    tts_sem = asyncio.Semaphore(tts_concurrency)
    await asyncio.gather(*(
        _process_tts_entry(
            cartesia_client, entry, output_dir,
            i, len(manifest), tts_sem,
            model_id=model_id,
            output_format=output_format,
        )
        for i, entry in enumerate(manifest, 1)
    ))

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
    print(f"Total audio files generated: {total_files}")
    return successful_ids
