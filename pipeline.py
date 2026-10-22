#!/usr/bin/env python3
"""
TTS Evaluation Generator — unified pipeline entry point.

Usage:
  python pipeline.py --dimension lexical_stress
  python pipeline.py --dimension phonetic_accuracy
  python pipeline.py --dimension prosodic_stress
  python pipeline.py --dimension speech_rate
  python pipeline.py --dimension speaker_identity_consistency
  python pipeline.py --dimension intonation           # F0 manipulation via Praat
  python pipeline.py --dimension all

Intonation is sentence-level: each utterance is synthesized verbatim by
ElevenLabs, then scores 1 and 2 are produced by applying Praat/PSOLA F0
manipulations to the base audio (score 3 is the clean base).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from shared.clients import (
    validate_env, validate_env_elevenlabs,
    make_openai_client, make_cartesia_client, make_elevenlabs_client,
)
from shared.utterances import load_utterances, pick_target_word, is_intonation_candidate, output_names
from shared.pipeline import run_two_phase_pipeline, run_speech_rate_pipeline
from shared.tts_utils import generate_tts
from shared.speaker_identity import (
    build_si_samples,
    run_speaker_identity_consistency,
    select_crossfade_utterances,
    select_stitch_pairs,
)
from shared.intonation import run_intonation_pipeline
from shared.artifact_pipeline import run_artifact_pipeline
from shared.expressiveness import run_expressiveness_pipeline
from shared.prosodic_strategies import make_prosodic_generator

ROOT       = Path(__file__).parent
CACHE_PATH = ROOT / ".cache" / "utterances.json"

# ---------------------------------------------------------------------------
# Dimension registry
# ---------------------------------------------------------------------------

@dataclass
class DimensionConfig:
    display_name: str
    num_utterances: int
    voice_id: str | None = None         # Cartesia voice UUID; None for ElevenLabs-only / speaker_identity_consistency
    require_polysyllabic: bool = False
    # When False (or sentence_level=True), utterance selection skips target-word picking
    # and assigns a placeholder so folder/file naming stays consistent.
    needs_target_word: bool = True
    pipeline_type: str = "phoneme"      # "phoneme" (OpenAI+Cartesia) or "speech_rate"
    kind: str = "two_phase"             # "two_phase" | "speaker_identity_consistency"
    tts_engine: str = "cartesia"        # "cartesia" or "elevenlabs"
    elevenlabs_voice_id: str | None = None
    sentence_level: bool = False        # True = no target_word selection needed (intonation)
    uses_artifact_pipeline: bool = False  # True for human_plausibility


DIMENSIONS: dict[str, DimensionConfig] = {
    "lexical_stress": DimensionConfig(
        display_name="lexical stress",
        voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",  # Blake - Helpful Agent
        num_utterances=16,
        require_polysyllabic=True,
    ),
    "phonetic_accuracy": DimensionConfig(
        display_name="phonetic accuracy",
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # default English voice
        num_utterances=20,
        require_polysyllabic=False,
    ),
    "prosodic_stress": DimensionConfig(
        display_name="prosodic stress",
        voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",  # Blake - Helpful Agent
        num_utterances=16,
        require_polysyllabic=False,
        needs_target_word=False,
    ),
    "speech_rate": DimensionConfig(
        display_name="speech rate",
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # default English voice
        num_utterances=20,
        require_polysyllabic=False,
        pipeline_type="speech_rate",
    ),
    "speaker_identity_consistency": DimensionConfig(
        display_name="speaker identity consistency",
        kind="speaker_identity_consistency",
        num_utterances=8,  # N paired samples (each = 1 inconsistent + 1 consistent WAV).
                           # Consumes ~1.5N utterances; default 8 = one full alternation cycle.
    ),
    "intonation": DimensionConfig(
        display_name="intonation",
        num_utterances=5,
        require_polysyllabic=False,
        tts_engine="elevenlabs",
        elevenlabs_voice_id="nPczCjzI2devNBz1zQrb",  # Brian
        sentence_level=True,
    ),
    "human_plausibility": DimensionConfig(
        display_name="human plausibility",
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # default English voice
        num_utterances=20,
        require_polysyllabic=False,
        uses_artifact_pipeline=True,
    ),
    "expressiveness": DimensionConfig(
        display_name="expressiveness",
        voice_id="2f22b9bc-b0eb-4cb6-b5ae-0c099a0fdfad",  # Scott - Sportscaster (default)
        num_utterances=100,
        needs_target_word=False,
        sentence_level=False,
        pipeline_type="expressiveness",
    ),
    "prosodic_boundary": DimensionConfig(
        display_name="prosodic boundary",
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",
        num_utterances=60,
        needs_target_word=False,
        pipeline_type="prosodic_boundary",
        # NOTE: The preset strategy uses a 1/2/3 severity scale (1=worst, 3=canonical),
        # NOT the binary 0/1 used by lexical_stress. Downstream consumers
        # should be aware of this difference when processing manifest scores.
    ),
}

# Dimensions that call OpenAI/Azure OpenAI to generate variants. prosodic_boundary
# only needs it for the extra_punctuation strategy (checked separately below).
NEEDS_OPENAI = {"lexical_stress", "phonetic_accuracy", "prosodic_stress", "human_plausibility"}


def _env_requirements(dimension: str, args: argparse.Namespace) -> tuple[bool, bool]:
    """Return (needs_openai, needs_cartesia) for the selected dimension(s)."""
    if dimension == "all":
        return True, True  # the batch includes GPT dimensions and Cartesia-only ones
    needs_openai = dimension in NEEDS_OPENAI or (
        dimension == "prosodic_boundary" and (args.strategy or "preset") == "extra_punctuation"
    )
    needs_cartesia = DIMENSIONS[dimension].tts_engine == "cartesia"
    return needs_openai, needs_cartesia


# Valid score values per dimension, used to validate/scope --only-scores.
# prosodic_boundary is strategy-dependent (see valid_scores_for) so it's omitted here.
DIMENSION_SCORE_RANGES: dict[str, frozenset[int]] = {
    "lexical_stress":                frozenset({0, 1}),
    "phonetic_accuracy":             frozenset({1, 2, 3}),
    "prosodic_stress":                frozenset({0, 1}),
    "speech_rate":                   frozenset({1, 2, 3}),
    "speaker_identity_consistency":  frozenset({0, 1}),
    "intonation":                    frozenset({1, 2, 3}),
    "human_plausibility":            frozenset({0, 1}),
    "expressiveness":                frozenset({1, 2, 3}),
}


def valid_scores_for(key: str, args: argparse.Namespace) -> frozenset[int]:
    """The set of score values this dimension can produce right now.

    prosodic_boundary's scale depends on --strategy: 'preset' uses 1/2/3
    (1=worst), the other strategies use a binary 0/1 (0=worst) pair.
    """
    if key == "prosodic_boundary":
        strategy = getattr(args, "strategy", None) or "preset"
        return frozenset({1, 2, 3}) if strategy == "preset" else frozenset({0, 1})
    return DIMENSION_SCORE_RANGES[key]


def _parse_only_scores(raw: str | None) -> set[int] | None:
    """Parse the --only-scores CLI value into a set of ints, or None if unset."""
    if not raw:
        return None
    try:
        return {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        sys.exit(f"Error: --only-scores must be comma-separated integers, got: {raw!r}")

CARTESIA_OUTPUT_FORMAT = {
    "container": "wav",
    "encoding": "pcm_s16le",
    "sample_rate": 44100,
}

# Expressiveness scoring. Each sentence gets ONE score (not three):
#   1 = over-expressive  - Scott + emotion="confident"
#   2 = under-expressive - Sienna + emotion="bored" + PSOLA flattening
#   3 = natural          - Brooke + speed=0.8
EXPRESSIVENESS_SCORE_META: dict[int, dict] = {
    1: {
        "label":         "over",
        "voice_name":    "Scott - Sportscaster",
        "voice_id":      "2f22b9bc-b0eb-4cb6-b5ae-0c099a0fdfad",
        "tts_emotion":   "confident",
        "speed":         None,
        "psola_flatten": False,
        "description": (
            "Over-expressive - Scott - Sportscaster with Cartesia "
            "emotion='confident'; theatrical broadcast cadence on routine content."
        ),
    },
    2: {
        "label":         "under",
        "voice_name":    "Sienna - Encourager",
        "voice_id":      "4e41a434-85fc-4614-b203-af79ba44d473",
        "tts_emotion":   "bored",
        "speed":         None,
        "psola_flatten": True,
        "description": (
            "Under-expressive - Sienna - Encourager with Cartesia "
            "emotion='bored', then PSOLA flattening (alpha=0.5, intensity "
            "comp, timing comp); F0 excursion halved around speaker median."
        ),
    },
    3: {
        "label":         "natural",
        "voice_name":    "Brooke - Big Sister",
        "voice_id":      "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
        "tts_emotion":   None,
        "speed":         0.8,
        "psola_flatten": False,
        "description": (
            "Natural - Brooke - Big Sister at default delivery "
            "(no emotion tag), speed=0.8."
        ),
    },
}

DATASETS: dict[str, Path | None] = {
    "mozilla":  ROOT / "data" / "utterances.json",
    "harvard":  ROOT / "data" / "harvard_sentences.json",
    "emergent": None,  # fetched from bosonai/EmergentTTS-Eval on HuggingFace
    "prosodic_boundary_preset": ROOT / "data" / "prosodic_boundary_preset.json",
}


# ---------------------------------------------------------------------------
# Utterance cache — tracks used IDs per (dimension, dataset) across runs
# ---------------------------------------------------------------------------

def load_cache() -> dict[str, list[int]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict[str, list[int]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _dataset_tag(args: argparse.Namespace) -> str:
    """Identifier for the utterance source, used in cache keys and folder names.

    For --dataset emergent the active category is folded into the tag so that
    different categories cache independently and produce non-colliding folder
    names (emergent IDs re-number from 0 within each category).
    """
    if args.utterances:
        return args.utterances.stem
    if args.dataset == "emergent" and args.category:
        safe = re.sub(r"[^a-z0-9]+", "_", args.category.lower()).strip("_")
        return f"emergent_{safe}" if safe else "emergent"
    return args.dataset


def cache_key_for(dimension: str, cfg: DimensionConfig, args: argparse.Namespace) -> str:
    """Cache key encoding dimension and dataset, e.g. "intonation_harvard"."""
    return f"{dimension}_{_dataset_tag(args)}"


def rebuild_cache_from_disk(output_base: Path, dataset_tag: str) -> dict[str, list[int]]:
    """
    Scan output locations and build a cache for *dataset_tag*.

    Because the dataset a WAV came from isn't recorded on disk, this attributes
    every scanned ID to *dataset_tag*. Pass the same --dataset used when the
    audio was originally produced, or clear the cache manually before
    rebuilding for a different dataset.

    Sentence-level dimensions are split across method subfolders
    ({output_base}/{dim}/{method}/...); each method gets its own cache key.
    """
    cache: dict[str, list[int]] = {}
    for dim in DIMENSIONS:
        seen: set[int] = set()
        for root in (ROOT / dim / "output", output_base / dim):
            if not root.exists():
                continue
            for wav in root.rglob("*.wav"):
                # folder name shapes vary: {dataset}_{id:03d}_{word} (two_phase),
                # {id:03d} (speaker_identity crossfade),
                # {id_a:03d}_{id_b:03d} (speaker_identity stitch).
                # Grab every int-parseable part to cover all shapes.
                for part in wav.parent.name.split("_"):
                    try:
                        seen.add(int(part))
                    except ValueError:
                        continue
        if seen:
            cache[f"{dim}_{dataset_tag}"] = sorted(seen)
    return cache


# ---------------------------------------------------------------------------
# Excel export — scans all manifests under results/ and writes a flat table
# ---------------------------------------------------------------------------

EXCEL_HEADERS = [
    "Dimension",
    "Utterance ID",
    "Original Text",
    "Target Word",
    "Direction",
    "Score",
    "Error Description",
    "IPA Transcription",
    "Synthesized Sentence",
    "Duration Ratio",
    "Audio File",
]

HEADER_FILL  = PatternFill(fill_type="solid", fgColor="1F4E79")
HEADER_FONT  = Font(bold=True, color="FFFFFF")
ALT_ROW_FILL = PatternFill(fill_type="solid", fgColor="D6E4F0")

COL_WIDTHS = {
    "Dimension":           18,
    "Utterance ID":        14,
    "Original Text":       50,
    "Target Word":         16,
    "Direction":           12,
    "Score":                8,
    "Error Description":   60,
    "IPA Transcription":   30,
    "Synthesized Sentence":55,
    "Duration Ratio":      14,
    "Audio File":          45,
}


def generate_excel(results_dir: Path) -> None:
    rows: list[list] = []

    for manifest_path in sorted(results_dir.rglob("manifest.json")):
        dimension = manifest_path.relative_to(results_dir).parts[0]

        data = json.loads(manifest_path.read_text())
        for v in data.get("versions", []):
            audio_rel = (
                str(manifest_path.parent.relative_to(results_dir) / v["audio_file"])
                if v.get("audio_file") else ""
            )
            rows.append([
                dimension,
                data.get("utterance_id", ""),
                data.get("original_text", ""),
                data.get("target_word", ""),
                v.get("direction", ""),
                v.get("score", ""),
                v.get("error_description", ""),
                v.get("transcription", ""),
                v.get("sentence", ""),
                v.get("duration_ratio", ""),
                audio_rel,
            ])

    if not rows:
        print("\nNo manifest data found — skipping Excel export.")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    # Header row
    ws.append(EXCEL_HEADERS)
    for col_idx, header in enumerate(EXCEL_HEADERS, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Data rows
    for row_idx, row in enumerate(rows, 2):
        ws.append(row)
        if row_idx % 2 == 0:
            for col_idx in range(1, len(EXCEL_HEADERS) + 1):
                ws.cell(row=row_idx, column=col_idx).fill = ALT_ROW_FILL

    # Column widths + wrap for long text columns
    wrap_cols = {"Original Text", "Error Description", "Synthesized Sentence"}
    for col_idx, header in enumerate(EXCEL_HEADERS, 1):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = COL_WIDTHS.get(header, 20)
        if header in wrap_cols:
            for row_idx in range(2, len(rows) + 2):
                ws.cell(row=row_idx, column=col_idx).alignment = Alignment(wrap_text=True)

    ws.freeze_panes = "A2"

    excel_path = results_dir / "summary.xlsx"
    wb.save(excel_path)
    print(f"\nExcel summary written → {excel_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    dim_choices = list(DIMENSIONS.keys()) + ["all"]
    p = argparse.ArgumentParser(
        description=(
            "BeyondNaturalnessProbe — produce scored audio samples for the Beyond Naturalness "
            "benchmark. Word-level dimensions (lexical_stress, phonetic_accuracy) "
            "synthesize GPT-generated IPA variants via Cartesia; sentence-level "
            "intonation synthesizes via ElevenLabs and applies Praat F0 manipulations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dimension", "-d",
        choices=dim_choices, required=True,
        help="Dimension to run, or 'all' to run every dimension concurrently.",
    )
    p.add_argument(
        "--stack-rate", type=float, default=0.0, metavar="P",
        help="Intonation dimension: probability that a score-1 sample composes two distinct "
             "F0 manipulations in sequence instead of applying one. "
             "0.0 disables stacking (default — best for consistent severity across samples); "
             "raise to 0.3+ if you want more variety at the cost of some in-bucket variance.",
    )
    p.add_argument(
        "--num", "-n", type=int, default=None, metavar="N",
        help="Number of utterances to process. Overrides the dimension default. "
             "Ignored when --dimension all (each dimension uses its own default).",
    )
    p.add_argument(
        "--voice", type=str, default=None, metavar="VOICE_ID",
        help="Cartesia voice UUID. Overrides the dimension default. "
             "Ignored when --dimension all.",
    )
    p.add_argument(
        "--model", type=str, default="sonic-3", metavar="MODEL",
        help="Cartesia model ID.",
    )
    p.add_argument(
        "--deployment", type=str, default="gpt-5", metavar="DEPLOYMENT",
        help="OpenAI model name (or Azure OpenAI deployment name, if AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY are set).",
    )
    p.add_argument(
        "--prompts-dir", type=Path, default=ROOT / "prompts", metavar="DIR",
        help="Directory containing {dimension}.txt prompt files.",
    )
    p.add_argument(
        "--dataset", choices=list(DATASETS.keys()), default="mozilla",
        help="Named dataset to draw utterances from. "
             "`emergent` pulls from the bosonai/EmergentTTS-Eval HuggingFace repo.",
    )
    p.add_argument(
        "--utterances", type=Path, default=None, metavar="FILE",
        help="Path to a custom utterances JSON file. Overrides --dataset.",
    )
    p.add_argument(
        "--category", type=str, default=None, metavar="CAT",
        help="(--dataset emergent only) Filter to rows whose `category` field matches "
             "this value. Case/space/underscore-insensitive. Available: "
             "Questions, Foreign Words, Paralinguistics, Emotions, "
             "Syntactic Complexity, Pronunciation.",
    )
    p.add_argument(
        "--strategy",
        choices=["extra_punctuation", "omit_spaces", "omit_punctuation", "preset"],
        default=None,
        help="(prosodic_boundary only) Strategy for score-0 transform. "
             "Defaults to 'preset' (pre-defined 60-sample distribution).",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "results", metavar="DIR",
        help="Results directory. Audio + manifests go to {DIR}/{dimension}/. "
             "Excel summary is written to {DIR}/summary.xlsx.",
    )
    p.add_argument(
        "--openai-concurrency", type=int, default=5, metavar="N",
        help="Max concurrent OpenAI requests.",
    )
    p.add_argument(
        "--tts-concurrency", type=int, default=10, metavar="N",
        help="Max concurrent TTS requests (Cartesia or ElevenLabs, depending on dimension).",
    )
    p.add_argument(
        "--samples", "-s", type=int, default=1, metavar="N",
        help="Number of successive batches to run. Each batch picks a fresh set "
             "of utterances (seed increments by 1) and appends to the same "
             "output directory. Already-used IDs are tracked in the cache so no "
             "utterance repeats across batches without --reset-cache.",
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="Random seed for reproducible utterance selection. "
             "When running multiple samples, seed increments by 1 per batch.",
    )
    p.add_argument(
        "--max-words", type=int, default=None, metavar="N",
        help="(intonation only) Discard utterances longer than N words. "
             "Useful with --dataset emergent where sentences can be very long.",
    )
    p.add_argument(
        "--min-words", type=int, default=None, metavar="N",
        help="(intonation only) Discard utterances shorter than N words.",
    )
    p.add_argument(
        "--only-scores", type=str, default=None, metavar="N[,N]",
        help="Comma-separated score numbers to generate, e.g. '2' or '1,2'. "
             "Skipped scores produce no audio and are omitted from the manifest. "
             "Valid values are dimension-specific (see README); with --dimension all, "
             "a dimension whose scale doesn't include any requested value is skipped. "
             "Some dimensions (speech_rate, human_plausibility, "
             "speaker_identity_consistency) still pay part of the generation cost for "
             "an unrequested 'anchor' score internally — see README for details.",
    )
    p.add_argument(
        "--counts",
        type=str,
        default="50,20,30",
        metavar="N1,N2,N3",
        help="(expressiveness only) Comma-separated sentence counts for "
             "scores 1,2,3 (over,under,natural). Default: '50,20,30'. "
             "If --only-scores is also given, counts for excluded scores are zeroed.",
    )
    p.add_argument(
        "--reset-cache", action="store_true",
        help="Clear the utterance cache for the selected dimension(s) before running.",
    )
    p.add_argument(
        "--rebuild-cache", action="store_true",
        help="Rebuild the cache from actual WAV files on disk (fixes stale cache entries "
             "from interrupted runs). Scans all output locations and only keeps IDs that "
             "have at least one WAV file.",
    )
    p.add_argument(
        "--psola-only", action="store_true",
        help="(speech_rate only) Generate one Cartesia render at normal speed, "
             "then produce all speed variants via PSOLA time-stretching only. "
             "Eliminates pause/phrasing variance between versions.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utterance selection
# ---------------------------------------------------------------------------

def select_items(
    utterances: list[dict],
    n: int,
    require_polysyllabic: bool,
    exclude_ids: set[int],
    needs_target_word: bool = True,
    *,
    sentence_level: bool = False,
    skip_candidate_filter: bool = False,
    min_words: int | None = None,
    max_words: int | None = None,
) -> list[dict]:
    available = [u for u in utterances if u["id"] not in exclude_ids]
    if len(available) < n:
        print(
            f"  Warning: only {len(available)} unused utterances available "
            f"(requested {n}). Consider --reset-cache to reuse utterances."
        )

    if sentence_level:
        if skip_candidate_filter:
            filtered = available
        else:
            filtered = [u for u in available if is_intonation_candidate(u["text"])]
            dropped = len(available) - len(filtered)
            if dropped:
                print(
                    f"  Filtered out {dropped} utterance(s) unsuitable for intonation "
                    f"(non-declarative, contain punctuation, or outside 4–12 words); "
                    f"{len(filtered)} candidates remain."
                )
        if min_words is not None:
            filtered = [u for u in filtered if len(u["text"].split()) >= min_words]
        if max_words is not None:
            before = len(filtered)
            filtered = [u for u in filtered if len(u["text"].split()) <= max_words]
            dropped = before - len(filtered)
            if dropped:
                print(f"  Filtered out {dropped} utterance(s) exceeding {max_words} words; "
                      f"{len(filtered)} candidates remain.")
        sample = random.sample(filtered, min(n, len(filtered)))
        items = [{"id": u["id"], "text": u["text"], "target_word": ""} for u in sample]
        return items

    items: list[dict] = []
    if not needs_target_word:
        # Whole-sentence dimensions (e.g. prosodic_stress) don't target one
        # word. Assign a fixed placeholder so folder/file naming still fits
        # the `{id:03d}_{word}` convention.
        for utt in random.sample(available, min(n, len(available))):
            items.append({"id": utt["id"], "text": utt["text"],
                          "target_word": "stress"})
        return items

    if require_polysyllabic:
        pool = list(available)
        random.shuffle(pool)
        for utt in pool:
            if len(items) >= n:
                break
            target = pick_target_word(utt["text"], require_polysyllabic=True)
            if target is None:
                print(f"  Skipping utterance {utt['id']}: no polysyllabic target word")
                continue
            items.append({"id": utt["id"], "text": utt["text"], "target_word": target})
    else:
        for utt in random.sample(available, min(n, len(available))):
            target = pick_target_word(utt["text"], require_polysyllabic=False)
            if target is None:
                print(f"  Skipping utterance {utt['id']}: no suitable target word")
                continue
            items.append({"id": utt["id"], "text": utt["text"], "target_word": target})
    return items


# ---------------------------------------------------------------------------
# Speaker-identity-consistency runner
# ---------------------------------------------------------------------------

async def _run_speaker_identity_dimension(
    key: str,
    cfg: DimensionConfig,
    args: argparse.Namespace,
    utterances: list[dict],
    cache: dict[str, list[int]],
    only_scores: set[int] | None = None,
) -> tuple[str, list[int]]:
    """Produces N paired samples, each with (score 0 inconsistent, score 1 consistent).

    Alternates mode × gender × voice-order on a 2×2×2 schedule so every 8
    consecutive samples visit all combinations exactly once. Half the samples
    are crossfade (1 utterance each); half are stitch (2 utterances each).
    All consumed utterance IDs land in the `speaker_identity_consistency`
    cache key.
    """
    running_all = args.dimension == "all"
    n = cfg.num_utterances if running_all or args.num is None else args.num
    cf_n = (n + 1) // 2   # crossfade takes the ceil on odd N
    st_n = n // 2

    exclude_ids = set(cache.get(key, []))
    crossfade_items = select_crossfade_utterances(utterances, cf_n, exclude_ids)
    after_cf_excl = exclude_ids | {u["id"] for u in crossfade_items}
    stitch_pairs = select_stitch_pairs(utterances, st_n, after_cf_excl)

    samples = build_si_samples(crossfade_items, stitch_pairs, n)
    if not samples:
        print(f"[{key}] No unused utterances available — skipping. Use --reset-cache to reuse.")
        return key, []

    print(f"\nBuilt {len(samples)} paired sample(s) for {cfg.display_name} "
          f"({len(exclude_ids)} already used). Alternation schedule "
          f"(mode/gender/order):\n")
    for s in samples:
        order_tag = "rev" if s.reversed_order else "fwd"
        if s.mode == "crossfade":
            print(f"  [{s.index:>2}] crossfade / {s.pair.name} / {order_tag}   "
                  f"utt [{s.utt_primary['id']:>3}]")
        else:
            print(f"  [{s.index:>2}] stitch    / {s.pair.name} / {order_tag}   "
                  f"utts [{s.utt_primary['id']:>3}] + "
                  f"[{s.utt_partner['id']:>3}]")
    print()

    consumed = await run_speaker_identity_consistency(
        samples,
        args.output / key,
        make_cartesia_client(),
        model_id=args.model,
        tts_concurrency=args.tts_concurrency,
        only_scores=only_scores,
    )

    successful_ids = list(dict.fromkeys(consumed))
    success_set = set(successful_ids)
    def _uids(s) -> set[int]:
        uids = {s.utt_primary["id"]}
        if s.utt_partner is not None:
            uids.add(s.utt_partner["id"])
        return uids
    failed = sum(1 for s in samples if not _uids(s) <= success_set)
    if failed:
        print(f"  [{key}] Note: {failed} sample(s) had failures — "
              f"those utterances not cached, will retry next run.")
    return key, successful_ids


# ---------------------------------------------------------------------------
# Expressiveness dimension helpers
# ---------------------------------------------------------------------------

def select_expressiveness_items(
    utterances: list[dict],
    counts_str: str,
    exclude_ids: set[int],
) -> list[dict]:
    """
    Build the item list for the expressiveness dimension.

    Each item gets exactly ONE score (1, 2, or 3) based on the split defined
    by *counts_str* (e.g. "50,20,30" = 50 over + 20 under + 30 natural).
    Returns items with keys: id, text, score, label, voice_id, voice_name,
    versions (single-element list).
    """
    try:
        n1, n2, n3 = [int(x.strip()) for x in counts_str.split(",")]
    except (ValueError, TypeError):
        sys.exit(f"--counts must be three comma-separated integers, got: {counts_str!r}")
    score_counts = [(1, n1), (2, n2), (3, n3)]

    available = [u for u in utterances if u["id"] not in exclude_ids]
    total_needed = n1 + n2 + n3
    if len(available) < total_needed:
        print(
            f"  Warning: only {len(available)} unused utterances available "
            f"(requested {total_needed}). Consider --reset-cache to reuse utterances."
        )

    pool = random.sample(available, min(total_needed, len(available)))
    items: list[dict] = []
    offset = 0
    for score, count in score_counts:
        meta = EXPRESSIVENESS_SCORE_META[score]
        for utt in pool[offset : offset + count]:
            gen_cfg: dict = {}
            if meta["tts_emotion"] is not None:
                gen_cfg["tts_emotion"] = meta["tts_emotion"]
            if meta["speed"] is not None:
                gen_cfg["speed"] = meta["speed"]
            items.append({
                "id":         utt["id"],
                "text":       utt["text"],
                "score":      score,
                "label":      meta["label"],
                "voice_id":   meta["voice_id"],
                "voice_name": meta["voice_name"],
                "versions": [{
                    "score":             score,
                    "label":             meta["label"],
                    "tts_emotion":       meta.get("tts_emotion"),
                    "speed":             meta.get("speed"),
                    "psola_flatten":     meta["psola_flatten"],
                    "voice_id":          meta["voice_id"],
                    "voice_name":        meta["voice_name"],
                    "sentence":          utt["text"],
                    "error_description": meta["description"],
                }],
            })
        offset += count
    return items


async def run_expressiveness_dimension(
    cfg: DimensionConfig,
    args: argparse.Namespace,
    cache: dict[str, list[int]],
    only_scores: set[int] | None = None,
) -> tuple[str, list[int]]:
    """Harvard sentences split into over/under/natural batches, synthesized via Cartesia.

    Each sentence already gets exactly one score, so ``only_scores`` is applied
    by zeroing out the --counts entries for excluded scores before selection —
    no wasted generation.
    """
    cache_key = "expressiveness_harvard"
    output_dir = args.output / "expressiveness"

    utterances = load_utterances(ROOT / "data" / "harvard_sentences.json")

    counts_str = args.counts
    if only_scores is not None:
        try:
            n1, n2, n3 = [int(x.strip()) for x in counts_str.split(",")]
        except (ValueError, TypeError):
            sys.exit(f"--counts must be three comma-separated integers, got: {counts_str!r}")
        counts_str = ",".join(str(n if score in only_scores else 0)
                               for score, n in ((1, n1), (2, n2), (3, n3)))

    exclude_ids = set(cache.get(cache_key, []))
    items = select_expressiveness_items(
        utterances,
        counts_str=counts_str,
        exclude_ids=exclude_ids,
    )
    if not items:
        print("[expressiveness] No unused utterances available - skipping. Use --reset-cache to reuse.")
        return cache_key, []

    counts = {1: sum(1 for it in items if it["score"] == 1),
              2: sum(1 for it in items if it["score"] == 2),
              3: sum(1 for it in items if it["score"] == 3)}
    print(
        f"\nSelected {len(items)} utterances for expressiveness "
        f"(over={counts[1]}, under={counts[2]}, natural={counts[3]}):\n"
    )
    for item in items:
        print(f"  [{item['id']:>3}] score={item['score']} ({item['label']:7s})  \"{item['text'][:60]}\"")
    print()

    successful_ids = await run_expressiveness_pipeline(
        items=items,
        output_dir=output_dir,
        cartesia_client=make_cartesia_client(),
        model_id=args.model,
        output_format=CARTESIA_OUTPUT_FORMAT,
        tts_concurrency=args.tts_concurrency,
    )

    failed = len(items) - len(successful_ids)
    if failed:
        print(f"  [expressiveness] Note: {failed} item(s) failed - not cached, will retry next run.")
    return cache_key, successful_ids


# ---------------------------------------------------------------------------
# Prosodic boundary dimension
# ---------------------------------------------------------------------------

async def run_prosodic_boundary_dimension(
    cfg: DimensionConfig,
    args: argparse.Namespace,
    cache: dict[str, list[int]],
    only_scores: set[int] | None = None,
) -> tuple[str, list[int]]:
    """
    Prosodic boundary pipeline: generates manifest entries via make_prosodic_generator,
    then synthesizes each version via Cartesia. Default strategy is 'preset'
    (pre-defined 60-sample distribution — no OpenAI call needed).

    ``only_scores`` filters each entry's versions before synthesis — for the
    preset strategy (each utterance has one fixed score) this means only the
    matching utterances get synthesized at all; for the other strategies it
    means only the requested side of the {0,1} pair is rendered.
    """
    strategy = getattr(args, "strategy", None) or "preset"
    dataset_name = "prosodic_boundary_preset"
    output_dir = args.output / "prosodic_boundary"
    cache_key = f"prosodic_boundary_{dataset_name}"
    dataset_tag = dataset_name

    utterances = load_utterances(DATASETS[dataset_name])
    items = [{"id": u["id"], "text": u["text"]} for u in utterances]

    running_all = args.dimension == "all"
    voice = cfg.voice_id if (running_all or args.voice is None) else args.voice

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Prosodic boundary [{strategy}]: {len(items)} items")
    print(sep)

    generate_variants = make_prosodic_generator(
        client=make_openai_client() if strategy == "extra_punctuation" else None,
        deployment=args.deployment,
        dataset=dataset_name,
        strategy=strategy,
        prompts_dir=args.prompts_dir,
    )

    # Phase 1: generate manifest entries
    phase1_sem = asyncio.Semaphore(args.openai_concurrency)

    async def _gen(item: dict) -> dict | None:
        async with phase1_sem:
            return await generate_variants(item)

    results = await asyncio.gather(*(_gen(item) for item in items))
    manifest_entries = [r for r in results if r is not None]

    if only_scores is not None:
        for entry in manifest_entries:
            entry["versions"] = [v for v in entry["versions"] if v["score"] in only_scores]
        manifest_entries = [e for e in manifest_entries if e["versions"]]

    if not manifest_entries:
        print("No manifest entries produced — skipping TTS.")
        return cache_key, []

    # Phase 2: Cartesia TTS
    print(f"\n{sep}")
    print(f"Phase 2: Synthesizing {len(manifest_entries)} items via Cartesia")
    print(sep)

    output_dir.mkdir(parents=True, exist_ok=True)
    cartesia = make_cartesia_client()
    tts_sem = asyncio.Semaphore(args.tts_concurrency)

    async def _synth(entry: dict, index: int, total: int) -> None:
        async with tts_sem:
            uid = entry["utterance_id"]
            folder_name, file_prefix = output_names(uid, "", dataset_tag)
            folder = output_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            lines = [f"[{index}/{total}] Item {uid}"]

            for v in entry["versions"]:
                score = v["score"]
                wav_name = f"{file_prefix}_{score}.wav"
                wav_path = folder / wav_name
                if wav_path.exists():
                    v["audio_file"] = wav_name
                    lines.append(f"  Score {score}... skipped -> {wav_name}")
                    continue
                try:
                    await generate_tts(
                        cartesia, v["sentence"], wav_path,
                        model_id=args.model, voice_id=voice,
                        output_format=CARTESIA_OUTPUT_FORMAT,
                    )
                    v["audio_file"] = wav_name
                    lines.append(f"  Score {score}... done -> {wav_name}")
                except Exception as exc:
                    v["audio_file"] = None
                    lines.append(f"  Score {score}... FAILED: {exc}")

            print("\n" + "\n".join(lines), flush=True)
            (folder / "manifest.json").write_text(
                json.dumps(entry, indent=2, ensure_ascii=False)
            )

    await asyncio.gather(*(
        _synth(entry, i, len(manifest_entries))
        for i, entry in enumerate(manifest_entries, 1)
    ))

    successful_ids = [
        e["utterance_id"]
        for e in manifest_entries
        if any(v.get("audio_file") for v in e["versions"])
    ]
    print(f"\n{sep}")
    print(f"Done! {len(successful_ids)} prosodic boundary items in {output_dir}")
    return cache_key, successful_ids


# ---------------------------------------------------------------------------
# Run one dimension
# ---------------------------------------------------------------------------

async def run_dimension(
    key: str,
    cfg: DimensionConfig,
    args: argparse.Namespace,
    utterances: list[dict],
    cache: dict[str, list[int]],
) -> tuple[str, list[int]]:
    """Run one dimension and return (cache_key, successful_ids). Cache is NOT
    updated here — the caller does that after all concurrent dimensions finish."""
    running_all = args.dimension == "all"

    only_scores = _parse_only_scores(args.only_scores)
    if only_scores is not None:
        valid = valid_scores_for(key, args)
        if running_all:
            only_scores = only_scores & valid
            if not only_scores:
                cache_key = cache_key_for(key, cfg, args)
                print(
                    f"[{cache_key}] Skipping — none of the requested --only-scores apply "
                    f"to this dimension (valid: {sorted(valid)})."
                )
                return cache_key, []
        elif not only_scores <= valid:
            sys.exit(
                f"Error: --only-scores {sorted(only_scores)} invalid for dimension "
                f"{key!r} (valid scores: {sorted(valid)})."
            )

    if cfg.kind == "speaker_identity_consistency":
        return await _run_speaker_identity_dimension(key, cfg, args, utterances, cache, only_scores)
    if key == "expressiveness":
        return await run_expressiveness_dimension(cfg, args, cache, only_scores)
    if key == "prosodic_boundary":
        return await run_prosodic_boundary_dimension(cfg, args, cache, only_scores)

    n = cfg.num_utterances if running_all or args.num is None else args.num
    if not cfg.sentence_level:
        voice = cfg.voice_id if (running_all or args.voice is None) else args.voice
        if not voice:
            sys.exit(f"Error: no Cartesia voice_id configured for dimension '{key}'")
    else:
        voice = None

    output_dir = args.output / key
    cache_key = cache_key_for(key, cfg, args)
    dataset_tag = _dataset_tag(args)

    # Non-sentence-level, non-speech-rate dimensions need a prompt file.
    prompt_path = args.prompts_dir / f"{key}.txt"
    if not cfg.sentence_level and cfg.pipeline_type != "speech_rate" and not prompt_path.exists():
        sys.exit(f"Error: prompt file not found: {prompt_path}")

    exclude_ids = set(cache.get(cache_key, []))

    if cfg.pipeline_type == "speech_rate":
        # Speech rate doesn't need a target word — just pick utterances.
        available = [u for u in utterances if u["id"] not in exclude_ids]
        if len(available) < n:
            print(
                f"  Warning: only {len(available)} unused utterances available "
                f"(requested {n}). Consider --reset-cache to reuse utterances."
            )
        items = [
            {"id": u["id"], "text": u["text"]}
            for u in random.sample(available, min(n, len(available)))
        ]
    else:
        items = select_items(
            utterances, n, cfg.require_polysyllabic, exclude_ids,
            needs_target_word=cfg.needs_target_word,
            sentence_level=cfg.sentence_level,
            skip_candidate_filter=bool(getattr(args, "category", None)),
            min_words=getattr(args, "min_words", None),
            max_words=getattr(args, "max_words", None),
        )

    if not items:
        print(f"[{cache_key}] No unused utterances available — skipping. Use --reset-cache to reuse.")
        return cache_key, []

    cached_count = len(exclude_ids)
    print(
        f"\nSelected {len(items)} utterances for {cfg.display_name} "
        f"({cached_count} already used, {len(utterances) - cached_count - len(items)} remaining):\n"
    )
    for item in items:
        if "target_word" in item and item["target_word"]:
            print(f"  [{item['id']:>3}] \"{item['text']}\"  ->  target: \"{item['target_word']}\"")
        elif "target_word" in item:
            print(f"  [{item['id']:>3}] \"{item['text']}\"  ->  target: (sentence)")
        else:
            print(f"  [{item['id']:>3}] \"{item['text']}\"")
    print()

    if cfg.pipeline_type == "speech_rate":
        successful_ids = await run_speech_rate_pipeline(
            items              = items,
            output_dir         = output_dir,
            cartesia_client    = make_cartesia_client(),
            model_id           = args.model,
            voice_id           = voice,
            output_format      = CARTESIA_OUTPUT_FORMAT,
            tts_concurrency    = args.tts_concurrency,
            psola_only         = args.psola_only,
            only_scores        = only_scores,
        )
    elif cfg.sentence_level:
        # Intonation: ElevenLabs-synthesized base + Praat F0 manipulation.
        validate_env_elevenlabs()
        el_client = make_elevenlabs_client()
        successful_ids = await run_intonation_pipeline(
            dimension_name    = cfg.display_name,
            items             = items,
            output_dir        = output_dir,
            elevenlabs_client = el_client,
            voice_id          = cfg.elevenlabs_voice_id,
            tts_concurrency   = args.tts_concurrency,
            stack_rate        = args.stack_rate,
            dataset_tag       = dataset_tag,
            only_scores       = only_scores,
        )
    elif cfg.uses_artifact_pipeline:
        # Human plausibility: artifact-injection pipeline.
        successful_ids = await run_artifact_pipeline(
            dimension_name     = cfg.display_name,
            items              = items,
            prompt_template    = prompt_path.read_text(),
            output_dir         = output_dir,
            openai_client      = make_openai_client(),
            deployment         = args.deployment,
            cartesia_client    = make_cartesia_client(),
            model_id           = args.model,
            voice_id           = voice,
            output_format      = CARTESIA_OUTPUT_FORMAT,
            openai_concurrency = args.openai_concurrency,
            tts_concurrency    = args.tts_concurrency,
            only_scores        = only_scores,
        )
    else:
        if cfg.tts_engine == "elevenlabs":
            validate_env_elevenlabs()
            el_client = make_elevenlabs_client()
            ca_client = None
        else:
            el_client = None
            ca_client = make_cartesia_client()

        successful_ids = await run_two_phase_pipeline(
            dimension_name     = cfg.display_name,
            items              = items,
            prompt_template    = prompt_path.read_text(),
            output_dir         = output_dir,
            openai_client      = make_openai_client(),
            deployment         = args.deployment,
            cartesia_client    = ca_client,
            model_id           = args.model,
            voice_id           = voice,
            output_format      = CARTESIA_OUTPUT_FORMAT,
            elevenlabs_client  = el_client,
            elevenlabs_voice_id= cfg.elevenlabs_voice_id,
            openai_concurrency = args.openai_concurrency,
            tts_concurrency    = args.tts_concurrency,
            dataset_tag        = dataset_tag,
            only_scores        = only_scores,
        )

    failed = len(items) - len(successful_ids)
    if failed:
        print(f"  [{cache_key}] Note: {failed} utterance(s) failed — not cached, will retry next run.")

    return cache_key, successful_ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    needs_openai, needs_cartesia = _env_requirements(args.dimension, args)
    validate_env(needs_openai=needs_openai, needs_cartesia=needs_cartesia)
    if args.category and args.dataset != "emergent":
        sys.exit("--category only applies with --dataset emergent")
    if args.dataset == "emergent" and not args.utterances:
        from shared.emergent_dataset import load_emergent_utterances, HF_REPO
        utterances = load_emergent_utterances(args.category)
        label = f"emergent / {args.category}" if args.category else "emergent"
        print(f"Dataset: {label}  ({HF_REPO}, {len(utterances)} utterances)")
    else:
        utterances_path = args.utterances if args.utterances else DATASETS[args.dataset]
        print(f"Dataset: {args.dataset if not args.utterances else args.utterances}  ({utterances_path})")
        utterances = load_utterances(utterances_path)

    cache = load_cache()

    if args.rebuild_cache:
        cache = rebuild_cache_from_disk(args.output, _dataset_tag(args))
        save_cache(cache)
        for key, ids in cache.items():
            print(f"Cache rebuilt — {key}: {len(ids)} IDs with confirmed audio")
        print()

    if args.reset_cache:
        dims_to_reset = list(DIMENSIONS.keys()) if args.dimension == "all" else [args.dimension]
        keys_to_reset = [cache_key_for(dim, DIMENSIONS[dim], args) for dim in dims_to_reset]
        for key in keys_to_reset:
            cache.pop(key, None)
        save_cache(cache)
        print(f"Cache cleared for: {', '.join(keys_to_reset)}\n")

    for i in range(1, args.samples + 1):
        if args.seed is not None:
            # Seed Python and numpy RNGs together — numpy drives F0-manipulation jitter.
            random.seed(args.seed + i)
            np.random.seed(args.seed + i)

        if args.dimension == "all":
            # All dimensions are independent — run concurrently.
            # Expressiveness uses a different scoring model (one score per item) and
            # always loads Harvard internally; prosodic_boundary uses its own fixed
            # preset dataset. Both are excluded from --dimension all batch runs.
            results = await asyncio.gather(*(
                run_dimension(key, cfg, args, utterances, cache)
                for key, cfg in DIMENSIONS.items()
                if key not in ("expressiveness", "prosodic_boundary")
            ))
        else:
            results = [await run_dimension(
                args.dimension, DIMENSIONS[args.dimension], args, utterances, cache
            )]

        # Update cache once after all dimensions in this sample complete
        for key, successful_ids in results:
            if successful_ids:
                used = cache.get(key, [])
                used.extend(successful_ids)
                cache[key] = used
        save_cache(cache)

    generate_excel(args.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
