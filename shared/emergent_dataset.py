"""
Lazy loader for the bosonai/EmergentTTS-Eval HuggingFace dataset.

Emits the same ``[{"id": int, "text": str}, ...]`` shape as the local JSON
loaders so it drops into the pipeline interchangeably. IDs are assigned
0..N-1 after filtering, so a filter change produces a disjoint ID space
(see pipeline cache-key scheme in pipeline.py: the cache tag includes the
category so this is safe).

Heavy imports (``datasets``) happen only when the loader is actually
called — importing this module has no runtime cost.
"""
from __future__ import annotations

from functools import lru_cache

HF_REPO = "bosonai/EmergentTTS-Eval"
# Pinned dataset commit (last modified 2025-06-23) so runs are reproducible.
HF_REVISION = "a7406fa315a1df2a5ffcc1a782404648bf84fbd3"

CATEGORIES: tuple[str, ...] = (
    "Questions",
    "Foreign Words",
    "Paralinguistics",
    "Emotions",
    "Syntactic Complexity",
    "Pronunciation",
)


def _canon(cat: str) -> str:
    """Normalize a category string for case/whitespace/separator-insensitive match."""
    return " ".join(cat.strip().lower().replace("_", " ").replace("-", " ").split())


@lru_cache(maxsize=1)
def _load_raw() -> list[dict]:
    """Fetch and cache the dataset rows in memory (audio column dropped)."""
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, split="train", revision=HF_REVISION)
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])
    return [
        {"category": r["category"], "text": r["text_to_synthesize"]}
        for r in ds
    ]


def load_emergent_utterances(category: str | None = None) -> list[dict]:
    """Return ``[{"id": int, "text": str}, ...]`` for the EmergentTTS-Eval split.

    If ``category`` is supplied, rows are filtered to that category first
    (case/space/underscore-insensitive). IDs are re-numbered 0..N-1 after
    filtering.
    """
    rows = _load_raw()
    if category:
        wanted = _canon(category)
        rows = [r for r in rows if _canon(r["category"]) == wanted]
        if not rows:
            raise ValueError(
                f"No EmergentTTS-Eval rows match category {category!r}. "
                f"Available: {', '.join(CATEGORIES)}"
            )
    return [{"id": i, "text": r["text"]} for i, r in enumerate(rows)]
