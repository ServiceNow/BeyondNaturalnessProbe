"""
Utterance loading, target-word selection, and output naming.

Used by all three dimensions:
  - lexical_stress / phonetic_accuracy select a single target word per utterance
    via ``pick_target_word``; the word-level flag ``require_polysyllabic``
    differs between them.
  - intonation operates at sentence level and selects whole utterances via
    ``is_intonation_candidate`` (clean declaratives only).

``output_names`` produces the {folder_name, file_prefix} pair every pipeline
uses so on-disk layout is consistent: ``{dataset}_{id:03d}[_{word}]``.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

# Minimum character length for a word to be a candidate target
MIN_WORD_LENGTH = 4

# Function words and very common short words that make poor TTS targets
SKIP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "but", "for", "nor", "or", "so", "yet",
    "in", "on", "at", "to", "by", "of", "up", "is", "it", "he", "she",
    "we", "as", "if", "do", "no", "not", "all", "are", "was", "were",
    "has", "had", "his", "her", "its", "our", "who", "how", "can",
    "did", "get", "got", "let", "may", "own", "say", "too", "use", "good", "know",
    "way", "you", "him", "out", "now", "new", "old", "one", "two",
    "few", "than", "then", "them", "they", "this", "that", "with",
    "will", "from", "just", "been", "have", "each", "make", "like",
    "very", "when", "what", "some", "into", "your", "which", "their",
    "would", "there", "about", "could", "other", "after", "come",
})

_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

# Word-count bounds for sentences suitable for sentence-level intonation manipulation.
# Too short → contour too brief to carry a clear terminal rise/fall.
# Too long → contour differences diluted across multiple intonational phrases.
_INTONATION_MIN_WORDS = 4
_INTONATION_MAX_WORDS = 12


def output_names(uid: int, target_word: str, dataset_tag: str = "") -> tuple[str, str]:
    """
    Return (folder_name, file_prefix) for an utterance's output directory.

    Format: "{dataset}_{id:03d}[_{word}]" — dataset prefix omitted if empty,
    target word suffix omitted for sentence-level dimensions.

        uid=2,  target_word="eclipse", dataset_tag="mozilla"  →  "mozilla_002_eclipse"
        uid=48, target_word="",        dataset_tag="harvard"  →  "harvard_048"
        uid=48, target_word="",        dataset_tag=""         →  "048"

    Folder name and file prefix are identical so output files take the form
    {folder}/{folder}_{score}.wav.
    """
    parts: list[str] = []
    if dataset_tag:
        parts.append(dataset_tag)
    parts.append(f"{uid:03d}")
    word = re.sub(r"[^a-zA-Z0-9]", "", target_word.lower())
    if word:
        parts.append(word)
    name = "_".join(parts)
    return name, name


def is_intonation_candidate(text: str) -> bool:
    """
    True iff *text* is a clean declarative sentence suitable for the
    intonation pipeline.

    The F0 manipulations operate on the whole utterance contour, so we want
    a single-clause declarative — no internal phrase boundaries from commas,
    dashes, or other phrasal punctuation that would already shape the contour.

    Rejects sentences that:
    - Don't end with a period (already questions, exclamations, or fragments).
    - Contain phrasal punctuation (comma, colon, semicolon, em-dash, en-dash,
      mid-sentence ``?``/``!``). Word-internal hyphens are allowed — they're
      part of compound words, not phrase boundaries.
    - Fall outside the [MIN, MAX] word range.
    """
    t = text.strip()
    if not t.endswith("."):
        return False
    body = t[:-1]  # strip the terminal period before checking for other punctuation
    if any(c in body for c in ",;:?!—–"):
        return False
    words = re.findall(r"[A-Za-z']+", t)
    return _INTONATION_MIN_WORDS <= len(words) <= _INTONATION_MAX_WORDS


def _syllable_count(word: str) -> int:
    """Rough syllable count based on vowel clusters."""
    count = len(_VOWEL_RE.findall(word))
    if word.lower().endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def load_utterances(path: Path, *, category: str | None = None) -> list[dict]:
    """Load utterances from the JSON file at *path*.

    If *category* is given, keep only entries where entry.get("category") == category.
    """
    with open(path) as f:
        items = json.load(f)["utterances"]
    if category is None:
        return items
    return [u for u in items if u.get("category") == category]


def pick_target_word(text: str, *, require_polysyllabic: bool = False) -> str | None:
    """
    Pick a random content word from *text* to use as the pipeline target.

    Args:
        text: The full sentence string.
        require_polysyllabic: If True, only words with ≥ 2 syllables qualify
            (needed for lexical-stress pipeline; phonetic-accuracy doesn't care).

    Returns:
        A word string, or None if no suitable candidate exists.
    """
    words = re.findall(r"[A-Za-z']+", text)

    def _is_polysyllabic(w: str) -> bool:
        return _syllable_count(w) >= 2

    def _candidates(min_len: int, must_be_content: bool, check_polysyllabic: bool) -> list[str]:
        return [
            w for w in words
            if len(w) >= min_len
            and (w.lower() not in SKIP_WORDS if must_be_content else True)
            and not w[0].isupper()
            and (_is_polysyllabic(w) if check_polysyllabic else True)
        ]

    # Tier 1: strict — long content word, polysyllabic if required
    pool = _candidates(MIN_WORD_LENGTH, must_be_content=True, check_polysyllabic=require_polysyllabic)

    # Tier 2: relax minimum length
    if not pool:
        pool = _candidates(3, must_be_content=True, check_polysyllabic=require_polysyllabic)

    # Tier 3: drop skip-word filter — only used when polysyllabic is not required
    # (lexical_stress stops at Tier 2; phonetic_accuracy original had this third fallback)
    if not pool and not require_polysyllabic:
        pool = _candidates(3, must_be_content=False, check_polysyllabic=False)

    return random.choice(pool) if pool else None
