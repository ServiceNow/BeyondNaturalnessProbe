"""
Prosodic boundary strategy implementations.

Each strategy takes a sentence string and returns a manifest-shaped dict:
    {"versions": [{score: 0, error_description, sentence},
                  {score: 1, error_description, sentence}]}

The score-1 entry is always the original sentence verbatim with a canonical
description. Programmatic strategies build both entries themselves; the
GPT-driven strategies build only the score-0 entry and let the caller
append the canonical score-1 entry — see make_prosodic_generator.
"""
from __future__ import annotations

import json

CANONICAL_DESCRIPTION = "Canonical sentence with natural prosodic boundaries."


def _canonical_version(text: str) -> dict:
    return {
        "score": 1,
        "error_description": CANONICAL_DESCRIPTION,
        "sentence": text,
    }


def omit_spaces(text: str) -> dict:
    """Strip ALL whitespace from the sentence for the score-0 version."""
    stripped = "".join(text.split())
    return {
        "versions": [
            {
                "score": 0,
                "error_description": "All inter-word whitespace removed; words run together.",
                "sentence": stripped,
            },
            _canonical_version(text),
        ]
    }


import re

# Internal punctuation to strip. Terminal marks (.?!) are preserved separately.
_INTERNAL_PUNCT = re.compile(r"[,;:—–\-…]")


def omit_punctuation(text: str) -> dict:
    """
    Strip internal punctuation but preserve a terminal .?! if present.
    Terminal punctuation gives Cartesia a sentence boundary; internal marks
    cue intra-sentence pauses that we want to remove.
    """
    if not text:
        return {"versions": [
            {"score": 0, "error_description": "Empty input.", "sentence": ""},
            _canonical_version(""),
        ]}

    terminal = ""
    body = text
    if text[-1] in ".?!":
        terminal = text[-1]
        body = text[:-1]
    stripped = _INTERNAL_PUNCT.sub("", body) + terminal
    return {
        "versions": [
            {
                "score": 0,
                "error_description": "Internal punctuation (, ; : — – … -) removed; terminal mark preserved.",
                "sentence": stripped,
            },
            _canonical_version(text),
        ]
    }


from pathlib import Path

from .openai_utils import VariantGenerator, build_prompt, call_openai

STRATEGIES = ("extra_punctuation", "omit_spaces", "omit_punctuation", "preset")
_GPT_STRATEGIES = {"extra_punctuation"}
_PROGRAMMATIC = {
    "omit_spaces": omit_spaces,
    "omit_punctuation": omit_punctuation,
}

# Pre-defined 60-sample distribution for preset strategy
_PRESET_DEFINITIONS_FILE = Path(__file__).parent.parent / "data" / "prosodic_boundary_preset_definitions.json"


def _load_preset_samples() -> dict[int, dict]:
    """Load the preset distribution and index by utterance ID."""
    with open(_PRESET_DEFINITIONS_FILE) as f:
        data = json.load(f)
    return {u["id"]: u for u in data["utterances"]}


def make_prosodic_generator(
    *,
    client,                       # AsyncAzureOpenAI | AsyncOpenAI; unused for programmatic strategies
    deployment: str,              # ignored for programmatic strategies
    prompt_template: str = "",    # unused — prosodic loads its own prompt file from disk
    dataset: str,
    strategy: str,
    category: str | None = None,
    prompts_dir: Path | None = None,
    **_ignored,                   # tolerate extra kwargs from a uniform call site
) -> VariantGenerator:
    """
    Build the Phase 1 callable for the prosodic_boundary dimension.

    `strategy` selects the score-0 transform. GPT-driven strategies require
    `client`, `deployment`, and `prompts_dir`. Programmatic strategies use
    only the input text and ignore the OpenAI args.

    The returned callable wraps the strategy's score-0 dict with a uniform
    canonical score-1 entry and returns a complete manifest dict
    (utterance_id, original_text, dataset, [category,] strategy, versions).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown prosodic strategy: {strategy!r}. Valid: {STRATEGIES}")

    if strategy == "preset":
        samples = _load_preset_samples()

        async def generate(item: dict) -> dict | None:
            sample = samples.get(item["id"])
            if sample is None:
                return None
            version = {
                "score": sample["score"],
                "error_description": sample["description"],
                "sentence": sample["tts_text"],
            }
            return _build_manifest(item, dataset, category, strategy, [version])

        return generate

    if strategy in _GPT_STRATEGIES:
        if prompts_dir is None:
            raise ValueError(f"Strategy {strategy!r} requires prompts_dir.")
        prompt_path = prompts_dir / f"prosodic_boundary_{strategy}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file missing: {prompt_path}")
        loaded_prompt = prompt_path.read_text()

        async def generate(item: dict) -> dict | None:
            filled = build_prompt(loaded_prompt, item["text"], "")
            try:
                gpt_out = await call_openai(client, deployment, filled)
            except Exception:
                raise
            if not gpt_out or "sentence" not in gpt_out:
                return None
            score_zero = {
                "score": 0,
                "error_description": gpt_out.get("error_description", ""),
                "sentence": gpt_out["sentence"],
            }
            return _build_manifest(item, dataset, category, strategy,
                                   [score_zero, _canonical_version(item["text"])])

        return generate

    # Programmatic
    transform = _PROGRAMMATIC[strategy]

    async def generate(item: dict) -> dict | None:
        variants = transform(item["text"])
        return _build_manifest(item, dataset, category, strategy, variants["versions"])

    return generate


def _build_manifest(item: dict, dataset: str, category: str | None,
                    strategy: str, versions: list[dict]) -> dict:
    entry = {
        "utterance_id": item["id"],
        "original_text": item["text"],
        "dataset": dataset,
        "strategy": strategy,
        "versions": versions,
    }
    if category is not None:
        entry["category"] = category
    return entry
