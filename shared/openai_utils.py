"""
OpenAI helpers — prompt construction and response parsing.

Identical logic was duplicated across both pipelines; extracted here once.
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable

from openai import AsyncAzureOpenAI, AsyncOpenAI

# Type alias for a Phase 1 callable: takes an item dict, returns a manifest dict or None.
VariantGenerator = Callable[[dict], Awaitable[dict | None]]


def build_prompt(template: str, sentence: str, target_word: str, **extra: str) -> str:
    """
    Fill in the two standard placeholders (`{{INPUT_UTTERANCE}}`, `{{TARGET_WORD}}`)
    and any additional `{{KEY}}` placeholders passed as keyword args.
    """
    result = (
        template
        .replace("{{INPUT_UTTERANCE}}", sentence)
        .replace("{{TARGET_WORD}}", target_word)
    )
    for key, val in extra.items():
        result = result.replace(f"{{{{{key}}}}}", val)
    return result


async def call_openai(
    client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment: str,
    prompt: str,
) -> dict:
    """
    Send *prompt* to OpenAI (standard or Azure) and return the parsed JSON response.

    Strips markdown code fences if the model wraps its output in them.
    Raises ValueError if the model returns no content.
    Raises json.JSONDecodeError if the response isn't valid JSON.
    """
    response = await client.chat.completions.create(
        model=deployment,
        max_completion_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content or not content.strip():
        raise ValueError(f"Model returned no content (finish_reason={choice.finish_reason})")
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)
