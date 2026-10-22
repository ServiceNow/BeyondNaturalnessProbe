"""
API client factories and environment validation.

All pipelines call validate_env() at startup, then use make_openai_client(),
make_cartesia_client(), or make_elevenlabs_client() to get configured clients.
"""
from __future__ import annotations

import os
import sys

from openai import AsyncAzureOpenAI, AsyncOpenAI
from cartesia import AsyncCartesia
from elevenlabs.client import AsyncElevenLabs

_AZURE_ENV_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
_OPENAI_ENV_VAR = "OPENAI_API_KEY"
_CARTESIA_ENV_VAR = "CARTESIA_API_KEY"
_ELEVENLABS_ENV_VAR = "ELEVEN_LABS_API_KEY"


def _has_openai_creds() -> bool:
    return bool(os.environ.get(_OPENAI_ENV_VAR)) or all(
        os.environ.get(v) for v in _AZURE_ENV_VARS
    )


def validate_env(*, needs_openai: bool = True, needs_cartesia: bool = True) -> None:
    """Exit with a clear message if a required env var for this run is missing.

    Only checks what the selected dimension actually uses — e.g. `speech_rate`
    never calls OpenAI, so it shouldn't be blocked by missing OpenAI creds.
    """
    missing = []
    if needs_cartesia and not os.environ.get(_CARTESIA_ENV_VAR):
        missing.append(_CARTESIA_ENV_VAR)
    if needs_openai and not _has_openai_creds():
        missing.append(f"{_OPENAI_ENV_VAR} (or {' + '.join(_AZURE_ENV_VARS)} for Azure OpenAI)")
    if missing:
        sys.exit(f"Error: missing environment variables: {', '.join(missing)}")


def validate_env_elevenlabs() -> None:
    """Exit with a clear message if the ElevenLabs API key is missing."""
    if not os.environ.get(_ELEVENLABS_ENV_VAR):
        sys.exit(f"Error: missing environment variable: {_ELEVENLABS_ENV_VAR}")


def make_openai_client() -> AsyncAzureOpenAI | AsyncOpenAI:
    """Prefer Azure OpenAI if configured, else fall back to standard OpenAI."""
    if all(os.environ.get(v) for v in _AZURE_ENV_VARS):
        return AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2025-03-01-preview",
        )
    if os.environ.get(_OPENAI_ENV_VAR):
        return AsyncOpenAI(api_key=os.environ[_OPENAI_ENV_VAR])
    sys.exit(
        f"Error: no OpenAI credentials found. Set {_OPENAI_ENV_VAR}, "
        f"or {' + '.join(_AZURE_ENV_VARS)} for Azure OpenAI."
    )


def make_cartesia_client() -> AsyncCartesia:
    return AsyncCartesia(api_key=os.environ["CARTESIA_API_KEY"])


def make_elevenlabs_client() -> AsyncElevenLabs:
    return AsyncElevenLabs(api_key=os.environ[_ELEVENLABS_ENV_VAR])
