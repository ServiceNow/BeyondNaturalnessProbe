"""Tests for shared/clients.py: dual OpenAI client selection + dimension-aware validation."""
from __future__ import annotations

import pytest

from shared.clients import make_openai_client, validate_env, validate_env_elevenlabs


AZURE_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
AZURE_KEY = "AZURE_OPENAI_API_KEY"
OPENAI_KEY = "OPENAI_API_KEY"
CARTESIA_KEY = "CARTESIA_API_KEY"
ELEVENLABS_KEY = "ELEVEN_LABS_API_KEY"

_ALL_VARS = [AZURE_ENDPOINT, AZURE_KEY, OPENAI_KEY, CARTESIA_KEY, ELEVENLABS_KEY]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with none of the relevant env vars set."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


class TestMakeOpenaiClient:
    def test_azure_creds_present_returns_azure_client(self, monkeypatch):
        monkeypatch.setenv(AZURE_ENDPOINT, "https://example.openai.azure.com/")
        monkeypatch.setenv(AZURE_KEY, "azure-secret")
        client = make_openai_client()
        assert type(client).__name__ == "AsyncAzureOpenAI"

    def test_standard_openai_key_returns_standard_client(self, monkeypatch):
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        client = make_openai_client()
        assert type(client).__name__ == "AsyncOpenAI"

    def test_azure_takes_priority_when_both_set(self, monkeypatch):
        monkeypatch.setenv(AZURE_ENDPOINT, "https://example.openai.azure.com/")
        monkeypatch.setenv(AZURE_KEY, "azure-secret")
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        client = make_openai_client()
        assert type(client).__name__ == "AsyncAzureOpenAI"

    def test_partial_azure_creds_falls_back_to_standard(self, monkeypatch):
        # Only the endpoint set, not the key — shouldn't count as "Azure configured".
        monkeypatch.setenv(AZURE_ENDPOINT, "https://example.openai.azure.com/")
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        client = make_openai_client()
        assert type(client).__name__ == "AsyncOpenAI"

    def test_no_creds_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            make_openai_client()
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)


class TestValidateEnv:
    def test_missing_cartesia_exits(self, monkeypatch):
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        with pytest.raises(SystemExit) as exc_info:
            validate_env(needs_openai=True, needs_cartesia=True)
        assert "CARTESIA_API_KEY" in str(exc_info.value)

    def test_missing_openai_exits(self, monkeypatch):
        monkeypatch.setenv(CARTESIA_KEY, "cartesia-fake")
        with pytest.raises(SystemExit) as exc_info:
            validate_env(needs_openai=True, needs_cartesia=True)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_needs_openai_false_ignores_missing_openai_creds(self, monkeypatch):
        monkeypatch.setenv(CARTESIA_KEY, "cartesia-fake")
        # Should not raise even though no OpenAI creds of any kind are set.
        validate_env(needs_openai=False, needs_cartesia=True)

    def test_needs_cartesia_false_ignores_missing_cartesia(self, monkeypatch):
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        # Should not raise even though CARTESIA_API_KEY is unset (e.g. intonation-only run).
        validate_env(needs_openai=True, needs_cartesia=False)

    def test_neither_needed_never_exits(self):
        validate_env(needs_openai=False, needs_cartesia=False)

    def test_both_present_passes(self, monkeypatch):
        monkeypatch.setenv(OPENAI_KEY, "sk-fake")
        monkeypatch.setenv(CARTESIA_KEY, "cartesia-fake")
        validate_env(needs_openai=True, needs_cartesia=True)

    def test_azure_creds_satisfy_needs_openai(self, monkeypatch):
        monkeypatch.setenv(AZURE_ENDPOINT, "https://example.openai.azure.com/")
        monkeypatch.setenv(AZURE_KEY, "azure-secret")
        monkeypatch.setenv(CARTESIA_KEY, "cartesia-fake")
        validate_env(needs_openai=True, needs_cartesia=True)

    def test_missing_both_lists_both_in_message(self, monkeypatch):
        with pytest.raises(SystemExit) as exc_info:
            validate_env(needs_openai=True, needs_cartesia=True)
        msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in msg
        assert "CARTESIA_API_KEY" in msg


class TestValidateEnvElevenlabs:
    def test_missing_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            validate_env_elevenlabs()
        assert "ELEVEN_LABS_API_KEY" in str(exc_info.value)

    def test_present_passes(self, monkeypatch):
        monkeypatch.setenv(ELEVENLABS_KEY, "el-fake")
        validate_env_elevenlabs()
