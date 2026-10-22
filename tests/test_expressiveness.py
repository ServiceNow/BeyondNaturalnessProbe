"""Tests for the expressiveness dimension and the generate_tts generation_config param."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.expressiveness import _folder_name, run_expressiveness_pipeline
from shared.tts_utils import generate_tts


# ---------------------------------------------------------------------------
# _folder_name (pure)
# ---------------------------------------------------------------------------

class TestFolderName:
    def test_zero_pads_uid_to_3_digits(self):
        assert _folder_name(1, 7) == "1_007"

    def test_score_prefix(self):
        assert _folder_name(3, 100) == "3_100"

    def test_four_digit_uid_not_truncated(self):
        assert _folder_name(2, 1234) == "2_1234"


# ---------------------------------------------------------------------------
# generate_tts — generation_config backward compat
# ---------------------------------------------------------------------------

def _mock_cartesia_client():
    response = AsyncMock()
    response.write_to_file = AsyncMock()
    client = MagicMock()
    client.tts.generate = AsyncMock(return_value=response)
    return client


class TestGenerateTtsGenerationConfig:
    def test_speed_only_forwarded_as_generation_config(self, tmp_path):
        client = _mock_cartesia_client()
        asyncio.run(generate_tts(
            client, "Hello world.", tmp_path / "out.wav",
            model_id="sonic-3", voice_id="v1", speed=1.2,
        ))
        _, kw = client.tts.generate.call_args
        assert kw["generation_config"] == {"speed": 1.2}

    def test_generation_config_only_forwarded_as_is(self, tmp_path):
        client = _mock_cartesia_client()
        asyncio.run(generate_tts(
            client, "Hello world.", tmp_path / "out.wav",
            model_id="sonic-3", voice_id="v1",
            generation_config={"emotion": "confident"},
        ))
        _, kw = client.tts.generate.call_args
        assert kw["generation_config"] == {"emotion": "confident"}

    def test_generation_config_speed_takes_precedence_over_speed_arg(self, tmp_path):
        client = _mock_cartesia_client()
        asyncio.run(generate_tts(
            client, "Hello world.", tmp_path / "out.wav",
            model_id="sonic-3", voice_id="v1", speed=0.5,
            generation_config={"speed": 0.8, "emotion": "bored"},
        ))
        _, kw = client.tts.generate.call_args
        assert kw["generation_config"]["speed"] == 0.8  # generation_config wins
        assert "emotion" in kw["generation_config"]

    def test_no_speed_no_config_omits_generation_config(self, tmp_path):
        client = _mock_cartesia_client()
        asyncio.run(generate_tts(
            client, "Hello world.", tmp_path / "out.wav",
            model_id="sonic-3", voice_id="v1",
        ))
        _, kw = client.tts.generate.call_args
        assert "generation_config" not in kw

    def test_emotion_and_speed_merged_when_no_conflict(self, tmp_path):
        client = _mock_cartesia_client()
        asyncio.run(generate_tts(
            client, "Hello world.", tmp_path / "out.wav",
            model_id="sonic-3", voice_id="v1", speed=0.9,
            generation_config={"emotion": "bored"},
        ))
        _, kw = client.tts.generate.call_args
        gen = kw["generation_config"]
        assert gen["emotion"] == "bored"
        assert gen["speed"] == 0.9  # speed arg fills in since not in generation_config


# ---------------------------------------------------------------------------
# run_expressiveness_pipeline — smoke test with mocked Cartesia
# ---------------------------------------------------------------------------

def _make_item(uid: int, score: int) -> dict:
    labels = {1: "over", 2: "under", 3: "natural"}
    return {
        "id":         uid,
        "text":       f"Sentence {uid}.",
        "score":      score,
        "label":      labels[score],
        "voice_id":   "fake-voice-id",
        "voice_name": "Fake Voice",
        "versions": [{
            "score":             score,
            "label":             labels[score],
            "tts_emotion":       "confident" if score == 1 else None,
            "speed":             0.8 if score == 3 else None,
            "psola_flatten":     False,
            "voice_id":          "fake-voice-id",
            "voice_name":        "Fake Voice",
            "sentence":          f"Sentence {uid}.",
            "error_description": "test",
        }],
    }


def test_run_expressiveness_pipeline_returns_successful_ids(tmp_path):
    items = [_make_item(1, 1), _make_item(2, 3)]
    client = _mock_cartesia_client()

    successful = asyncio.run(run_expressiveness_pipeline(
        items, tmp_path, client, model_id="sonic-3", tts_concurrency=2,
    ))

    assert set(successful) == {1, 2}


def test_run_expressiveness_pipeline_writes_manifest(tmp_path):
    items = [_make_item(42, 3)]
    client = _mock_cartesia_client()

    asyncio.run(run_expressiveness_pipeline(
        items, tmp_path, client, model_id="sonic-3",
    ))

    manifest_path = tmp_path / "3_042" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["utterance_id"] == 42
    assert manifest["score"] == 3


def test_run_expressiveness_pipeline_skips_psola_when_false(tmp_path):
    items = [_make_item(5, 1)]  # score 1 = over, psola_flatten=False
    client = _mock_cartesia_client()

    with patch("shared.expressiveness.asyncio.to_thread") as mock_thread:
        asyncio.run(run_expressiveness_pipeline(
            items, tmp_path, client, model_id="sonic-3",
        ))
        mock_thread.assert_not_called()


def test_run_expressiveness_pipeline_calls_psola_for_score_2(tmp_path):
    items = [_make_item(7, 2)]
    items[0]["versions"][0]["psola_flatten"] = True
    client = _mock_cartesia_client()

    # pyworld/soundfile are optional deps not installed in CI; fake them so the
    # lazy `from .flatten_expressiveness import flatten_one` succeeds, then
    # intercept asyncio.to_thread to verify flatten_one is scheduled.
    fake_modules = {"pyworld": MagicMock(), "soundfile": MagicMock()}
    with patch.dict("sys.modules", fake_modules):
        with patch("shared.expressiveness.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            asyncio.run(run_expressiveness_pipeline(
                items, tmp_path, client, model_id="sonic-3",
            ))
            mock_thread.assert_called_once()
            args = mock_thread.call_args[0]
            assert args[0].__name__ == "flatten_one"
