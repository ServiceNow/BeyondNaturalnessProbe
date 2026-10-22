"""
Tests for shared/artifact_pipeline.py: run_artifact_pipeline (the human_plausibility dimension).

Score 1 (clean) never depends on the GPT-picked artifact recipe — so
only_scores={1} should skip the OpenAI call entirely. Score 0 needs the clean
audio as its literal DSP input, so requesting score 0 alone still synthesizes
the clean WAV internally (then deletes it, since it wasn't asked for).
"""
from __future__ import annotations

import asyncio
import json

from shared.artifact_pipeline import run_artifact_pipeline
from tests.helpers import mock_cartesia_client, mock_openai_client


FULL_PAYLOAD = {
    "target_word": "word",
    "versions": [
        {
            "score": 1,
            "sentence": "The word is here.",
            "error_description": "Clean reference — no artifacts applied.",
            "artifacts": [],
        },
        {
            "score": 0,
            "sentence": "The word is here.",
            "error_description": "A harsh clipping distorts the whole sentence.",
            "artifacts": [{"type": "hard_clipping", "location": "throughout", "severity": "moderate"}],
        },
    ],
}


def _item(uid=1, text="The word is here.", target_word="word"):
    return {"id": uid, "text": text, "target_word": target_word}


def _run(items, output_dir, *, only_scores=None):
    openai_client = mock_openai_client(FULL_PAYLOAD)
    cartesia_client = mock_cartesia_client()
    successful_ids = asyncio.run(run_artifact_pipeline(
        dimension_name="human plausibility",
        items=items,
        prompt_template="{{INPUT_UTTERANCE}} / {{TARGET_WORD}} / {{ARTIFACT_SELECTION}}",
        output_dir=output_dir,
        openai_client=openai_client,
        deployment="gpt-5",
        cartesia_client=cartesia_client,
        model_id="sonic-3",
        voice_id="fake-voice",
        only_scores=only_scores,
    ))
    return successful_ids, openai_client, cartesia_client


class TestFullRun:
    def test_default_delivers_both_scores(self, tmp_path):
        successful_ids, openai_client, cartesia_client = _run([_item()], tmp_path)
        assert successful_ids == [1]
        folder = tmp_path / "001_word"
        assert (folder / "word_1.wav").exists()
        assert (folder / "word_0.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {0, 1}
        openai_client.chat.completions.create.assert_called_once()
        # Clean audio is synthesized exactly once and shared by both versions.
        cartesia_client.tts.generate.assert_called_once()


class TestOnlyScoreOneSkipsOpenAI:
    def test_only_score_1_never_calls_openai(self, tmp_path):
        successful_ids, openai_client, cartesia_client = _run([_item()], tmp_path, only_scores={1})
        assert successful_ids == [1]
        folder = tmp_path / "001_word"
        assert (folder / "word_1.wav").exists()
        assert not (folder / "word_0.wav").exists()
        openai_client.chat.completions.create.assert_not_called()
        cartesia_client.tts.generate.assert_called_once()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [1]
        assert manifest["versions"][0]["sentence"] == "The word is here."


class TestOnlyScoreZeroKeepsCleanInternalOnly:
    def test_only_score_0_calls_openai_but_deletes_clean_wav(self, tmp_path):
        successful_ids, openai_client, cartesia_client = _run([_item()], tmp_path, only_scores={0})
        assert successful_ids == [1]
        folder = tmp_path / "001_word"
        assert (folder / "word_0.wav").exists()
        # Clean audio was the unavoidable DSP input, but wasn't requested — deleted.
        assert not (folder / "word_1.wav").exists()
        openai_client.chat.completions.create.assert_called_once()
        cartesia_client.tts.generate.assert_called_once()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [0]
