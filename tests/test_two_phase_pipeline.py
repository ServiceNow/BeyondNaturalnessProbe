"""
Tests for shared/pipeline.py: run_two_phase_pipeline — the shared runner behind
the lexical_stress, phonetic_accuracy, and prosodic_stress dimensions.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from shared.pipeline import run_two_phase_pipeline
from tests.helpers import mock_cartesia_client, mock_openai_client


BINARY_PAYLOAD = {
    "target_word": "cat",
    "versions": [
        {"score": 0, "sentence": "The <<k|ˈ|æ|t>> sat.", "error_description": "wrong stress"},
        {"score": 1, "sentence": "The <<ˈ|k|æ|t>> sat.", "error_description": "correct stress"},
    ],
}

TERNARY_PAYLOAD = {
    "target_word": "cat",
    "versions": [
        {"score": 1, "sentence": "The sit sat.", "error_description": "vowel substitution"},
        {"score": 2, "sentence": "The gat sat.", "error_description": "consonant error"},
        {"score": 3, "sentence": "The cat sat.", "error_description": "correct"},
    ],
}


def _item(uid=1, text="The cat sat.", target_word="cat"):
    return {"id": uid, "text": text, "target_word": target_word}


def _run(items, payload, output_dir, *, only_scores=None):
    openai_client = mock_openai_client(payload)
    cartesia_client = mock_cartesia_client()
    successful_ids = asyncio.run(run_two_phase_pipeline(
        dimension_name="test dimension",
        items=items,
        prompt_template="{{INPUT_UTTERANCE}} / {{TARGET_WORD}}",
        output_dir=output_dir,
        openai_client=openai_client,
        deployment="gpt-5",
        cartesia_client=cartesia_client,
        model_id="sonic-3",
        voice_id="fake-voice",
        only_scores=only_scores,
    ))
    return successful_ids, openai_client, cartesia_client


class TestFullRunAllScoresDelivered:
    def test_binary_dimension_delivers_both_scores(self, tmp_path):
        successful_ids, openai_client, _ = _run([_item()], BINARY_PAYLOAD, tmp_path)
        assert successful_ids == [1]
        folder = tmp_path / "001_cat"
        assert (folder / "001_cat_0.wav").exists()
        assert (folder / "001_cat_1.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {0, 1}
        assert manifest["utterance_id"] == 1
        assert manifest["target_word"] == "cat"
        openai_client.chat.completions.create.assert_called_once()

    def test_ternary_dimension_delivers_all_three_scores(self, tmp_path):
        successful_ids, _, _ = _run([_item()], TERNARY_PAYLOAD, tmp_path)
        assert successful_ids == [1]
        folder = tmp_path / "001_cat"
        for score in (1, 2, 3):
            assert (folder / f"001_cat_{score}.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {1, 2, 3}


class TestOnlyScoresFiltering:
    @pytest.mark.parametrize("wanted", [0, 1])
    def test_binary_only_scores_delivers_just_that_score(self, tmp_path, wanted):
        successful_ids, _, _ = _run([_item()], BINARY_PAYLOAD, tmp_path, only_scores={wanted})
        assert successful_ids == [1]
        folder = tmp_path / "001_cat"
        assert (folder / f"001_cat_{wanted}.wav").exists()
        other = 1 - wanted
        assert not (folder / f"001_cat_{other}.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [wanted]

    @pytest.mark.parametrize("wanted", [1, 2, 3])
    def test_ternary_only_scores_delivers_just_that_score(self, tmp_path, wanted):
        successful_ids, _, _ = _run([_item()], TERNARY_PAYLOAD, tmp_path, only_scores={wanted})
        assert successful_ids == [1]
        folder = tmp_path / "001_cat"
        assert (folder / f"001_cat_{wanted}.wav").exists()
        for other in {1, 2, 3} - {wanted}:
            assert not (folder / f"001_cat_{other}.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [wanted]

    def test_gpt_called_exactly_once_regardless_of_only_scores(self, tmp_path):
        # Phase 1 can't ask the model for a subset — it always returns everything
        # in one call, and filtering happens afterward on the Phase-2 side.
        _, openai_client, _ = _run([_item()], TERNARY_PAYLOAD, tmp_path, only_scores={1})
        openai_client.chat.completions.create.assert_called_once()

    def test_multiple_requested_scores_delivers_all_of_them(self, tmp_path):
        successful_ids, _, _ = _run([_item()], TERNARY_PAYLOAD, tmp_path, only_scores={1, 3})
        assert successful_ids == [1]
        folder = tmp_path / "001_cat"
        assert (folder / "001_cat_1.wav").exists()
        assert (folder / "001_cat_3.wav").exists()
        assert not (folder / "001_cat_2.wav").exists()
