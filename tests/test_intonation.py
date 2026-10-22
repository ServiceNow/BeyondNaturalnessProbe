"""
Tests for shared/intonation.py: run_intonation_pipeline (the intonation dimension).

All three scores share one verbatim ElevenLabs base. When score 3 isn't
requested, that base is still synthesized (Praat needs *something* to
manipulate for scores 1/2) but saved as a scratch `_base.wav`, never a
delivered `_3.wav`.
"""
from __future__ import annotations

import asyncio
import json

from shared.intonation import run_intonation_pipeline
from tests.helpers import mock_elevenlabs_client


def _item(uid=1, text="The weather today is quite pleasant outside."):
    return {"id": uid, "text": text}


def _run(items, output_dir, *, only_scores=None):
    client = mock_elevenlabs_client(duration_s=1.0)
    successful_ids = asyncio.run(run_intonation_pipeline(
        dimension_name="intonation",
        items=items,
        output_dir=output_dir,
        elevenlabs_client=client,
        voice_id="fake-voice",
        only_scores=only_scores,
    ))
    return successful_ids, client


class TestFullRun:
    def test_default_delivers_all_three_scores(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path)
        assert successful_ids == [1]
        folder = tmp_path / "001"
        for score in (1, 2, 3):
            assert (folder / f"001_{score}.wav").exists()
        assert not (folder / "001_base.wav").exists()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {1, 2, 3}
        client.text_to_speech.convert.assert_called_once()


class TestOnlyScoresBaseIsScratchWhenNotDelivered:
    def test_only_score_3_delivers_the_anchor_directly(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={3})
        assert successful_ids == [1]
        folder = tmp_path / "001"
        assert (folder / "001_3.wav").exists()
        assert not (folder / "001_1.wav").exists()
        assert not (folder / "001_2.wav").exists()
        client.text_to_speech.convert.assert_called_once()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [3]

    def test_only_score_2_uses_scratch_base_not_delivered_as_3(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={2})
        assert successful_ids == [1]
        folder = tmp_path / "001"
        assert (folder / "001_2.wav").exists()
        assert not (folder / "001_3.wav").exists()
        assert not (folder / "001_1.wav").exists()
        # The base was still synthesized once (as scratch) — Praat needs an input.
        client.text_to_speech.convert.assert_called_once()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [2]

    def test_only_score_1_uses_scratch_base_not_delivered_as_3(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={1})
        assert successful_ids == [1]
        folder = tmp_path / "001"
        assert (folder / "001_1.wav").exists()
        assert not (folder / "001_3.wav").exists()
        assert not (folder / "001_2.wav").exists()
        client.text_to_speech.convert.assert_called_once()
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [1]

    def test_multiple_requested_scores(self, tmp_path):
        _, client = _run([_item()], tmp_path, only_scores={2, 3})
        folder = tmp_path / "001"
        assert (folder / "001_3.wav").exists()
        assert (folder / "001_2.wav").exists()
        assert not (folder / "001_1.wav").exists()
        client.text_to_speech.convert.assert_called_once()
