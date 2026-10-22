"""
Tests for shared/pipeline.py: run_speech_rate_pipeline (the speech_rate dimension).

Score 3 (baseline) is always the real anchor; score 2 is Cartesia-at-deviation
(or a PSOLA copy of score 3 in --psola-only mode); score 1 is always a copy of
score 2 + extra PSOLA — never a separate Cartesia call. These tests check that
the 3->2->1 dependency chain generates only what's structurally necessary and
that --only-scores never leaves an undeleted, unrequested WAV behind.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from shared.pipeline import run_speech_rate_pipeline
from tests.helpers import mock_cartesia_client


def _item(uid=1, text="The quick fox jumps over the lazy dog today."):
    return {"id": uid, "text": text}


def _run(items, output_dir, *, only_scores=None, psola_only=False):
    client = mock_cartesia_client()
    successful_ids = asyncio.run(run_speech_rate_pipeline(
        items=items,
        output_dir=output_dir,
        cartesia_client=client,
        model_id="sonic-3",
        voice_id="fake-voice",
        only_scores=only_scores,
        psola_only=psola_only,
    ))
    return successful_ids, client


class TestFullRun:
    def test_default_delivers_all_three_scores(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path)
        assert successful_ids == [1]
        folder = tmp_path / "001_fast"
        for score in (1, 2, 3):
            assert (folder / f"score_{score}.wav").exists()
        # score 1 is derived from score 2's file, not a separate Cartesia call —
        # only score 3 (baseline) and score 2 (deviation) call the TTS client.
        assert client.tts.generate.call_count == 2

    def test_manifest_has_all_three_versions_with_ratios(self, tmp_path):
        _run([_item()], tmp_path)
        manifest = json.loads((tmp_path / "001_fast" / "manifest.json").read_text())
        scores = {v["score"]: v for v in manifest["versions"]}
        assert set(scores) == {1, 2, 3}
        assert scores[3]["duration_ratio"] == 1.0
        assert scores[1]["psola_applied"] is True
        assert scores[3]["psola_applied"] is False


class TestOnlyScoresSkipsUnneededWork:
    def test_only_score_3_skips_score_2_and_1_entirely(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={3})
        assert successful_ids == [1]
        folder = tmp_path / "001_fast"
        assert (folder / "score_3.wav").exists()
        assert not (folder / "score_2.wav").exists()
        assert not (folder / "score_1.wav").exists()
        # No score 2/1 work at all — exactly one Cartesia call for the baseline.
        assert client.tts.generate.call_count == 1
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [3]

    def test_only_score_2_keeps_baseline_internal_only(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={2})
        assert successful_ids == [1]
        folder = tmp_path / "001_fast"
        assert (folder / "score_2.wav").exists()
        # Score 3 was generated as the ratio baseline, then deleted — not delivered.
        assert not (folder / "score_3.wav").exists()
        assert not (folder / "score_1.wav").exists()
        assert client.tts.generate.call_count == 2
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [2]

    def test_only_score_1_generates_full_chain_but_keeps_only_score_1(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={1})
        assert successful_ids == [1]
        folder = tmp_path / "001_fast"
        assert (folder / "score_1.wav").exists()
        assert not (folder / "score_2.wav").exists()
        assert not (folder / "score_3.wav").exists()
        # Score 1 has no Cartesia call of its own (copy + PSOLA of score 2) —
        # still just 2 calls total (score 3 baseline + score 2 deviation).
        assert client.tts.generate.call_count == 2
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [1]

    def test_multiple_requested_scores_keeps_both_no_extra_calls(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, only_scores={1, 3})
        folder = tmp_path / "001_fast"
        assert (folder / "score_1.wav").exists()
        assert (folder / "score_3.wav").exists()
        assert not (folder / "score_2.wav").exists()
        assert client.tts.generate.call_count == 2


class TestPsolaOnlyMode:
    def test_psola_only_makes_exactly_one_cartesia_call(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, psola_only=True)
        assert successful_ids == [1]
        folder = tmp_path / "001_fast"
        for score in (1, 2, 3):
            assert (folder / f"score_{score}.wav").exists()
        # Only the baseline is a real Cartesia render; 2 and 1 are pure PSOLA.
        assert client.tts.generate.call_count == 1

    def test_psola_only_respects_only_scores(self, tmp_path):
        successful_ids, client = _run([_item()], tmp_path, psola_only=True, only_scores={1})
        folder = tmp_path / "001_fast"
        assert (folder / "score_1.wav").exists()
        assert not (folder / "score_2.wav").exists()
        assert not (folder / "score_3.wav").exists()
        assert client.tts.generate.call_count == 1


class TestDirectionAlternation:
    def test_odd_index_items_are_fast_even_are_slow(self, tmp_path):
        items = [_item(uid=1), _item(uid=2)]
        successful_ids, _ = _run(items, tmp_path)
        assert set(successful_ids) == {1, 2}
        assert (tmp_path / "001_fast").exists()
        assert (tmp_path / "002_slow").exists()
