"""
Tests for pipeline.py's own dispatch/validation logic — the parts not
delegated to shared/*.py: env-requirement computation, --only-scores parsing
and validation, the run_dimension skip/error paths, and the expressiveness /
prosodic_boundary item-selection helpers.
"""
from __future__ import annotations

import asyncio
import json
import types
from unittest.mock import patch

import pytest

import pipeline
from tests.helpers import mock_cartesia_client, mock_openai_client


def _args(**overrides):
    defaults = dict(
        dimension="lexical_stress",
        strategy=None,
        output=None,
        voice=None,
        num=None,
        model="sonic-3",
        deployment="gpt-5",
        openai_concurrency=5,
        tts_concurrency=10,
        prompts_dir=pipeline.ROOT / "prompts",
        counts="50,20,30",
        only_scores=None,
        utterances=None,
        dataset="mozilla",
        category=None,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestEnvRequirements:
    @pytest.mark.parametrize("key,expected", [
        ("lexical_stress", (True, True)),
        ("phonetic_accuracy", (True, True)),
        ("prosodic_stress", (True, True)),
        ("human_plausibility", (True, True)),
        ("speech_rate", (False, True)),
        ("speaker_identity_consistency", (False, True)),
        ("intonation", (False, False)),
        ("expressiveness", (False, True)),
    ])
    def test_per_dimension(self, key, expected):
        assert pipeline._env_requirements(key, _args()) == expected

    def test_prosodic_boundary_default_preset_needs_no_openai(self):
        assert pipeline._env_requirements("prosodic_boundary", _args(strategy=None)) == (False, True)

    def test_prosodic_boundary_extra_punctuation_needs_openai(self):
        assert pipeline._env_requirements(
            "prosodic_boundary", _args(strategy="extra_punctuation"),
        ) == (True, True)

    def test_all_needs_everything(self):
        assert pipeline._env_requirements("all", _args(dimension="all")) == (True, True)


class TestValidScoresFor:
    @pytest.mark.parametrize("key,expected", [
        ("lexical_stress", {0, 1}),
        ("phonetic_accuracy", {1, 2, 3}),
        ("prosodic_stress", {0, 1}),
        ("speech_rate", {1, 2, 3}),
        ("speaker_identity_consistency", {0, 1}),
        ("intonation", {1, 2, 3}),
        ("human_plausibility", {0, 1}),
        ("expressiveness", {1, 2, 3}),
    ])
    def test_static_dimensions(self, key, expected):
        assert set(pipeline.valid_scores_for(key, _args())) == expected

    def test_prosodic_boundary_preset_is_ternary(self):
        assert set(pipeline.valid_scores_for("prosodic_boundary", _args(strategy="preset"))) == {1, 2, 3}

    def test_prosodic_boundary_default_is_ternary(self):
        assert set(pipeline.valid_scores_for("prosodic_boundary", _args(strategy=None))) == {1, 2, 3}

    @pytest.mark.parametrize("strategy", ["omit_spaces", "omit_punctuation", "extra_punctuation"])
    def test_prosodic_boundary_other_strategies_are_binary(self, strategy):
        assert set(pipeline.valid_scores_for("prosodic_boundary", _args(strategy=strategy))) == {0, 1}


class TestParseOnlyScores:
    def test_none_returns_none(self):
        assert pipeline._parse_only_scores(None) is None

    def test_empty_string_returns_none(self):
        assert pipeline._parse_only_scores("") is None

    def test_single_value(self):
        assert pipeline._parse_only_scores("2") == {2}

    def test_multiple_values(self):
        assert pipeline._parse_only_scores("1,2,3") == {1, 2, 3}

    def test_tolerates_whitespace(self):
        assert pipeline._parse_only_scores(" 1 , 2 ") == {1, 2}

    def test_invalid_value_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            pipeline._parse_only_scores("abc")
        assert "--only-scores" in str(exc_info.value)


class TestRunDimensionValidation:
    def test_single_dimension_invalid_score_exits(self, tmp_path):
        args = _args(dimension="lexical_stress", only_scores="5", output=tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(pipeline.run_dimension(
                "lexical_stress", pipeline.DIMENSIONS["lexical_stress"], args, [], {},
            ))
        msg = str(exc_info.value)
        assert "lexical_stress" in msg
        assert "[0, 1]" in msg

    def test_single_dimension_partially_invalid_score_exits(self, tmp_path):
        # {0, 5}: 0 is valid for lexical_stress but 5 isn't — still a hard error
        # for an explicit single-dimension run (no silent partial-narrowing).
        args = _args(dimension="lexical_stress", only_scores="0,5", output=tmp_path)
        with pytest.raises(SystemExit):
            asyncio.run(pipeline.run_dimension(
                "lexical_stress", pipeline.DIMENSIONS["lexical_stress"], args, [], {},
            ))

    def test_dimension_all_no_overlap_skips_without_dispatch(self, tmp_path, capsys):
        # speaker_identity_consistency's scale is {0,1} — requesting only score 2
        # under --dimension all should skip cleanly, not error, and not touch
        # any client (no mocking here — it must return before dispatch).
        #
        # An empty utterances list alone would also produce (cache_key, []) via
        # _run_speaker_identity_dimension's own "no utterances available" path —
        # that would make this test pass even if the only_scores skip guard were
        # broken. So we assert on the actual skip message instead of just the
        # return value, to prove *this* code path is what produced it.
        args = _args(dimension="all", only_scores="2", output=tmp_path)
        cache_key, successful_ids = asyncio.run(pipeline.run_dimension(
            "speaker_identity_consistency",
            pipeline.DIMENSIONS["speaker_identity_consistency"],
            args, [], {},
        ))
        assert successful_ids == []
        assert "speaker_identity_consistency" in cache_key
        out = capsys.readouterr().out
        assert "Skipping — none of the requested --only-scores apply" in out
        assert "No unused utterances available" not in out


class TestSelectExpressivenessItems:
    def _utterances(self, n=20):
        return [{"id": i, "text": f"Sentence number {i}."} for i in range(1, n + 1)]

    def test_splits_by_counts(self):
        items = pipeline.select_expressiveness_items(self._utterances(), "2,1,3", set())
        counts = {1: 0, 2: 0, 3: 0}
        for it in items:
            counts[it["score"]] += 1
        assert counts == {1: 2, 2: 1, 3: 3}
        assert len(items) == 6

    def test_excludes_ids_already_used(self):
        used = {1, 2, 3, 4, 5}
        items = pipeline.select_expressiveness_items(self._utterances(), "2,0,0", used)
        assert all(it["id"] not in used for it in items)

    def test_zero_counts_for_two_scores_yields_single_score_items(self):
        items = pipeline.select_expressiveness_items(self._utterances(), "0,0,5", set())
        assert len(items) == 5
        assert all(it["score"] == 3 for it in items)

    def test_invalid_counts_string_exits(self):
        with pytest.raises(SystemExit):
            pipeline.select_expressiveness_items(self._utterances(), "not,valid", set())


class TestRunExpressivenessDimension:
    def _utterances_file(self, tmp_path, n=20):
        data = {"utterances": [{"id": i, "text": f"Sentence number {i}."} for i in range(1, n + 1)]}
        p = tmp_path / "harvard_sentences.json"
        p.write_text(json.dumps(data))
        return p

    def test_only_scores_zeroes_excluded_counts(self, tmp_path, monkeypatch):
        harvard_path = self._utterances_file(tmp_path)
        monkeypatch.setattr(pipeline, "ROOT", tmp_path)
        # ROOT / "data" / "harvard_sentences.json" must resolve to our fixture.
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "harvard_sentences.json").write_text(harvard_path.read_text())

        args = _args(output=tmp_path / "results", counts="5,3,2")
        with patch("pipeline.make_cartesia_client", return_value=mock_cartesia_client()):
            cache_key, successful_ids = asyncio.run(pipeline.run_expressiveness_dimension(
                pipeline.DIMENSIONS["expressiveness"], args, {}, only_scores={2},
            ))
        assert len(successful_ids) == 3  # only the "under" (score 2) count survives
        output_dir = args.output / "expressiveness"
        wavs = list(output_dir.glob("2_*/2_*.wav"))
        assert len(wavs) == 3
        assert not list(output_dir.glob("1_*")) and not list(output_dir.glob("3_*"))


class TestRunProsodicBoundaryDimension:
    def test_preset_only_scores_filters_to_matching_items(self, tmp_path):
        args = _args(dimension="prosodic_boundary", strategy="preset", output=tmp_path)
        with patch("pipeline.make_cartesia_client", return_value=mock_cartesia_client()):
            cache_key, successful_ids = asyncio.run(pipeline.run_prosodic_boundary_dimension(
                pipeline.DIMENSIONS["prosodic_boundary"], args, {}, only_scores={2},
            ))
        # The curated preset has exactly 20 utterances per score.
        assert len(successful_ids) == 20
        output_dir = args.output / "prosodic_boundary"
        all_wavs = list(output_dir.glob("*/*.wav"))
        assert len(all_wavs) == 20
        assert all(w.name.endswith("_2.wav") for w in all_wavs)

    def test_omit_punctuation_only_score_0_skips_canonical_pairing(self, tmp_path):
        args = _args(dimension="prosodic_boundary", strategy="omit_punctuation", output=tmp_path)
        with patch("pipeline.make_cartesia_client", return_value=mock_cartesia_client()):
            asyncio.run(pipeline.run_prosodic_boundary_dimension(
                pipeline.DIMENSIONS["prosodic_boundary"], args, {}, only_scores={0},
            ))
        output_dir = args.output / "prosodic_boundary"
        all_wavs = list(output_dir.glob("*/*.wav"))
        # One WAV per item (score 0 only) — the canonical (score 1) pairing
        # that omit_punctuation always attaches is filtered out before synthesis.
        assert all(w.name.endswith("_0.wav") for w in all_wavs)
        assert len(all_wavs) == len(list(output_dir.glob("*/manifest.json")))
