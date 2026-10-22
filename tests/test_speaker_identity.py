"""
Tests for shared/speaker_identity.py: run_speaker_identity_consistency
(the speaker_identity_consistency dimension) — both crossfade and stitch modes.

The Cartesia SSE boundary (`synthesize_with_timestamps`) is mocked with a
synthetic sine-tone "voice" carrying plausible word/phoneme timestamps, so
the real pitch-screening and crossfade/stitch mixing code runs for real
(Praat can measure a clean F0 off a pure tone) — these tests validate
orchestration (which voices get rendered, which scores get delivered) rather
than crossfade audio quality.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from shared.speaker_identity import (
    PhonemeEntry,
    SISample,
    TimedAudio,
    VoicePair,
    run_speaker_identity_consistency,
)

SAMPLE_RATE = 44100
PAIR = VoicePair("test", "VoiceA", "voice-a-id", "VoiceB", "voice-b-id")


def _fake_timed_audio(freq=150.0, n_words=5, duration_s=3.0, include_phonemes=True):
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE
    pcm = (3000 * np.sin(2 * np.pi * freq * t)).astype(np.int16).tobytes()

    word_dur = duration_s / n_words
    words = [f"word{i}" for i in range(n_words)]
    starts = [i * word_dur for i in range(n_words)]
    ends = [(i + 1) * word_dur for i in range(n_words)]

    phonemes = None
    if include_phonemes:
        # One vowel phoneme per word, matching the module's IPA_VOWELS check.
        phonemes = [
            PhonemeEntry(phoneme="a", start=starts[i] + 0.02, end=ends[i] - 0.02)
            for i in range(n_words)
        ]
    return TimedAudio(pcm=pcm, words=words, starts=starts, ends=ends, phonemes=phonemes)


def _fake_synth_side_effect(client, text, voice_id, *, model_id="sonic-3",
                             sample_rate=SAMPLE_RATE, include_phonemes=False):
    # Same frequency for every voice — keeps pitch screening deterministic
    # (no shift branch triggered), since we're testing orchestration, not
    # the pitch-matching DSP itself.
    return _fake_timed_audio(include_phonemes=include_phonemes)


def _run(samples, output_base, *, only_scores=None):
    fake_synth = AsyncMock(side_effect=_fake_synth_side_effect)
    with patch("shared.speaker_identity.synthesize_with_timestamps", fake_synth):
        consumed = asyncio.run(run_speaker_identity_consistency(
            samples, output_base, MagicMock(), only_scores=only_scores,
        ))
    return consumed, fake_synth


def _crossfade_sample(uid=1, reversed_order=False):
    utt = {"id": uid, "text": "The quick brown fox jumps over the lazy dog today."}
    return SISample(index=0, mode="crossfade", pair=PAIR, reversed_order=reversed_order,
                     utt_primary=utt, utt_partner=None)


def _stitch_sample(uid_a=1, uid_b=2, reversed_order=False):
    utt_a = {"id": uid_a, "text": "The quick brown fox jumps over the lazy dog today."}
    utt_b = {"id": uid_b, "text": "A completely different sentence spoken here today."}
    return SISample(index=1, mode="stitch", pair=PAIR, reversed_order=reversed_order,
                     utt_primary=utt_a, utt_partner=utt_b)


class TestCrossfadeFullRun:
    def test_default_delivers_both_scores_and_renders_both_voices(self, tmp_path):
        consumed, fake_synth = _run([_crossfade_sample()], tmp_path)
        assert consumed == [1]
        folder = tmp_path / "001_test_fwd"
        assert (folder / "0.wav").exists()
        assert (folder / "1.wav").exists()
        assert fake_synth.call_count == 2  # voice A + voice B
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {0, 1}
        assert manifest["mode"] == "crossfade"


class TestCrossfadeOnlyScores:
    def test_only_score_1_skips_voice_b_entirely(self, tmp_path):
        consumed, fake_synth = _run([_crossfade_sample()], tmp_path, only_scores={1})
        assert consumed == [1]
        folder = tmp_path / "001_test_fwd"
        assert (folder / "1.wav").exists()
        assert not (folder / "0.wav").exists()
        # Only voice A was rendered — no crossfade computation needed at all.
        assert fake_synth.call_count == 1
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [1]

    def test_only_score_0_still_needs_both_voices(self, tmp_path):
        consumed, fake_synth = _run([_crossfade_sample()], tmp_path, only_scores={0})
        assert consumed == [1]
        folder = tmp_path / "001_test_fwd"
        assert (folder / "0.wav").exists()
        assert not (folder / "1.wav").exists()
        assert fake_synth.call_count == 2
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [0]


class TestStitchFullRun:
    def test_default_delivers_both_scores_and_renders_three_times(self, tmp_path):
        consumed, fake_synth = _run([_stitch_sample()], tmp_path)
        assert set(consumed) == {1, 2}
        folder = tmp_path / "001_002_test_fwd"
        assert (folder / "0.wav").exists()
        assert (folder / "1.wav").exists()
        # voice_a@U1, voice_a@U2, voice_b@U2
        assert fake_synth.call_count == 3
        manifest = json.loads((folder / "manifest.json").read_text())
        assert {v["score"] for v in manifest["versions"]} == {0, 1}
        assert manifest["mode"] == "stitch"


class TestStitchOnlyScores:
    def test_only_score_1_skips_voice_b_render(self, tmp_path):
        consumed, fake_synth = _run([_stitch_sample()], tmp_path, only_scores={1})
        assert set(consumed) == {1, 2}
        folder = tmp_path / "001_002_test_fwd"
        assert (folder / "1.wav").exists()
        assert not (folder / "0.wav").exists()
        # voice_a@U1, voice_a@U2 — voice B (used only for score 0) is never rendered.
        assert fake_synth.call_count == 2
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [1]

    def test_only_score_0_skips_redundant_voice_a_at_u2(self, tmp_path):
        consumed, fake_synth = _run([_stitch_sample()], tmp_path, only_scores={0})
        assert set(consumed) == {1, 2}
        folder = tmp_path / "001_002_test_fwd"
        assert (folder / "0.wav").exists()
        assert not (folder / "1.wav").exists()
        # voice_a@U1, voice_b@U2 — voice_a@U2 (used only for score 1) is skipped.
        assert fake_synth.call_count == 2
        manifest = json.loads((folder / "manifest.json").read_text())
        assert [v["score"] for v in manifest["versions"]] == [0]


class TestMixedSamplesInOneRun:
    def test_crossfade_and_stitch_together(self, tmp_path):
        samples = [_crossfade_sample(uid=10), _stitch_sample(uid_a=20, uid_b=21)]
        consumed, _ = _run(samples, tmp_path)
        assert set(consumed) == {10, 20, 21}
        assert (tmp_path / "010_test_fwd" / "0.wav").exists()
        assert (tmp_path / "020_021_test_fwd" / "0.wav").exists()
