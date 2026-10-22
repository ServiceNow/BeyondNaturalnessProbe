"""
Helpers for the speaker-identity-consistency dimension.

Each sample produces a paired (score 0, score 1) output — a deliberately
inconsistent utterance alongside a consistent reference rendered by a
single voice. Two inconsistency modes:

  - crossfade — render one sentence with both voices of a pair, then
    equal-power fade voice A → voice B across the primary vowel of a
    middle-of-sentence target word. Phoneme-aligned so consonant stops
    play from one voice only. Voice B is constant-pitch-shifted (PSOLA
    via Parselmouth) when the F0 mismatch at the target vowel exceeds
    ~0.5 semitones, and candidate words with >3-semitone mismatch are
    rejected in favor of the next best one.
  - stitch — concatenate two different utterances, one spoken by each
    voice, with a 400 ms silence gap.

Audio is handled as raw 16-bit mono PCM bytes. `synthesize_with_timestamps`
pulls audio + Cartesia word timestamps (and, for crossfade, phoneme
timestamps) together via the SSE endpoint.

`run_speaker_identity_consistency` is the top-level entry point used by
pipeline.py; `build_si_samples` weaves utterances into a balanced
mode × gender × voice-order schedule.
"""
from __future__ import annotations

import asyncio
import base64
import json
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cartesia import AsyncCartesia


SAMPLE_RATE = 44100  # Hz — matches rest of pipeline

# Rough IPA vowel set — used to locate the sustained nucleus inside a word.
# Base chars only; length marks (ː, ˑ) are stripped before membership check.
IPA_VOWELS = set("aɑɒæeɛɜəɘɤɪɨiɔoøœɶuʊʉʌyɐ")


def _is_vowel(phoneme: str) -> bool:
    base = phoneme.replace("ː", "").replace("ˑ", "")
    return any(ch in IPA_VOWELS for ch in base.lower())


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PhonemeEntry:
    phoneme: str
    start: float
    end: float


@dataclass
class TimedAudio:
    pcm: bytes                              # raw s16le mono
    words: list[str]
    starts: list[float]                     # seconds, per word
    ends: list[float]
    phonemes: list[PhonemeEntry] | None = None   # flat list across the whole utterance


def _phonemes_in_word(ta: TimedAudio, word_index: int, *, tol: float = 0.002) -> list[PhonemeEntry]:
    """Return phoneme entries whose midpoint falls inside word `word_index`.

    Cartesia's phoneme_timestamps events do NOT emit one event per word — short
    neighbouring words can get batched into a single event. So we cannot index
    phoneme groups by word index. Instead, we align by time.
    """
    if ta.phonemes is None:
        return []
    ws, we = ta.starts[word_index], ta.ends[word_index]
    out = []
    for p in ta.phonemes:
        mid = 0.5 * (p.start + p.end)
        if ws - tol <= mid <= we + tol:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

async def synthesize_with_timestamps(
    client: AsyncCartesia,
    text: str,
    voice_id: str,
    *,
    model_id: str = "sonic-3",
    sample_rate: int = SAMPLE_RATE,
    include_phonemes: bool = False,
) -> TimedAudio:
    """Render `text` with `voice_id` via SSE. Returns PCM + word timestamps;
    optionally also collects per-word IPA phoneme timestamps."""
    stream = await client.tts.generate_sse(
        model_id=model_id,
        transcript=text,
        voice={"mode": "id", "id": voice_id},
        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": sample_rate},
        add_timestamps=True,
        add_phoneme_timestamps=include_phonemes,
    )
    pcm = bytearray()
    words: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    phonemes: list[PhonemeEntry] = []
    async for ev in stream:
        if ev.type == "chunk":
            pcm.extend(base64.b64decode(ev.data))
        elif ev.type == "timestamps":
            ts = ev.word_timestamps
            words.extend(ts.words)
            starts.extend(ts.start)
            ends.extend(ts.end)
        elif ev.type == "phoneme_timestamps":
            pt = ev.phoneme_timestamps
            for ph, s, e in zip(pt.phonemes, pt.start, pt.end):
                phonemes.append(PhonemeEntry(phoneme=ph, start=s, end=e))
    return TimedAudio(
        pcm=bytes(pcm),
        words=words,
        starts=starts,
        ends=ends,
        phonemes=phonemes if include_phonemes else None,
    )


# ---------------------------------------------------------------------------
# Target word selection
# ---------------------------------------------------------------------------

def _rank_candidate_words(
    a: TimedAudio,
    b: TimedAudio,
    *,
    middle_band: tuple[float, float] = (0.25, 0.75),
) -> list[int]:
    """Return candidate word indices ranked best-first by the duration + center heuristic."""
    n = min(len(a.words), len(b.words))
    if n == 0:
        return []

    lo = max(0, int(n * middle_band[0]))
    hi = min(n - 1, int(n * middle_band[1]))
    if lo > hi:
        lo, hi = 0, n - 1

    def has_vowel(t: TimedAudio, i: int) -> bool:
        if t.phonemes is None:
            return True
        return any(_is_vowel(p.phoneme) for p in _phonemes_in_word(t, i))

    candidates = [i for i in range(lo, hi + 1) if has_vowel(a, i) and has_vowel(b, i)]
    if not candidates:
        candidates = list(range(lo, hi + 1))

    def avg_dur(i: int) -> float:
        return 0.5 * ((a.ends[i] - a.starts[i]) + (b.ends[i] - b.starts[i]))

    mid = (n - 1) / 2.0
    return sorted(candidates, key=lambda i: (avg_dur(i), -abs(i - mid)), reverse=True)


def select_target_word(
    a: TimedAudio,
    b: TimedAudio,
    *,
    middle_band: tuple[float, float] = (0.25, 0.75),
) -> int:
    """Pick the best target word purely by duration + centrality (no pitch check)."""
    ranked = _rank_candidate_words(a, b, middle_band=middle_band)
    if not ranked:
        raise ValueError("no candidate words available")
    return ranked[0]


@dataclass
class PitchScreenedPick:
    word_index: int
    f0_a: float | None
    f0_b: float | None
    shift_ratio: float      # multiplier to apply to voice B's pcm (1.0 = no shift)
    rejected_indices: list[int]   # words we skipped because of pitch mismatch


def _semitone_diff(f0_a: float, f0_b: float) -> float:
    """Absolute difference between two F0 values in semitones."""
    import math
    return abs(12.0 * math.log2(f0_a / f0_b))


def select_target_word_pitch_screened(
    a: TimedAudio,
    b: TimedAudio,
    *,
    middle_band: tuple[float, float] = (0.25, 0.75),
    skip_above_semitones: float = 3.0,      # reject if mismatch exceeds this
    shift_above_semitones: float = 0.5,     # below this: don't bother shifting
    sample_rate: int = SAMPLE_RATE,
) -> PitchScreenedPick:
    """Rank middle-band words best-first; accept the top one whose vowel F0
    mismatch between A and B is under `skip_above_semitones`. If the residual
    mismatch exceeds `shift_above_semitones`, return `shift_ratio = F0_A / F0_B`
    so voice B's PCM can be pitch-matched.

    Semitones (not raw Hz) because the same absolute Hz gap is perceptually
    larger on a low-pitched voice than a high one.

    If every candidate exceeds the skip threshold, falls back to whichever
    candidate had the smallest mismatch, applying a shift.
    """
    ranked = _rank_candidate_words(a, b, middle_band=middle_band)
    if not ranked:
        raise ValueError("no candidate words available")

    rejected: list[int] = []
    best_fallback: PitchScreenedPick | None = None
    best_fallback_diff = float("inf")

    for idx in ranked:
        try:
            a_v = _primary_vowel_span(_phonemes_in_word(a, idx))
            b_v = _primary_vowel_span(_phonemes_in_word(b, idx))
        except ValueError:
            rejected.append(idx)
            continue
        f0_a = measure_vowel_f0(a.pcm, a_v, sample_rate=sample_rate)
        f0_b = measure_vowel_f0(b.pcm, b_v, sample_rate=sample_rate)
        if f0_a is None or f0_b is None:
            rejected.append(idx)
            continue

        diff_st = _semitone_diff(f0_a, f0_b)
        if diff_st > skip_above_semitones:
            rejected.append(idx)
            if diff_st < best_fallback_diff:
                best_fallback = PitchScreenedPick(
                    word_index=idx, f0_a=f0_a, f0_b=f0_b,
                    shift_ratio=f0_a / f0_b, rejected_indices=[],
                )
                best_fallback_diff = diff_st
            continue

        shift_ratio = 1.0 if diff_st <= shift_above_semitones else f0_a / f0_b
        return PitchScreenedPick(
            word_index=idx, f0_a=f0_a, f0_b=f0_b,
            shift_ratio=shift_ratio, rejected_indices=list(rejected),
        )

    if best_fallback is not None:
        best_fallback.rejected_indices = rejected
        return best_fallback

    return PitchScreenedPick(
        word_index=ranked[0], f0_a=None, f0_b=None,
        shift_ratio=1.0, rejected_indices=rejected,
    )


# ---------------------------------------------------------------------------
# Crossfade (within-word, vowel-only, equal-power)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pitch (F0) measurement + constant pitch shift, via Parselmouth (Praat)
# ---------------------------------------------------------------------------

def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0


def _float_to_pcm(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def measure_vowel_f0(
    pcm: bytes,
    span: tuple[float, float],
    *,
    sample_rate: int = SAMPLE_RATE,
    pad_s: float = 0.05,
) -> float | None:
    """Return median F0 (Hz) across voiced frames inside `span`, or None if none.

    `pad_s` extends the analysis window so Praat has enough context around the
    vowel to find pitch periods; we still only score voiced frames *inside*
    the vowel span.
    """
    import parselmouth
    samples = _pcm_to_float(pcm)
    snd = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    lo = max(0.0, span[0] - pad_s)
    hi = min(snd.duration, span[1] + pad_s)
    if hi - lo <= 0.02:
        return None
    clip = snd.extract_part(from_time=lo, to_time=hi, preserve_times=True)
    pitch = clip.to_pitch(time_step=0.005, pitch_floor=75, pitch_ceiling=600)
    times = pitch.xs()
    f0 = pitch.selected_array["frequency"]
    in_span = [f for t, f in zip(times, f0) if f > 0 and span[0] <= t <= span[1]]
    if not in_span:
        in_span = [f for f in f0 if f > 0]
    return float(np.median(in_span)) if in_span else None


def pitch_shift_pcm(
    pcm: bytes,
    ratio: float,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Constant pitch shift (time-preserving) via Praat's PitchTier manipulation.

    `ratio` > 1 raises pitch, < 1 lowers it. Timbre is preserved as well as
    Praat's PSOLA-based resynthesis allows. Ratio close to 1 is ~no-op.
    """
    import parselmouth
    from parselmouth.praat import call
    if abs(ratio - 1.0) < 0.002:
        return pcm
    samples = _pcm_to_float(pcm)
    snd = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    manip = call(snd, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manip, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", 0.0, snd.duration, float(ratio))
    call([manip, pitch_tier], "Replace pitch tier")
    shifted = call(manip, "Get resynthesis (overlap-add)")
    return _float_to_pcm(np.asarray(shifted.values[0]))


def _primary_vowel_span(entries: list[PhonemeEntry]) -> tuple[float, float]:
    """Pick the 'main' vowel to fade across.

    Cartesia emits roughly uniform per-phoneme durations, so purely
    time-based picks degenerate. We rank by phonological priority so both
    voices in a pair independently pick the same vowel:

      1. Prefer non-schwa (avoid the reduced vowel `ə`)
      2. Prefer length-marked vowels (`ː` = long/stressed nucleus)
      3. Prefer vowel whose midpoint is closest to the word midpoint
    """
    vowels = [p for p in entries if _is_vowel(p.phoneme)]
    if not vowels:
        raise ValueError(f"no vowel found in phonemes: {[p.phoneme for p in entries]}")
    word_start = min(p.start for p in entries)
    word_end   = max(p.end   for p in entries)
    word_mid   = 0.5 * (word_start + word_end)

    def priority(p: PhonemeEntry) -> tuple:
        base = p.phoneme.replace("ː", "").replace("ˑ", "").lower()
        is_schwa   = base == "ə"
        has_length = "ː" in p.phoneme
        mid        = 0.5 * (p.start + p.end)
        return (is_schwa, not has_length, abs(mid - word_mid))

    best = min(vowels, key=priority)
    return (best.start, best.end)


def crossfade_within_vowel(
    a: TimedAudio,
    b: TimedAudio,
    word_index: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    xfade_ms: int | None = None,
) -> bytes:
    """Fade voice A → voice B across the longest vowel phoneme of `word_index`.

    Output layout:
      [0, A.vowel_start)     — voice A at full gain (words before target,
                               plus onset consonants of the target word)
      [fade region]          — equal-power (cos/sin) crossfade:
                               A's vowel fades out, B's vowel fades in
      [B.vowel_start + n, …) — voice B at full gain (coda consonants of
                               target word, then all following words)

    `xfade_ms=None` (default) spans the shorter of the two vowel durations.
    Pass a smaller value to cap the fade window.
    """
    if a.phonemes is None or b.phonemes is None:
        raise ValueError(
            "crossfade_within_vowel requires phoneme timestamps; call "
            "synthesize_with_timestamps(..., include_phonemes=True)"
        )
    if len(a.words) != len(b.words):
        raise ValueError(
            f"word count mismatch: A has {len(a.words)}, B has {len(b.words)}"
        )
    if not (0 <= word_index < len(a.words)):
        raise ValueError(f"word_index {word_index} out of range [0, {len(a.words) - 1}]")

    a_v = _primary_vowel_span(_phonemes_in_word(a, word_index))
    b_v = _primary_vowel_span(_phonemes_in_word(b, word_index))
    span_s = min(a_v[1] - a_v[0], b_v[1] - b_v[0])
    if xfade_ms is not None:
        span_s = min(span_s, xfade_ms / 1000.0)

    arr_a = np.frombuffer(a.pcm, dtype=np.int16)
    arr_b = np.frombuffer(b.pcm, dtype=np.int16)

    a_start = int(round(a_v[0] * sample_rate))
    b_start = int(round(b_v[0] * sample_rate))
    xfade_n = int(round(span_s * sample_rate))
    xfade_n = max(0, min(xfade_n, len(arr_a) - a_start, len(arr_b) - b_start))
    if xfade_n <= 0:
        return np.concatenate([arr_a[:a_start], arr_b[b_start:]]).tobytes()

    # Equal-power (cos/sin) gain curves — sum of squares stays at 1.
    t = np.linspace(0.0, 1.0, xfade_n, dtype=np.float32)
    fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)
    fade_in  = np.sin(0.5 * np.pi * t).astype(np.float32)

    a_head = arr_a[:a_start]
    a_xf_f = arr_a[a_start : a_start + xfade_n].astype(np.float32) * fade_out
    b_xf_f = arr_b[b_start : b_start + xfade_n].astype(np.float32) * fade_in
    b_tail = arr_b[b_start + xfade_n:]

    mixed = np.clip(a_xf_f + b_xf_f, -32768.0, 32767.0).astype(np.int16)
    return np.concatenate([a_head, mixed, b_tail]).tobytes()


# ---------------------------------------------------------------------------
# Stitch & WAV I/O
# ---------------------------------------------------------------------------

def stitch(
    a: TimedAudio,
    b: TimedAudio,
    *,
    sample_rate: int = SAMPLE_RATE,
    gap_ms: int = 400,
) -> bytes:
    """Concatenate A then B with a silence gap in between."""
    gap_n = int(round(gap_ms * sample_rate / 1000))
    silence = np.zeros(gap_n, dtype=np.int16)
    arr_a = np.frombuffer(a.pcm, dtype=np.int16)
    arr_b = np.frombuffer(b.pcm, dtype=np.int16)
    return np.concatenate([arr_a, silence, arr_b]).tobytes()


def write_wav(path: Path, pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


# ---------------------------------------------------------------------------
# Voice pairs (locked in after A/B audition — see MEMORY.md)
# ---------------------------------------------------------------------------

@dataclass
class VoicePair:
    name: str                  # "feminine" | "masculine"
    voice_a_name: str
    voice_a_id: str
    voice_b_name: str
    voice_b_id: str


VOICE_PAIRS: list[VoicePair] = [
    VoicePair("feminine",
              "Brooke", "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
              "Emma",   "f6ff7c0c-e396-40a9-a70b-f7607edb6937"),
    VoicePair("masculine",
              "Ronald", "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
              "Henry",  "87286a8d-7ea7-4235-a41a-dd9fa6630feb"),
]


def pair_by_name(name: str) -> VoicePair:
    for p in VOICE_PAIRS:
        if p.name == name:
            return p
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Utterance selection
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip(".,;:!?\"'-")])


def select_crossfade_utterances(
    utterances: list[dict],
    n: int,
    exclude_ids: set[int],
    *,
    min_words: int = 8,
    max_words: int = 18,
) -> list[dict]:
    """Pick `n` utterances with a word count suitable for a mid-sentence fade."""
    pool = [
        u for u in utterances
        if u["id"] not in exclude_ids
        and min_words <= _word_count(u["text"]) <= max_words
    ]
    random.shuffle(pool)
    return pool[:n]


def select_stitch_pairs(
    utterances: list[dict],
    n: int,
    exclude_ids: set[int],
    *,
    min_words: int = 5,
    max_words: int = 14,
) -> list[tuple[dict, dict]]:
    """Pick `n` disjoint pairs of utterances for stitch dimension."""
    pool = [
        u for u in utterances
        if u["id"] not in exclude_ids
        and min_words <= _word_count(u["text"]) <= max_words
    ]
    random.shuffle(pool)
    pairs: list[tuple[dict, dict]] = []
    it = iter(pool)
    for u in it:
        try:
            partner = next(it)
        except StopIteration:
            break
        pairs.append((u, partner))
        if len(pairs) >= n:
            break
    return pairs


# ---------------------------------------------------------------------------
# Paired-sample renderer (score 0 = inconsistent, score 1 = consistent)
# ---------------------------------------------------------------------------

@dataclass
class SISample:
    """One speaker-identity-consistency sample = one (score 0, score 1) pair.

    A full dimension run produces N of these, with `mode`, `pair`, and
    `reversed_order` rotating on a 2×2×2 schedule so 8 consecutive samples
    visit every combination once. `voice_a`/`voice_b` below reflect the
    already-applied ordering: voice_a is always the 'first / primary'
    voice in this sample.
    """
    index: int                     # 0-based
    mode: str                      # "crossfade" | "stitch"
    pair: VoicePair
    reversed_order: bool
    utt_primary: dict              # crossfade: the sentence; stitch: first utt
    utt_partner: dict | None       # stitch only

    @property
    def voice_a(self) -> tuple[str, str]:
        return ((self.pair.voice_b_name, self.pair.voice_b_id) if self.reversed_order
                else (self.pair.voice_a_name, self.pair.voice_a_id))

    @property
    def voice_b(self) -> tuple[str, str]:
        return ((self.pair.voice_a_name, self.pair.voice_a_id) if self.reversed_order
                else (self.pair.voice_b_name, self.pair.voice_b_id))


def build_si_samples(
    crossfade_items: list[dict],
    stitch_pairs: list[tuple[dict, dict]],
    n: int,
) -> list[SISample]:
    """Weave utterances into N samples, alternating (mode × gender × order).

    Alternation schedule (2×2×2, full coverage every 8 samples):
      - mode   flips every 1  sample (crossfade, stitch, crossfade, …)
      - gender flips every 2  samples (F, F, M, M, F, F, M, M, …)
      - order  flips every 4  samples (fwd, fwd, fwd, fwd, rev, rev, …)
    """
    samples: list[SISample] = []
    cf_iter = iter(crossfade_items)
    st_iter = iter(stitch_pairs)
    for i in range(n):
        mode = "crossfade" if i % 2 == 0 else "stitch"
        pair = VOICE_PAIRS[(i // 2) % 2]           # 0=F, 1=M
        reversed_order = bool((i // 4) % 2)

        if mode == "crossfade":
            try:
                utt = next(cf_iter)
            except StopIteration:
                break
            samples.append(SISample(i, mode, pair, reversed_order, utt, None))
        else:
            try:
                a, b = next(st_iter)
            except StopIteration:
                break
            # Reuse reversed_order to also reverse the utterance order for stitch,
            # so the 'primary' voice always speaks utt_primary (= utt_a here).
            samples.append(SISample(i, mode, pair, reversed_order, a, b))
    return samples


async def _render_si_sample(
    client: AsyncCartesia,
    sample: SISample,
    output_base: Path,
    *,
    model_id: str,
    gap_ms: int,
    total: int,
    semaphore: asyncio.Semaphore,
    only_scores: set[int] | None = None,
) -> tuple[list[int], bool]:
    """Render score-0 (inconsistent) and/or score-1 (consistent) for one sample.

    ``only_scores`` restricts which score(s) are produced. When only score 1
    is wanted, voice B and the crossfade/stitch computation are skipped
    entirely — score 1 only ever needs voice A. When only score 0 is wanted,
    the redundant "voice A alone" render used solely for score 1 is skipped.
    Returns (utterance_ids_consumed, success).
    """
    want = only_scores if only_scores is not None else {0, 1}

    async with semaphore:
        t0 = time.time()
        va_name, va_id = sample.voice_a
        vb_name, vb_id = sample.voice_b
        order_tag = "rev" if sample.reversed_order else "fwd"

        if sample.mode == "crossfade":
            uid = sample.utt_primary["id"]
            text = sample.utt_primary["text"]
            folder = output_base / f"{uid:03d}_{sample.pair.name}_{order_tag}"
            uids = [uid]
        else:
            uid_a = sample.utt_primary["id"]
            uid_b = sample.utt_partner["id"]   # type: ignore[union-attr]
            text_a = sample.utt_primary["text"]
            text_b = sample.utt_partner["text"]  # type: ignore[union-attr]
            folder = output_base / f"{uid_a:03d}_{uid_b:03d}_{sample.pair.name}_{order_tag}"
            uids = [uid_a, uid_b]

        folder.mkdir(parents=True, exist_ok=True)
        banner = (f"[{sample.index + 1}/{total}] {sample.mode}/{sample.pair.name}/"
                  f"{order_tag} — {va_name}→{vb_name}")
        lines = [banner]

        try:
            if sample.mode == "crossfade":
                if want == {1}:
                    # Score 1 (consistent) only needs voice A — no crossfade at all.
                    a_timed = await synthesize_with_timestamps(
                        client, text, va_id, model_id=model_id)
                    write_wav(folder / "1.wav", a_timed.pcm)
                    versions = [{
                        "score": 1,
                        "pair": sample.pair.name,
                        "voice_order": order_tag,
                        "voice_a_name": va_name,
                        "voice_b_name": None,
                        "error_description": f"Consistent — {va_name} only",
                        "transcription": "",
                        "sentence": text,
                        "audio_file": "1.wav",
                    }]
                    manifest = {
                        "mode": "crossfade",
                        "pair": sample.pair.name,
                        "voice_order": order_tag,
                        "utterance_id": uid,
                        "original_text": text,
                        "target_word": "",
                        "versions": versions,
                    }
                    lines.append(f"  consistent-only -> 1.wav  ({time.time() - t0:.1f}s)")
                else:
                    # Score 0 needs both voices + the crossfade; score 1 (if wanted) is
                    # then a free byproduct (voice A's render, already computed).
                    a_timed, b_timed = await asyncio.gather(
                        synthesize_with_timestamps(client, text, va_id, model_id=model_id,
                                                   include_phonemes=True),
                        synthesize_with_timestamps(client, text, vb_id, model_id=model_id,
                                                   include_phonemes=True),
                    )
                    pick = select_target_word_pitch_screened(a_timed, b_timed)
                    split = pick.word_index
                    split_word = a_timed.words[split]

                    # Apply constant pitch-shift to B if a residual mismatch remains
                    if abs(pick.shift_ratio - 1.0) > 0.002:
                        shifted_b_pcm = pitch_shift_pcm(b_timed.pcm, pick.shift_ratio)
                        b_for_fade = TimedAudio(
                            pcm=shifted_b_pcm,
                            words=b_timed.words,
                            starts=b_timed.starts,
                            ends=b_timed.ends,
                            phonemes=b_timed.phonemes,
                        )
                    else:
                        b_for_fade = b_timed

                    mixed_pcm = crossfade_within_vowel(a_timed, b_for_fade, split)

                    pitch_desc = (
                        f"F0 A={pick.f0_a:.0f}Hz B={pick.f0_b:.0f}Hz shift={pick.shift_ratio:.3f}"
                        if pick.f0_a and pick.f0_b
                        else "F0 unmeasured"
                    )

                    versions = []
                    written = []
                    if 0 in want:
                        write_wav(folder / "0.wav", mixed_pcm)
                        versions.append({
                            "score": 0,
                            "pair": sample.pair.name,
                            "voice_order": order_tag,
                            "voice_a_name": va_name,
                            "voice_b_name": vb_name,
                            "split_word_index": split,
                            "split_word": split_word,
                            "f0_a_hz": pick.f0_a,
                            "f0_b_hz": pick.f0_b,
                            "pitch_shift_ratio": pick.shift_ratio,
                            "rejected_word_indices": pick.rejected_indices,
                            "error_description": (
                                f"Inconsistent — {va_name}→{vb_name} vowel crossfade on "
                                f"{split_word!r} ({pitch_desc})"
                            ),
                            "transcription": "",
                            "sentence": text,
                            "audio_file": "0.wav",
                        })
                        written.append("0.wav")
                    if 1 in want:
                        write_wav(folder / "1.wav", a_timed.pcm)
                        versions.append({
                            "score": 1,
                            "pair": sample.pair.name,
                            "voice_order": order_tag,
                            "voice_a_name": va_name,
                            "voice_b_name": None,
                            "error_description": f"Consistent — {va_name} only",
                            "transcription": "",
                            "sentence": text,
                            "audio_file": "1.wav",
                        })
                        written.append("1.wav")

                    manifest = {
                        "mode": "crossfade",
                        "pair": sample.pair.name,
                        "voice_order": order_tag,
                        "utterance_id": uid,
                        "original_text": text,
                        "target_word": split_word,
                        "versions": versions,
                    }
                    skipped = f", skipped {pick.rejected_indices}" if pick.rejected_indices else ""
                    lines.append(f"  split@{split_word!r}  {pitch_desc}{skipped}  "
                                 f"-> {', '.join(written)}  ({time.time() - t0:.1f}s)")

            else:   # stitch
                joined = f"{text_a} | {text_b}"
                versions = []
                written = []

                if want == {1}:
                    # Consistent only: voice A speaks both sentences — voice B unneeded.
                    a1, a2 = await asyncio.gather(
                        synthesize_with_timestamps(client, text_a, va_id, model_id=model_id),
                        synthesize_with_timestamps(client, text_b, va_id, model_id=model_id),
                    )
                    consistent = stitch(a1, a2, gap_ms=gap_ms)
                    write_wav(folder / "1.wav", consistent)
                    versions.append({
                        "score": 1, "pair": sample.pair.name, "voice_order": order_tag,
                        "voice_a_name": va_name, "voice_b_name": None,
                        "utterance_id_a": uid_a, "utterance_id_b": uid_b, "gap_ms": gap_ms,
                        "error_description": (
                            f"Consistent — {va_name} speaks both (#{uid_a}, #{uid_b})"
                        ),
                        "transcription": "", "sentence": joined, "audio_file": "1.wav",
                    })
                    written.append("1.wav")
                elif want == {0}:
                    # Inconsistent only: voice A on U1, voice B on U2 — no need for voice_a@U2.
                    a1, b2 = await asyncio.gather(
                        synthesize_with_timestamps(client, text_a, va_id, model_id=model_id),
                        synthesize_with_timestamps(client, text_b, vb_id, model_id=model_id),
                    )
                    inconsistent = stitch(a1, b2, gap_ms=gap_ms)
                    write_wav(folder / "0.wav", inconsistent)
                    versions.append({
                        "score": 0, "pair": sample.pair.name, "voice_order": order_tag,
                        "voice_a_name": va_name, "voice_b_name": vb_name,
                        "utterance_id_a": uid_a, "utterance_id_b": uid_b, "gap_ms": gap_ms,
                        "error_description": (
                            f"Inconsistent — {va_name} (#{uid_a}) + {vb_name} (#{uid_b})"
                        ),
                        "transcription": "", "sentence": joined, "audio_file": "0.wav",
                    })
                    written.append("0.wav")
                else:
                    # Both wanted: three renders — voice_a@U1, voice_a@U2, voice_b@U2.
                    a1, a2, b2 = await asyncio.gather(
                        synthesize_with_timestamps(client, text_a, va_id, model_id=model_id),
                        synthesize_with_timestamps(client, text_b, va_id, model_id=model_id),
                        synthesize_with_timestamps(client, text_b, vb_id, model_id=model_id),
                    )
                    inconsistent = stitch(a1, b2, gap_ms=gap_ms)
                    consistent   = stitch(a1, a2, gap_ms=gap_ms)

                    write_wav(folder / "0.wav", inconsistent)
                    write_wav(folder / "1.wav", consistent)
                    versions = [
                        {
                            "score": 0, "pair": sample.pair.name, "voice_order": order_tag,
                            "voice_a_name": va_name, "voice_b_name": vb_name,
                            "utterance_id_a": uid_a, "utterance_id_b": uid_b, "gap_ms": gap_ms,
                            "error_description": (
                                f"Inconsistent — {va_name} (#{uid_a}) + {vb_name} (#{uid_b})"
                            ),
                            "transcription": "", "sentence": joined, "audio_file": "0.wav",
                        },
                        {
                            "score": 1, "pair": sample.pair.name, "voice_order": order_tag,
                            "voice_a_name": va_name, "voice_b_name": None,
                            "utterance_id_a": uid_a, "utterance_id_b": uid_b, "gap_ms": gap_ms,
                            "error_description": (
                                f"Consistent — {va_name} speaks both (#{uid_a}, #{uid_b})"
                            ),
                            "transcription": "", "sentence": joined, "audio_file": "1.wav",
                        },
                    ]
                    written = ["0.wav", "1.wav"]

                manifest = {
                    "mode": "stitch",
                    "pair": sample.pair.name,
                    "voice_order": order_tag,
                    "utterance_id": uid_a,
                    "utterance_id_b": uid_b,
                    "original_text": joined,
                    "target_word": "",
                    "versions": versions,
                }
                lines.append(f"  gap={gap_ms}ms  -> {', '.join(written)}  "
                             f"({time.time() - t0:.1f}s)")

        except Exception as exc:
            lines.append(f"  FAILED ({time.time() - t0:.1f}s): {exc}")
            print("\n" + "\n".join(lines), flush=True)
            return uids, False

        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("\n" + "\n".join(lines), flush=True)
        return uids, True


async def run_speaker_identity_consistency(
    samples: list[SISample],
    output_base: Path,
    cartesia_client: AsyncCartesia,
    *,
    model_id: str = "sonic-3",
    gap_ms: int = 400,
    tts_concurrency: int = 6,
    only_scores: set[int] | None = None,
) -> list[int]:
    """Render each SI sample (score 0 and/or score 1, per only_scores) concurrently.

    Returns the flat list of all utterance IDs consumed by successful samples.
    """
    n_scores = len(only_scores) if only_scores is not None else 2
    sep = "=" * 60
    print(f"\n{sep}\nSpeaker identity consistency: {len(samples)} sample(s) "
          f"= {len(samples)*n_scores} WAV(s) across {len({s.pair.name for s in samples})} "
          f"voice pair(s)\n{sep}")
    output_base.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(tts_concurrency)
    results = await asyncio.gather(*(
        _render_si_sample(
            cartesia_client, s, output_base,
            model_id=model_id, gap_ms=gap_ms,
            total=len(samples), semaphore=sem,
            only_scores=only_scores,
        )
        for s in samples
    ))
    consumed: list[int] = []
    for uids, ok in results:
        if ok:
            consumed.extend(uids)
    return consumed
