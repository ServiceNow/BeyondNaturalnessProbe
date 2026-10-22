"""
Acoustic flattening of speech audio — canonical post-processing step
for the expressiveness dimension's score-0 "under" wavs.

Goal: reduce perceived expressiveness while preserving intelligibility,
speaker identity, and acceptable voice quality.

Canonical defaults (validated by listening on the 10 Harvard sentences
under experiments/expressiveness_pairs/*_under/):
  --backend praat        # PSOLA in time domain, preserves voice quality
  --alpha 0.5            # halves F0 excursion around speaker median
  --flatten-timing       # compress outlier-long vowels in time
  --timing-threshold 2.5 # only egregious lengthening (>=2.5x median vowel)
  --timing-floor 0.5     # cap vowel compression at 0.5x (PSOLA-safe)
  (intensity compression on by default; gate -35dB below peak.)

Rationale (Banse & Scherer 1996; Juslin & Laukka 2003):
  1) F0 dynamics (range/variability around speaker mean) is the
     largest single contributor to perceived expressiveness.
     Compressing F0 toward the median is the strongest single move.
  2) Intensity dynamics (amplitude envelope excursion) is additive
     and second-order. Compress modestly once F0 is compressed.
  3) Temporal variability (final lengthening / pause variance) is
     real but high-risk to manipulate; off by default.
  4) Voice-quality cues are left untouched.

Per-file pipeline (default):
  * WORLD analysis (pyworld): F0 + spectral envelope + aperiodicity.
  * Median F0 over voiced frames (median > mean — robust to octave errors).
  * F0' = median + alpha * (F0 - median) on voiced frames, alpha=0.3.
    Floor alpha=0.15 (refuse lower; sounds robotic).
  * WORLD resynthesis with original spectral envelope + aperiodicity
    (preserves timbre and speaker identity).
  * Optional: intensity-envelope compression on a 20ms RMS contour,
    ratio ~3:1 toward the median, then renormalize loudness.
  * Optional: --flatten-timing pass — placeholder; emits a warning and
    skips, since reliable boundary-final lengthening compression needs
    forced alignment we don't have set up here.

CLI:
  python experiments/flatten_expressiveness.py \\
      --input-dir  experiments/expressiveness_pairs \\
      --output-dir experiments/expressiveness_flattened \\
      [--alpha 0.3] [--skip-intensity] [--flatten-timing]

Filters: input walk picks only files matching '0_under_*.wav', mirrors
the relative directory structure under --output-dir, and preserves
filename, sample rate, and bit depth.

Verification: for every file, prints F0 median / SD / range / mean and
RMS dB-SD before and after, plus the realised alpha and intensity ratio.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.signal import find_peaks

# WORLD frame interval (ms) — pyworld default; explicit here so the
# intensity stage uses a matching frame rate.
FRAME_PERIOD_MS = 5.0
INTENSITY_FRAME_MS = 20.0
ALPHA_FLOOR = 0.15


@dataclass
class F0Stats:
    median: float
    sd: float
    range: float
    mean: float
    voiced_frames: int


def _f0_stats(f0: np.ndarray) -> F0Stats:
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return F0Stats(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    return F0Stats(
        median=float(np.median(voiced)),
        sd=float(np.std(voiced)),
        range=float(np.max(voiced) - np.min(voiced)),
        mean=float(np.mean(voiced)),
        voiced_frames=int(voiced.size),
    )


def _voicing_envelope(
    f0: np.ndarray, n_samples: int, sr: int, frame_period_ms: float,
    *, ramp_ms: float = 20.0,
) -> np.ndarray:
    """
    Build a sample-rate weight envelope (∈ [0, 1]) — 1 where WORLD says
    the frame is voiced, 0 where unvoiced, with a short ramp at the
    boundaries so crossfades are click-free.
    """
    voiced = (f0 > 0).astype(np.float64)
    hop_samples = sr * frame_period_ms / 1000.0
    frame_centers = np.arange(voiced.size) * hop_samples
    sample_idx = np.arange(n_samples)
    weight = np.interp(sample_idx, frame_centers, voiced, left=0.0, right=0.0)
    ramp_samples = max(3, int(round(sr * ramp_ms / 1000.0)))
    if weight.size >= ramp_samples:
        kernel = np.ones(ramp_samples) / ramp_samples
        weight = np.convolve(weight, kernel, mode="same")
    return np.clip(weight, 0.0, 1.0)


def _compress_f0(f0: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Pull voiced F0 toward the median. Returns (new_f0, median_used)."""
    voiced_mask = f0 > 0
    if not voiced_mask.any():
        return f0.copy(), float("nan")
    median = float(np.median(f0[voiced_mask]))
    out = f0.copy()
    out[voiced_mask] = median + alpha * (f0[voiced_mask] - median)
    # Guard: F0 must stay strictly positive on voiced frames; clamp at
    # ~50 Hz to be safe (well below typical male F0 floor).
    out[voiced_mask] = np.maximum(out[voiced_mask], 50.0)
    return out, median


def _compress_intensity(
    y: np.ndarray, sr: int, ratio: float = 3.0,
    *, gate_db_below_peak: float = 35.0, smooth_frames: int = 9,
) -> tuple[np.ndarray, float, float]:
    """
    Mild dynamic-range compression on the amplitude envelope of SPEECH only.

    Compute log-RMS in 20ms frames, identify a speech mask via a peak-
    relative gate (frames within `gate_db_below_peak` of the peak count as
    speech), compute the reference median from speech frames only, and
    compress speech-frame log-RMS toward that median by `ratio`. Non-speech
    frames keep gain = 1.0 so silences are NOT boosted.

    Loudness preservation renormalizes against voiced-only RMS, not overall
    RMS — so silences don't drag the makeup gain around and speech stays
    at its original level.

    Returns (compressed_y, pre_rms_db_sd_speech, post_rms_db_sd_speech).
    """
    hop = max(1, int(round(sr * INTENSITY_FRAME_MS / 1000.0)))
    if y.size < hop * 2:
        return y, float("nan"), float("nan")

    n_frames = y.size // hop
    if n_frames == 0:
        return y, float("nan"), float("nan")
    framed = y[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1))
    rms = np.maximum(rms, 1e-6)
    log_rms = 20.0 * np.log10(rms)

    # Speech gate: frames within `gate_db_below_peak` of the peak count as
    # speech. Anything quieter is silence/breath/leading-trailing tail and
    # gets gain=1.0 (no change), which fixes the pop-on-silence bug.
    peak_db = float(np.max(log_rms))
    speech_mask = log_rms > (peak_db - gate_db_below_peak)
    if speech_mask.sum() < 4:
        # Not enough speech to compress; pass through.
        return y, float("nan"), float("nan")

    median_db = float(np.median(log_rms[speech_mask]))
    pre_sd_speech = float(np.std(log_rms[speech_mask]))

    # Per-frame target dB: speech frames compressed toward median; silence
    # frames untouched (target_db = log_rms there → delta_db = 0).
    target_db = log_rms.copy()
    target_db[speech_mask] = (
        median_db + (log_rms[speech_mask] - median_db) / ratio
    )
    delta_db = target_db - log_rms

    # Smooth the gain envelope so the gate edges (speech ↔ silence) don't
    # snap. ~180 ms at 20 ms hop = 9 frames.
    gain_lin = 10.0 ** (delta_db / 20.0)
    if gain_lin.size >= smooth_frames:
        kern = np.ones(smooth_frames) / smooth_frames
        gain_lin = np.convolve(gain_lin, kern, mode="same")

    # Upsample frame gains to sample-rate via linear interpolation.
    frame_centers = (np.arange(n_frames) + 0.5) * hop
    sample_idx = np.arange(y.size)
    gain_env = np.interp(sample_idx, frame_centers, gain_lin,
                         left=gain_lin[0], right=gain_lin[-1])

    out = y * gain_env

    # Renormalize against VOICED-only RMS (not overall RMS). With the
    # overall-RMS approach, boosted silence pulled the divisor up and the
    # speech got quieter; using voiced-only RMS makes makeup gain depend
    # purely on what changed in the speech region.
    speech_frames_pre  = framed[speech_mask]
    voiced_rms_pre = float(np.sqrt(np.mean(speech_frames_pre.astype(np.float64) ** 2)))
    framed_post = out[: n_frames * hop].reshape(n_frames, hop)
    speech_frames_post = framed_post[speech_mask]
    voiced_rms_post = float(np.sqrt(np.mean(speech_frames_post.astype(np.float64) ** 2)))
    if voiced_rms_post > 0:
        out = out * (voiced_rms_pre / voiced_rms_post)

    # Post-RMS-SD reported on the speech frames only.
    framed_post = out[: n_frames * hop].reshape(n_frames, hop)
    rms_post = np.maximum(
        np.sqrt(np.mean(framed_post.astype(np.float64) ** 2, axis=1)),
        1e-6,
    )
    post_sd_speech = float(np.std(20.0 * np.log10(rms_post[speech_mask])))
    return out, pre_sd_speech, post_sd_speech


def _load_mono(path: Path) -> tuple[np.ndarray, int, str]:
    """Load a wav as float64 mono; return (samples, sample_rate, subtype)."""
    y, sr = sf.read(str(path), always_2d=False)
    info = sf.info(str(path))
    if y.ndim == 2:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y.astype(np.float64)), sr, info.subtype


def _save_like(path: Path, y: np.ndarray, sr: int, subtype: str) -> None:
    """Write WAV preserving the source subtype (bit depth)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clip to avoid integer overflow if subtype is PCM.
    y_out = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y_out, sr, subtype=subtype)


def _flatten_world(
    y: np.ndarray, sr: int, alpha: float,
) -> tuple[np.ndarray, F0Stats, F0Stats]:
    """WORLD-backed pitch flatten with unvoiced passthrough."""
    f0, t  = pw.harvest(y, sr, frame_period=FRAME_PERIOD_MS)
    f0     = pw.stonemask(y, f0, t, sr)
    sp     = pw.cheaptrick(y, f0, t, sr)
    ap     = pw.d4c(y, f0, t, sr)

    pre = _f0_stats(f0)
    f0_new, _ = _compress_f0(f0, alpha)
    post = _f0_stats(f0_new)

    y_world = pw.synthesize(f0_new, sp, ap, sr, frame_period=FRAME_PERIOD_MS)

    # Pass original audio through on unvoiced frames; vocoder artifacts
    # on fricatives are bypassed entirely.
    n = min(y.size, y_world.size)
    weight = _voicing_envelope(f0_new, n, sr, FRAME_PERIOD_MS)
    y_new = weight * y_world[:n] + (1.0 - weight) * y[:n]
    return y_new, pre, post


def _longest_voiced_run(
    pitch_times: np.ndarray, start: float, end: float, *, max_gap: float = 0.04,
) -> tuple[float, float] | None:
    """
    Within [start, end], return the longest contiguous run of pitch-tier
    times (gaps ≤ max_gap apart). Used to identify the vowel inside an
    inter-nucleus interval — consonants are unvoiced or have brief voicing,
    so the longest connected run is almost always the vowel.
    """
    in_range = pitch_times[(pitch_times >= start) & (pitch_times <= end)]
    if in_range.size == 0:
        return None
    best = (float(in_range[0]), float(in_range[0]))
    seg_s = seg_e = float(in_range[0])
    for t in in_range[1:].tolist():
        if t - seg_e > max_gap:
            if seg_e - seg_s > best[1] - best[0]:
                best = (seg_s, seg_e)
            seg_s = t
        seg_e = t
    if seg_e - seg_s > best[1] - best[0]:
        best = (seg_s, seg_e)
    # Reject runs too short to be a vowel (< 50 ms) — likely voiced
    # consonant transients (nasals, voiced stops).
    if best[1] - best[0] < 0.05:
        return None
    return best


def _detect_long_vowels(
    snd: parselmouth.Sound,
    *, threshold_ratio: float = 1.3, floor_ratio: float = 0.5,
    smooth_frames: int = 10, min_peak_distance_s: float = 0.12,
    intensity_pctile_floor: float = 60.0, drop_db: float = 6.0,
    max_extent_s: float = 0.12, min_peak_prominence_db: float = 3.0,
) -> tuple[list[tuple[float, float, float]], dict]:
    """
    Per-vowel heuristic timing detection — finds vowel *regions* (not
    inter-peak intervals) and flags vowels that are unusually long.

    Algorithm:
      1) Smooth the intensity contour, find peaks (syllable nuclei).
      2) For each peak, walk outward (left, then right) until intensity
         drops `drop_db` below the peak OR we hit a neighboring peak.
         That bounded region is the vowel.
      3) Compute the file's median vowel duration. Flag vowels whose
         duration exceeds `threshold_ratio` × median; the target
         multiplier compresses them back toward the median (clamped to
         `floor_ratio`).

    Why this works where inter-peak intervals didn't: pre-boundary or
    emphasis lengthening is concentrated in the vowel ITSELF. Inter-peak
    intervals span from one nucleus across the trailing consonants into
    the next nucleus — compressing that region hits whatever sits between
    the peaks (often the next function word), not the lengthened vowel.

    Returns: (list of (vowel_start_s, vowel_end_s, target_multiplier),
              stats_dict).
    """
    intensity = snd.to_intensity()
    values = np.asarray(intensity.values).flatten().astype(np.float64)
    xs = np.asarray(intensity.xs())
    if values.size < 4 or xs.size != values.size:
        return [], {"reason": "too short / intensity x-mismatch"}

    if smooth_frames >= 3 and values.size >= smooth_frames:
        kernel = np.ones(smooth_frames) / smooth_frames
        values_smooth = np.convolve(values, kernel, mode="same")
    else:
        values_smooth = values.copy()

    height = float(np.percentile(values_smooth, intensity_pctile_floor))
    frame_step = float(xs[1] - xs[0]) if xs.size >= 2 else 0.01
    min_dist_frames = max(1, int(round(min_peak_distance_s / frame_step)))
    max_extent_frames = max(1, int(round(max_extent_s / frame_step)))

    peak_idx, _ = find_peaks(
        values_smooth,
        height=height,
        distance=min_dist_frames,
        prominence=min_peak_prominence_db,
    )
    if peak_idx.size < 3:
        return [], {
            "n_peaks": int(peak_idx.size),
            "median_vowel_dur_s": float("nan"),
            "reason": "fewer than 3 peaks detected",
        }

    # For each peak, walk outward to find the vowel boundaries.
    vowels: list[tuple[float, float, float]] = []  # (start_s, end_s, peak_time_s)
    for i, pi in enumerate(peak_idx):
        peak_val = float(values_smooth[pi])
        thr = peak_val - drop_db
        prev_p = int(peak_idx[i - 1]) if i > 0 else 0
        next_p = int(peak_idx[i + 1]) if i < peak_idx.size - 1 else values_smooth.size - 1

        # Left bound — stop at the midpoint between peaks (don't cross),
        # at the -drop_db level, or at max_extent.
        left_limit = max(prev_p + 1, (prev_p + pi) // 2, pi - max_extent_frames)
        L = pi
        while L > left_limit and values_smooth[L - 1] > thr:
            L -= 1
        # Right bound — symmetric.
        right_limit = min(next_p - 1, (pi + next_p) // 2, pi + max_extent_frames)
        R = pi
        while R < right_limit and values_smooth[R + 1] > thr:
            R += 1
        vowels.append((float(xs[L]), float(xs[R]), float(xs[pi])))

    durations = np.array([end - start for start, end, _ in vowels])
    median_dur = float(np.median(durations))
    long_thr = threshold_ratio * median_dur

    flagged: list[tuple[float, float, float]] = []
    for (start, end, peak_t), dur in zip(vowels, durations.tolist()):
        if dur > long_thr:
            target = max(floor_ratio, median_dur / dur)
            flagged.append((start, end, float(target)))

    return flagged, {
        "n_peaks":            len(vowels),
        "median_vowel_dur_s": median_dur,
        "max_vowel_dur_s":    float(durations.max()),
        "n_flagged":          len(flagged),
    }


# Back-compat alias — older code path can still call the old name.
_detect_long_intervals = _detect_long_vowels


def _clean_octave_errors(
    f0s: np.ndarray,
    times: np.ndarray | None = None,
    snd: parselmouth.Sound | None = None,
    *, halve_threshold: float = 0.55, double_threshold: float = 1.80,
) -> tuple[np.ndarray, int]:
    """
    Repair octave-halving and octave-doubling errors in a pitch-tier sequence.

    Praat's pulse-based pitch detection can lock onto a subharmonic when
    the source has period-doubling jitter (creak / diplophonia) or onto
    the second harmonic on strong-initial-burst pulses. PSOLA driven by
    such a tier imposes wrong periods, audible as a robotic step at the
    transition into the affected region.

    Heuristic: compute a robust central F0 from the IQR-trimmed values
    (excludes octave-error outliers from the reference itself). Any point
    above `double_threshold` × ref gets halved; any point below
    `halve_threshold` × ref gets doubled. Single pass, no running-median
    cascade.

    halve_threshold=0.55 and double_threshold=1.80 sit close to the
    exact-octave ratios of 0.5 and 2.0. Anything ≥ 1.8× or ≤ 0.55× ref
    is so far from natural pitch variation that it's almost certainly an
    octave detection error rather than a real high pitch accent or a
    real creak transient.

    `times` and `snd` are accepted for API compatibility (an earlier
    AC-cross-validation strategy needed them) but are unused.

    Returns (cleaned, n_fixed).
    """
    del times, snd  # accepted for API compat
    if f0s.size == 0:
        return f0s.copy(), 0

    p25, p75 = np.percentile(f0s, [25, 75])
    iqr_mask = (f0s >= p25) & (f0s <= p75)
    if iqr_mask.sum() < 3:
        return f0s.copy(), 0
    ref = float(np.median(f0s[iqr_mask]))

    cleaned = f0s.copy()
    halved_mask  = cleaned < halve_threshold  * ref
    doubled_mask = cleaned > double_threshold * ref

    cleaned[halved_mask]  *= 2.0
    cleaned[doubled_mask] /= 2.0
    n_fixed = int(halved_mask.sum() + doubled_mask.sum())
    return cleaned, n_fixed


def _flatten_praat(
    src_path: Path, alpha: float,
    *, pitch_floor: float = 75.0, pitch_ceiling: float = 500.0,
    flatten_timing: bool = False,
    timing_threshold: float = 1.3, timing_floor: float = 0.7,
) -> tuple[np.ndarray, int, F0Stats, F0Stats, dict]:
    """
    Praat PSOLA-backed pitch flatten.

    PSOLA modifies pitch period spacing in the time domain rather than
    decomposing/resynthesizing the spectrum. Voice quality, fricatives,
    and overall timbre are preserved much more faithfully than WORLD —
    the cost is that PSOLA can't manipulate aperiodicity or spectral
    envelope, but we don't want to touch those anyway.

    Pre-step: the extracted pitch tier is octave-cleaned (subharmonic
    detections at sentence-end creak get repaired) so PSOLA doesn't try
    to stretch pulses to half-rate on a signal whose real glottal
    periods stayed at the doubled rate. Without this pass, the
    constant-median compression formula produces an audible step at the
    transition from modal voicing into the period-doubled tail.

    Returns (y, sr, pre_stats, post_stats).
    """
    snd = parselmouth.Sound(str(src_path))
    sr = int(snd.sampling_frequency)
    manip = praat_call(snd, "To Manipulation", 0.005, pitch_floor, pitch_ceiling)
    pt    = praat_call(manip, "Extract pitch tier")

    timing_info: dict = {}

    n_pts = int(praat_call(pt, "Get number of points"))
    if n_pts == 0:
        y_orig, _ = sf.read(str(src_path), always_2d=False)
        if y_orig.ndim == 2:
            y_orig = y_orig.mean(axis=1)
        empty = F0Stats(float("nan"), float("nan"), float("nan"), float("nan"), 0)
        return y_orig.astype(np.float64), sr, empty, empty, timing_info

    times = np.array([praat_call(pt, "Get time from index", i + 1)  for i in range(n_pts)])
    f0s   = np.array([praat_call(pt, "Get value at index", i + 1) for i in range(n_pts)])

    # Repair octave-halving / octave-doubling errors in the pitch tier
    # before computing statistics or new values — otherwise PSOLA imposes
    # wrong periods on regions where the manipulation tier locked onto a
    # subharmonic or superharmonic (see _clean_octave_errors docstring).
    f0s, _ = _clean_octave_errors(f0s, times=times, snd=snd)

    pre = F0Stats(
        median=float(np.median(f0s)), sd=float(np.std(f0s)),
        range=float(f0s.max() - f0s.min()), mean=float(np.mean(f0s)),
        voiced_frames=n_pts,
    )

    median = pre.median
    new_f0s = np.maximum(median + alpha * (f0s - median), pitch_floor)
    post = F0Stats(
        median=float(np.median(new_f0s)), sd=float(np.std(new_f0s)),
        range=float(new_f0s.max() - new_f0s.min()), mean=float(np.mean(new_f0s)),
        voiced_frames=n_pts,
    )

    # Replace pitch-tier points in place.
    for _ in range(n_pts):
        praat_call(pt, "Remove point", 1)
    for t, f in zip(times, new_f0s):
        praat_call(pt, "Add point", float(t), float(f))
    praat_call([manip, pt], "Replace pitch tier")

    # Optionally apply a DurationTier to compress unusually long vowels.
    # _detect_long_vowels returns the vowel region directly (walked
    # outward from each intensity peak to the -6 dB drop), so we apply
    # the DurationTier multiplier to that region — consonants and
    # neighbouring words stay original-length.
    if flatten_timing:
        flagged, info = _detect_long_vowels(
            snd, threshold_ratio=timing_threshold, floor_ratio=timing_floor,
        )
        timing_info = info | {"flagged": flagged}
        if flagged:
            dur_tier = praat_call(manip, "Extract duration tier")
            eps = 0.005
            for start, end, mult in flagged:
                praat_call(dur_tier, "Add point", float(start - eps), 1.0)
                praat_call(dur_tier, "Add point", float(start + eps), float(mult))
                praat_call(dur_tier, "Add point", float(end - eps),   float(mult))
                praat_call(dur_tier, "Add point", float(end + eps),   1.0)
            praat_call([manip, dur_tier], "Replace duration tier")

    out_snd = praat_call(manip, "Get resynthesis (overlap-add)")
    y_new = np.asarray(out_snd.values).flatten().astype(np.float64)

    # Splice raw audio through creak regions. PSOLA stretches pulse
    # spacing to match a target F0 — that works on regular modal voicing
    # but mangles vocal fry, where glottal pulses are inherently
    # irregular (period jitter, low energy, often subharmonic structure).
    # Forcing those onto a flattened target produces audible glitches.
    # We detect creak from the autocorrelation pitch contour (any voiced
    # frame whose F0 is significantly below the modal range counts) plus
    # any frame the AC tracker marks unvoiced, and crossfade the raw
    # audio back in over those regions.
    y_orig = np.asarray(snd.values).flatten().astype(np.float64)
    y_new  = _splice_creak_from_raw(y_orig, y_new, snd, sr)

    return y_new, sr, pre, post, timing_info


def _splice_creak_from_raw(
    y_raw: np.ndarray, y_psola: np.ndarray, snd: parselmouth.Sound,
    sr: int, *, creak_threshold_ratio: float = 0.70,
    crossfade_ms: float = 15.0,
) -> np.ndarray:
    """
    Where the source contains creak / vocal fry, replace the PSOLA
    resynthesis with the raw audio (crossfaded at boundaries).

    Creak detection: a frame is "creak" when the autocorrelation pitch
    tracker reports F0 below `creak_threshold_ratio` × the global voiced
    median, OR reports unvoiced where a neighbouring voiced frame exists
    (gaps inside a voiced region). The two outputs are crossfaded at
    region boundaries over `crossfade_ms` to avoid clicks.

    If the input arrays differ in length, the shorter one defines the
    return length (PSOLA can slightly compress/expand duration if
    timing_flatten is on; in that case the splice is best-effort).
    """
    n = min(y_raw.size, y_psola.size)
    if n == 0:
        return y_psola

    pitch_ac = snd.to_pitch_ac(
        time_step=0.005, pitch_floor=75.0, pitch_ceiling=600.0,
        octave_jump_cost=2.0,
    )
    f0_ac = pitch_ac.selected_array["frequency"]
    voiced = f0_ac[f0_ac > 0]
    if voiced.size == 0:
        return y_psola
    voiced_median = float(np.median(voiced))
    creak_thr = voiced_median * creak_threshold_ratio

    # Per-frame creak mask: F0 == 0 (unvoiced) OR F0 < creak threshold.
    # We treat unvoiced as creak here so that aspiration/release tails
    # also pass through unmodified (PSOLA on unvoiced is also iffy).
    mask = (f0_ac == 0) | (f0_ac < creak_thr)

    # Smooth: any run of length 1 between voiced frames is likely a
    # spurious tracker dropout, ignore those (set mask back to False).
    # And require a minimum creak run length so we don't splice on
    # transient single-frame quirks.
    min_run = 3
    smoothed = mask.copy()
    i = 0
    while i < smoothed.size:
        if smoothed[i]:
            j = i
            while j < smoothed.size and smoothed[j]:
                j += 1
            if j - i < min_run:
                smoothed[i:j] = False
            i = j
        else:
            i += 1
    mask = smoothed
    if not mask.any():
        return y_psola

    # Sample-rate gate from the frame-rate mask.
    xs = np.asarray(pitch_ac.xs())
    sample_idx = np.arange(n)
    frame_centers = xs * sr
    gate = np.interp(sample_idx, frame_centers, mask.astype(np.float64),
                     left=0.0, right=0.0)
    gate = np.clip(gate, 0.0, 1.0)

    # Smooth the gate by a Hann-shaped kernel of crossfade_ms to soften
    # the edges of each spliced region into a real crossfade.
    cf_samples = max(3, int(round(sr * crossfade_ms / 1000.0)))
    kernel = np.hanning(2 * cf_samples + 1)
    kernel /= kernel.sum()
    gate = np.convolve(gate, kernel, mode="same")

    out = y_psola.copy()
    out[:n] = (1.0 - gate) * y_psola[:n] + gate * y_raw[:n]
    return out


def flatten_one(
    src: Path, dst: Path, *,
    alpha: float, skip_intensity: bool, flatten_timing: bool,
    backend: str = "praat",
    timing_threshold: float = 1.3, timing_floor: float = 0.7,
) -> dict:
    timing_info: dict = {}
    if backend == "praat":
        y_new, sr, pre, post_f0_stats, timing_info = _flatten_praat(
            src, alpha,
            flatten_timing=flatten_timing,
            timing_threshold=timing_threshold,
            timing_floor=timing_floor,
        )
        subtype = sf.info(str(src)).subtype
    else:  # world
        if flatten_timing:
            warnings.warn(
                "--flatten-timing is implemented only for --backend praat; "
                "ignored for WORLD backend.",
                stacklevel=2,
            )
        y, sr, subtype = _load_mono(src)
        y_new, pre, post_f0_stats = _flatten_world(y, sr, alpha)

    intensity_pre = intensity_post = float("nan")
    if not skip_intensity:
        y_new, intensity_pre, intensity_post = _compress_intensity(y_new, sr)

    _save_like(dst, y_new, sr, subtype)

    return {
        "src": str(src),
        "dst": str(dst),
        "sr": sr,
        "subtype": subtype,
        "alpha": alpha,
        "f0_median": pre.median,
        "f0_sd_pre":  pre.sd,
        "f0_sd_post": post_f0_stats.sd,
        "f0_range_pre":  pre.range,
        "f0_range_post": post_f0_stats.range,
        "f0_mean_pre":   pre.mean,
        "f0_mean_post":  post_f0_stats.mean,
        "voiced_frames": pre.voiced_frames,
        "intensity_db_sd_pre":  intensity_pre,
        "intensity_db_sd_post": intensity_post,
        "timing":      timing_info,
    }


def _walk_under_wavs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("0_under_*.wav"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[2].strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument("--input-dir",  type=Path,
                   default=repo_root / "experiments" / "expressiveness_pairs",
                   help="Directory containing the under wavs to flatten.")
    p.add_argument("--output-dir", type=Path,
                   default=repo_root / "experiments" / "expressiveness_flattened",
                   help="Where to write the flattened outputs (mirrors structure).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="F0 compression ratio toward median. 0.5 cuts the "
                        "F0 excursion around the median in half without "
                        f"sounding robotic. Floor enforced at {ALPHA_FLOOR}.")
    p.add_argument("--backend", choices=["praat", "world"], default="praat",
                   help="Pitch-flatten engine. Praat PSOLA preserves voice "
                        "quality better; WORLD allows joint spectral envelope "
                        "+ aperiodicity manipulation we don't currently use.")
    p.add_argument("--skip-intensity", action="store_true",
                   help="Disable the amplitude-envelope compression stage.")
    p.add_argument("--flatten-timing", action="store_true",
                   help="Compress inter-nucleus intervals that are longer than "
                        "timing_threshold * median back toward the median. "
                        "Uses heuristic syllable detection from the intensity "
                        "contour (no forced aligner). praat backend only.")
    p.add_argument("--timing-threshold", type=float, default=2.5,
                   help="A vowel longer than this multiple of the file's "
                        "median vowel duration is flagged for compression. "
                        "2.5× restricts flagging to clear outliers (egregious "
                        "lengthening). Lower it (e.g. 1.7) for more aggressive "
                        "flattening of moderate lengthening too.")
    p.add_argument("--timing-floor", type=float, default=0.5,
                   help="Floor on the DurationTier multiplier applied to the "
                        "VOWEL inside each flagged interval. PSOLA stays clean "
                        "down to ~0.5; lower than ~0.45 starts to chipmunk.")
    args = p.parse_args()

    if args.alpha < ALPHA_FLOOR:
        sys.exit(f"--alpha {args.alpha} below floor {ALPHA_FLOOR} — refusing "
                 "(below this the result reads as monotone/robotic).")
    if args.alpha > 1.0:
        sys.exit(f"--alpha {args.alpha} would AMPLIFY F0 variation; expected ≤ 1.")

    wavs = _walk_under_wavs(args.input_dir)
    if not wavs:
        sys.exit(f"No 0_under_*.wav files under {args.input_dir}")

    print(f"Found {len(wavs)} under wavs under {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"alpha={args.alpha}  skip_intensity={args.skip_intensity}  "
          f"flatten_timing={args.flatten_timing}\n")

    results: list[dict] = []
    for i, src in enumerate(wavs, 1):
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        try:
            r = flatten_one(
                src, dst,
                alpha=args.alpha,
                skip_intensity=args.skip_intensity,
                flatten_timing=args.flatten_timing,
                backend=args.backend,
                timing_threshold=args.timing_threshold,
                timing_floor=args.timing_floor,
            )
        except Exception as exc:
            print(f"[{i:>2}/{len(wavs)}] FAILED {rel}: {exc}", flush=True)
            continue

        d_sd  = r["f0_sd_post"]  - r["f0_sd_pre"]
        d_rng = r["f0_range_post"] - r["f0_range_pre"]
        intensity_str = (
            f"intensity dB SD {r['intensity_db_sd_pre']:.1f} -> "
            f"{r['intensity_db_sd_post']:.1f}"
            if not np.isnan(r["intensity_db_sd_pre"])
            else "intensity skipped"
        )
        timing = r.get("timing") or {}
        if timing.get("flagged"):
            n_flag = len(timing["flagged"])
            med_ms = timing.get("median_vowel_dur_s", float("nan")) * 1000
            spans = ", ".join(
                f"{a:.2f}-{b:.2f}s×{m:.2f}" for a, b, m in timing["flagged"]
            )
            timing_str = (f"timing: {n_flag} long vowel(s) flagged "
                          f"(median vowel {med_ms:.0f} ms) -> {spans}")
        elif timing:
            timing_str = (f"timing: {timing.get('n_peaks','?')} peaks, "
                          f"none long")
        else:
            timing_str = "timing: skipped"
        print(
            f"[{i:>2}/{len(wavs)}] {rel}\n"
            f"     F0 median {r['f0_median']:.1f} Hz  "
            f"voiced {r['voiced_frames']} frames\n"
            f"     F0 SD    {r['f0_sd_pre']:.1f} -> {r['f0_sd_post']:.1f} Hz "
            f"({d_sd:+.1f})\n"
            f"     F0 range {r['f0_range_pre']:.1f} -> {r['f0_range_post']:.1f} Hz "
            f"({d_rng:+.1f})\n"
            f"     {intensity_str}\n"
            f"     {timing_str}",
            flush=True,
        )
        results.append(r)

    if not results:
        sys.exit("No files flattened successfully.")

    # Summary
    def _agg(key):
        vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
        return (np.mean(vals), np.median(vals)) if vals else (float("nan"), float("nan"))

    sd_pre_mean,  sd_pre_med  = _agg("f0_sd_pre")
    sd_post_mean, sd_post_med = _agg("f0_sd_post")
    rng_pre_mean,  rng_pre_med  = _agg("f0_range_pre")
    rng_post_mean, rng_post_med = _agg("f0_range_post")

    print(f"\n=== Summary across {len(results)} files ===")
    print(f"  F0 SD    mean: {sd_pre_mean:.1f} -> {sd_post_mean:.1f} Hz "
          f"({sd_post_mean - sd_pre_mean:+.1f})")
    print(f"  F0 SD    median: {sd_pre_med:.1f} -> {sd_post_med:.1f} Hz")
    print(f"  F0 range mean: {rng_pre_mean:.1f} -> {rng_post_mean:.1f} Hz "
          f"({rng_post_mean - rng_pre_mean:+.1f})")
    print(f"  F0 range median: {rng_pre_med:.1f} -> {rng_post_med:.1f} Hz")
    if not args.skip_intensity:
        ipre_mean, _  = _agg("intensity_db_sd_pre")
        ipost_mean, _ = _agg("intensity_db_sd_post")
        print(f"  Intensity dB SD mean: {ipre_mean:.2f} -> {ipost_mean:.2f}")


if __name__ == "__main__":
    main()
