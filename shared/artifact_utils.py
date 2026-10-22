"""
Audio artifact post-processing for the human_plausibility pipeline.

Reads a clean WAV synthesized by Cartesia, applies one or more digital
artifact recipes from the OpenAI manifest, and writes a degraded WAV.

All artifact functions operate on float32 arrays normalized to [-1, 1].
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (float32 audio [-1,1], sample_rate, n_channels)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data, sample_rate, n_channels


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, n_channels: int) -> None:
    """Write float32 audio [-1,1] to a 16-bit PCM WAV file."""
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def _get_region(
    audio: np.ndarray,
    location: str,
    sample_rate: int,
    sentence: str,
    target_word: str,
) -> tuple[int, int]:
    """Return (start_sample, end_sample) for the artifact's target region."""
    n = len(audio)
    if location == "target_word":
        start_frac, end_frac = _estimate_word_bounds(sentence, target_word)
    elif location == "sentence_start":
        start_frac, end_frac = 0.0, 0.20
    elif location == "sentence_middle":
        start_frac, end_frac = 0.20, 0.80
    elif location == "sentence_end":
        start_frac, end_frac = 0.80, 1.0
    else:  # "throughout" or unknown
        start_frac, end_frac = 0.0, 1.0
    return int(n * start_frac), int(n * end_frac)


def _estimate_word_bounds(sentence: str, target_word: str) -> tuple[float, float]:
    """Estimate word time position from its character position in the sentence."""
    s = sentence.strip()
    pos = s.lower().find(target_word.lower())
    if pos == -1 or not s:
        return 0.25, 0.75
    start_frac = max(0.0, pos / len(s) - 0.05)
    end_frac = min(1.0, (pos + len(target_word)) / len(s) + 0.05)
    return start_frac, end_frac


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEV = {"mild": 0.25, "moderate": 0.6, "severe": 1.0}

def sev(severity: str) -> float:
    return _SEV.get(severity, 0.5)


# ---------------------------------------------------------------------------
# Artifact implementations
# Each receives a float32 region array and returns a transformed float32 array
# of the same length.
# ---------------------------------------------------------------------------

def _hard_clipping(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    threshold = 1.0 - sev(severity) * 0.72
    return np.clip(region, -threshold, threshold)


def _soft_clipping(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    gain = 1.0 + sev(severity) * 5.0
    return (np.tanh(region * gain) / np.tanh(np.array(gain))).astype(np.float32)


def _bit_crushing(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    bits = max(4, int(16 - sev(severity) * 12))
    steps = float(2 ** bits)
    return (np.round(region * steps / 2.0) / (steps / 2.0)).astype(np.float32)


def _white_noise(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    noise = np.random.normal(0.0, sev(severity) * 0.18, len(region)).astype(np.float32)
    return np.clip(region + noise, -1.0, 1.0)


def _pink_noise(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    n = len(region)
    white = np.random.randn(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    freqs[0] = 1.0
    fft = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n).astype(np.float32)
    peak = np.max(np.abs(pink))
    if peak > 1e-8:
        pink = pink / peak * sev(severity) * 0.14
    return np.clip(region + pink, -1.0, 1.0)


def _em_hum(region: np.ndarray, sr: int, severity: str, freq: float) -> np.ndarray:
    t = np.arange(len(region), dtype=np.float32) / sr
    level = sev(severity) * 0.18
    hum = (
        np.sin(2 * np.pi * freq * t) * level
        + np.sin(2 * np.pi * freq * 2 * t) * level * 0.4
        + np.sin(2 * np.pi * freq * 3 * t) * level * 0.2
    )
    return np.clip(region + hum, -1.0, 1.0)


def _em_hum_50hz(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    return _em_hum(region, sr, severity, 50.0)


def _em_hum_60hz(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    return _em_hum(region, sr, severity, 60.0)


def _buffer_dropout(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    chunk = int(sr * (0.04 + sev(severity) * 0.22))
    result = region.copy()
    if len(result) > chunk:
        start = np.random.randint(0, len(result) - chunk)
        result[start:start + chunk] = 0.0
    return result


def _stutter_repeat(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    slice_len = max(int(sr * 0.015), int(sr * (0.05 - sev(severity) * 0.035)))
    repeats = int(2 + sev(severity) * 6)
    result = region.copy()
    if len(result) > slice_len:
        start = np.random.randint(0, len(result) - slice_len)
        glitch = region[start:start + slice_len]
        end = min(start + slice_len * repeats, len(result))
        needed = end - start
        tiled = np.tile(glitch, repeats + 1)[:needed]
        result[start:end] = tiled
    return result


def _granular_fragmentation(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    grain_len = max(int(sr * 0.008), int(sr * (0.025 - sev(severity) * 0.017)))
    if len(region) < grain_len * 2:
        return region
    n_grains = len(region) // grain_len
    grains = [region[i * grain_len:(i + 1) * grain_len].copy() for i in range(n_grains)]
    remainder = region[n_grains * grain_len:]
    n_shuffle = max(2, int(n_grains * sev(severity)))
    indices = np.random.choice(n_grains, size=min(n_shuffle, n_grains), replace=False)
    vals = [grains[i].copy() for i in indices]
    np.random.shuffle(vals)
    for idx, val in zip(indices, vals):
        grains[idx] = val
    return np.concatenate(grains + [remainder])


def _vinyl_crackle(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    result = region.copy()
    n_crackles = int(5 + sev(severity) * 25)
    for _ in range(n_crackles):
        pos = np.random.randint(0, len(result))
        width = np.random.randint(1, 15)
        end = min(pos + width, len(result))
        result[pos:end] += float(np.random.choice([-1, 1])) * np.random.uniform(0.25, 0.9)
    result = np.clip(result, -1.0, 1.0)
    hiss = np.random.normal(0.0, sev(severity) * 0.04, len(result)).astype(np.float32)
    return np.clip(result + hiss, -1.0, 1.0)


def _quantization_noise(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    bits = max(5, int(16 - sev(severity) * 11))
    steps = float(2 ** bits)
    quantized = np.round(region * steps / 2.0) / (steps / 2.0)
    noise_scale = (1.0 / steps) * sev(severity) * 0.6
    noise = np.random.uniform(-noise_scale, noise_scale, len(region)).astype(np.float32)
    return np.clip(quantized + noise, -1.0, 1.0)


def _mp3_compression(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    cutoff_hz = max(3000, int(8000 - sev(severity) * 5000))
    b, a = butter(4, cutoff_hz / (sr / 2.0), btype="low")
    filtered = filtfilt(b, a, region.astype(np.float64)).astype(np.float32)
    delay = int(sr * 0.002)
    if delay > 0 and delay < len(filtered):
        pre = np.zeros_like(filtered)
        pre[:-delay] = filtered[delay:] * sev(severity) * 0.12
        filtered = np.clip(filtered + pre, -1.0, 1.0)
    return filtered


def _sample_rate_mismatch(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    factor = max(2, int(2 + sev(severity) * 7))
    downsampled = region[::factor]
    upsampled = np.repeat(downsampled, factor)[:len(region)]
    if len(upsampled) < len(region):
        upsampled = np.pad(upsampled, (0, len(region) - len(upsampled)))
    return upsampled.astype(np.float32)


def _aliasing(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    return _sample_rate_mismatch(region, sr, severity)


def _buffer_underrun(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    chunk_len = int(sr * 0.020)
    skip_every = max(2, int(5 - sev(severity) * 3))
    result = region.copy()
    n_chunks = len(result) // chunk_len
    for i in range(n_chunks):
        if i % skip_every == 0:
            s = i * chunk_len
            result[s:s + chunk_len] = 0.0
    return result


def _bit_errors(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    n_errors = max(3, int(sev(severity) * 60))
    result = region.copy()
    for _ in range(n_errors):
        pos = np.random.randint(0, len(result))
        result[pos] = float(np.random.choice([-1.0, 1.0]))
    return result


def _echo_feedback(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    delay = int(sr * (0.08 + sev(severity) * 0.22))
    feedback = 0.45 + sev(severity) * 0.40
    result = region.astype(np.float64).copy()
    for i in range(delay, len(result)):
        result[i] += result[i - delay] * feedback
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


def _resonant_comb_filter(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    delay = max(1, int(sr * (0.004 + sev(severity) * 0.012)))
    wet = 0.5 + sev(severity) * 0.4
    result = region.astype(np.float64).copy()
    if delay < len(result):
        result[delay:] += result[:-delay] * wet
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


def _feedback_squeal(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    freq = 2000.0 + sev(severity) * 4500.0
    t = np.arange(len(region), dtype=np.float64) / sr
    envelope = np.linspace(0.0, sev(severity) * 0.35, len(region))
    squeal = np.sin(2 * np.pi * freq * t) * envelope
    return np.clip(region.astype(np.float64) + squeal, -1.0, 1.0).astype(np.float32)


def _fft_smearing(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    fft = np.fft.rfft(region.astype(np.float64))
    mag = np.abs(fft)
    kernel_size = max(3, int(sev(severity) * 45))
    if kernel_size < len(mag):
        kernel = np.ones(kernel_size) / kernel_size
        mag = np.convolve(mag, kernel, mode="same")
    phase_noise = np.random.normal(0.0, sev(severity) * 0.9, len(fft))
    fft = mag * np.exp(1j * (np.angle(fft) + phase_noise))
    return np.clip(np.fft.irfft(fft, n=len(region)).astype(np.float32), -1.0, 1.0)


def _phase_distortion(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    n_stages = 1 + int(sev(severity) * 3)
    result = region.astype(np.float64)
    original_peak = np.max(np.abs(result))
    for stage in range(n_stages):
        cutoff = min((0.15 + stage * 0.12), 0.95)
        b, a = butter(3, cutoff)
        result = lfilter(b, a, result)
    peak = np.max(np.abs(result))
    if peak > 1e-8 and original_peak > 1e-8:
        result = result / peak * original_peak
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def _time_stretch_artifact(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    result = region.astype(np.float64).copy()
    n_echoes = 2 + int(sev(severity) * 3)
    for i in range(1, n_echoes + 1):
        shift = int(sr * 0.001 * i)
        if shift >= len(region):
            break
        wet = sev(severity) * 0.18 / i
        shifted = np.zeros_like(result)
        shifted[shift:] = region[:-shift].astype(np.float64) if shift > 0 else region.astype(np.float64)
        result += shifted * wet
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


def _pitch_correction_artifact(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    block = int(sr * 0.04)
    out_chunks = []
    for i in range(0, len(region), block):
        chunk = region[i:i + block]
        if len(chunk) < 2:
            out_chunks.append(chunk)
            continue
        step = int(sev(severity) * 4) + 1
        shift = int(np.random.choice([-step, 0, step])) * 0.005
        new_len = max(1, int(len(chunk) * (1.0 + shift)))
        stretched = np.interp(
            np.linspace(0, len(chunk) - 1, new_len),
            np.arange(len(chunk)),
            chunk,
        )
        out_chunks.append(stretched[:len(chunk)])
    result = np.concatenate(out_chunks)[:len(region)]
    if len(result) < len(region):
        result = np.pad(result, (0, len(region) - len(result)))
    return result.astype(np.float32)


def _phase_cancellation(region: np.ndarray, sr: int, severity: str) -> np.ndarray:
    delay = max(1, int(sev(severity) * 12))
    offset = np.zeros_like(region, dtype=np.float64)
    if delay < len(region):
        offset[delay:] = region[:-delay].astype(np.float64)
    result = region.astype(np.float64) - offset * (0.45 + sev(severity) * 0.45)
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARTIFACT_REGISTRY: dict = {
    "mp3_compression":          _mp3_compression,
    "quantization_noise":       _quantization_noise,
    "hard_clipping":            _hard_clipping,
    "soft_clipping":            _soft_clipping,
    "bit_crushing":             _bit_crushing,
    "buffer_dropout":           _buffer_dropout,
    "stutter_repeat":           _stutter_repeat,
    "granular_fragmentation":   _granular_fragmentation,
    "white_noise":              _white_noise,
    "pink_noise":               _pink_noise,
    "vinyl_crackle":            _vinyl_crackle,
    "em_hum_50hz":              _em_hum_50hz,
    "em_hum_60hz":              _em_hum_60hz,
    "time_stretch_artifact":    _time_stretch_artifact,
    "pitch_correction_artifact":_pitch_correction_artifact,
    "phase_cancellation":       _phase_cancellation,
    "bit_errors":               _bit_errors,
    "sample_rate_mismatch":     _sample_rate_mismatch,
    "buffer_underrun":          _buffer_underrun,
    "echo_feedback":            _echo_feedback,
    "resonant_comb_filter":     _resonant_comb_filter,
    "feedback_squeal":          _feedback_squeal,
    "fft_smearing":             _fft_smearing,
    "phase_distortion":         _phase_distortion,
    "aliasing":                 _aliasing,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_artifacts_to_wav(
    clean_wav: Path,
    output_wav: Path,
    artifacts: list[dict],
    sentence: str = "",
    target_word: str = "",
) -> None:
    """
    Read *clean_wav*, apply each artifact recipe in sequence, write *output_wav*.

    Each artifact dict must have: type (str), location (str), severity (str).
    """
    audio, sr, n_channels = read_wav(clean_wav)

    mono = audio.mean(axis=1) if audio.ndim > 1 else audio.copy()

    for artifact in artifacts:
        fn = ARTIFACT_REGISTRY.get(artifact.get("type", ""))
        if fn is None:
            continue
        start, end = _get_region(mono, artifact.get("location", "throughout"), sr, sentence, target_word)
        if start >= end:
            continue
        region = mono[start:end].copy()
        processed = fn(region, sr, artifact.get("severity", "moderate"))
        mono[start:end] = processed[:end - start]

    if audio.ndim > 1:
        result = np.stack([mono] * n_channels, axis=1)
    else:
        result = mono

    write_wav(output_wav, result, sr, n_channels)
