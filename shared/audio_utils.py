"""
Audio measurement and post-processing utilities.

Provides duration measurement, WPM computation, and Praat PSOLA
time-stretching via parselmouth.
"""
from __future__ import annotations

from pathlib import Path

import parselmouth
from parselmouth.praat import call

# Minimum duration (seconds) below which audio is considered invalid.
_MIN_DURATION = 0.1


def get_wav_duration(path: Path) -> float:
    """Return the duration of a WAV file in seconds.

    Uses parselmouth (Praat) rather than the stdlib ``wave`` module
    because Cartesia WAV files carry a non-standard data-chunk size
    that causes ``wave.getnframes()`` to return INT32_MAX.
    """
    snd = parselmouth.Sound(str(path))
    return snd.duration


def measure_wpm(path: Path, word_count: int) -> float:
    """Compute words-per-minute from a WAV file and a known word count."""
    duration = get_wav_duration(path)
    if duration < _MIN_DURATION:
        return 0.0
    return word_count / (duration / 60.0)


def psola_stretch(input_path: Path, output_path: Path, factor: float) -> None:
    """Time-stretch a WAV file using Praat PSOLA (overlap-add).

    Args:
        input_path:  Source WAV.
        output_path: Destination WAV (may be the same as *input_path*).
        factor:      Duration multiplier.  Values > 1.0 make the audio
                     *longer* (slower); values < 1.0 make it *shorter* (faster).
                     Pitch is preserved.
    """
    snd = parselmouth.Sound(str(input_path))
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    duration_tier = call(
        manipulation, "Create DurationTier", "stretch", snd.xmin, snd.xmax,
    )
    midpoint = (snd.xmin + snd.xmax) / 2
    call(duration_tier, "Add point", midpoint, factor)
    call([manipulation, duration_tier], "Replace duration tier")
    result = call(manipulation, "Get resynthesis (overlap-add)")
    result.save(str(output_path), "WAV")


def adjust_to_target_ratio(
    wav_path: Path,
    baseline_duration: float,
    *,
    target_ratio_min: float | None = None,
    target_ratio_max: float | None = None,
) -> tuple[float, bool]:
    """Measure the duration ratio and apply PSOLA if outside the target range.

    The *ratio* is ``version_duration / baseline_duration``:
      - ratio < 1 → faster than normal
      - ratio > 1 → slower than normal

    Args:
        wav_path:          Path to the WAV to (potentially) correct.
        baseline_duration: Duration (seconds) of the normal-speed version.
        target_ratio_min:  Minimum acceptable ratio (floor). If the actual
                           ratio falls below this, PSOLA stretches the audio.
        target_ratio_max:  Maximum acceptable ratio (ceiling). If the actual
                           ratio exceeds this, PSOLA compresses the audio.

    Returns:
        (final_ratio, psola_applied)
    """
    duration = get_wav_duration(wav_path)
    if duration < _MIN_DURATION or baseline_duration < _MIN_DURATION:
        return 0.0, False

    ratio = duration / baseline_duration

    # Determine target ratio to correct toward
    target: float | None = None
    if target_ratio_max is not None and ratio > target_ratio_max:
        target = target_ratio_max
    elif target_ratio_min is not None and ratio < target_ratio_min:
        target = target_ratio_min

    if target is None:
        return ratio, False

    # PSOLA factor: target_ratio / current_ratio
    #   > 1 → stretch (make longer/slower)
    #   < 1 → compress (make shorter/faster)
    psola_factor = target / ratio
    psola_stretch(wav_path, wav_path, psola_factor)

    final_duration = get_wav_duration(wav_path)
    final_ratio = final_duration / baseline_duration
    return final_ratio, True
