"""
Shared test doubles for the TTS/OpenAI/ElevenLabs client boundaries.

Several pipelines read the synthesized audio back off disk (duration checks,
PSOLA, DSP artifacts, Praat manipulation) — a no-op mock would crash those.
These helpers make the mocked clients write/return a real, tiny, synthetic
WAV so that downstream code runs for real. The audio content itself (a plain
sine tone) is a placeholder, not a correctness target — these tests validate
orchestration (which scores get generated/delivered/skipped), not audio
signal quality.
"""
from __future__ import annotations

import io
import json
import math
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


def _tiny_wav_bytes(duration_s: float = 1.0, freq: float = 220.0, sample_rate: int = 44100) -> bytes:
    """A real, minimal mono 16-bit PCM WAV as bytes (stdlib `wave`, no extra deps)."""
    n_samples = max(1, int(duration_s * sample_rate))
    amplitude = 3000
    frames = b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(n_samples)
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return buf.getvalue()


def write_tiny_wav(path: Path, duration_s: float = 1.0, freq: float = 220.0, sample_rate: int = 44100) -> None:
    """Write a real, minimal mono 16-bit WAV to *path*."""
    Path(path).write_bytes(_tiny_wav_bytes(duration_s, freq, sample_rate))


def mock_cartesia_client(duration_s: float = 1.0):
    """Mock AsyncCartesia whose tts.generate(...).write_to_file(path) writes a real tiny WAV.

    Code that calls get_wav_duration()/loads the file with parselmouth/etc.
    afterward gets a real file to work with, regardless of what `speed` or
    `generation_config` was requested (this mock ignores those — it's testing
    orchestration, not that a `speed` param actually changes tempo).
    """
    async def _write(path_str):
        write_tiny_wav(Path(path_str), duration_s=duration_s)

    response = MagicMock()
    response.write_to_file = AsyncMock(side_effect=_write)
    client = MagicMock()
    client.tts.generate = AsyncMock(return_value=response)
    return client


def mock_elevenlabs_client(duration_s: float = 1.0):
    """Mock AsyncElevenLabs whose text_to_speech.convert(...) yields real tiny WAV bytes."""
    wav_bytes = _tiny_wav_bytes(duration_s)

    async def _chunks(*_args, **_kwargs):
        chunk_size = 4096
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i:i + chunk_size]

    client = MagicMock()
    client.text_to_speech.convert = MagicMock(side_effect=_chunks)
    return client


def mock_openai_client(payload: dict):
    """Mock AsyncOpenAI/AsyncAzureOpenAI whose chat.completions.create(...) returns *payload* as JSON."""
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client
