"""
TTS helpers for Cartesia and ElevenLabs.

Both functions share the same signature shape: client, text, output_path,
plus engine-specific keyword args.
"""
from __future__ import annotations

from pathlib import Path

from cartesia import AsyncCartesia
from elevenlabs.client import AsyncElevenLabs

# Default output format for Cartesia pipelines
DEFAULT_OUTPUT_FORMAT: dict = {
    "container": "wav",
    "encoding": "pcm_s16le",
    "sample_rate": 44100,
}

# ElevenLabs model used by the intonation pipeline
ELEVENLABS_MODEL = "eleven_multilingual_v2"


async def generate_tts(
    client: AsyncCartesia,
    transcript: str,
    output_path: Path,
    *,
    model_id: str,
    voice_id: str,
    output_format: dict = DEFAULT_OUTPUT_FORMAT,
    speed: float | None = None,
    generation_config: dict | None = None,
) -> None:
    """Synthesize *transcript* with Cartesia and write WAV to *output_path*.

    *generation_config* is forwarded to Cartesia's generation_config parameter
    (e.g. {"emotion": "confident", "speed": 0.8}). When both *speed* and
    *generation_config* are provided, *generation_config* takes precedence.
    """
    kwargs: dict = dict(
        model_id=model_id,
        transcript=transcript,
        voice={"mode": "id", "id": voice_id},
        output_format=output_format,
    )
    gen_cfg = dict(generation_config or {})
    if speed is not None and "speed" not in gen_cfg:
        gen_cfg["speed"] = speed
    if gen_cfg:
        kwargs["generation_config"] = gen_cfg
    response = await client.tts.generate(**kwargs)
    await response.write_to_file(str(output_path))


async def generate_tts_elevenlabs(
    client: AsyncElevenLabs,
    transcript: str,
    output_path: Path,
    *,
    voice_id: str,
    model_id: str = ELEVENLABS_MODEL,
) -> None:
    """Synthesize *transcript* with ElevenLabs and write WAV to *output_path*."""
    with open(output_path, "wb") as f:
        async for chunk in client.text_to_speech.convert(
            voice_id=voice_id,
            text=transcript,
            model_id=model_id,
            output_format="wav_44100",
        ):
            f.write(chunk)
