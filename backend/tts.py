"""Text-to-speech powered by Kokoro (hexgrad/Kokoro-82M).

Pipelines are created lazily (one per language code) so the server starts
instantly and model weights are only downloaded on first use.
"""

import io
import logging
import threading

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# Voice ids follow Kokoro's convention: first letter = language
# ("a" = American English, "b" = British English), second = gender.
VOICES = [
    {"id": "af_heart", "name": "Heart — US English, female"},
    {"id": "af_bella", "name": "Bella — US English, female"},
    {"id": "af_nicole", "name": "Nicole — US English, female"},
    {"id": "af_sarah", "name": "Sarah — US English, female"},
    {"id": "am_adam", "name": "Adam — US English, male"},
    {"id": "am_michael", "name": "Michael — US English, male"},
    {"id": "am_puck", "name": "Puck — US English, male"},
    {"id": "bf_emma", "name": "Emma — UK English, female"},
    {"id": "bf_isabella", "name": "Isabella — UK English, female"},
    {"id": "bm_george", "name": "George — UK English, male"},
    {"id": "bm_lewis", "name": "Lewis — UK English, male"},
]

VOICE_IDS = {v["id"] for v in VOICES}
DEFAULT_VOICE = "af_heart"

_pipelines: dict[str, object] = {}
_lock = threading.Lock()


def _get_pipeline(lang_code: str):
    with _lock:
        if lang_code not in _pipelines:
            logger.info("Loading Kokoro pipeline for lang_code=%s ...", lang_code)
            from kokoro import KPipeline

            _pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id="hexgrad/Kokoro-82M"
            )
            logger.info("Kokoro pipeline ready (lang_code=%s)", lang_code)
        return _pipelines[lang_code]


def list_voices() -> list[dict]:
    return VOICES


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Synthesize `text` and return a 16-bit PCM WAV file as bytes."""
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if voice not in VOICE_IDS:
        raise ValueError(f"unknown voice {voice!r}")

    pipeline = _get_pipeline(voice[0])

    chunks: list[np.ndarray] = []
    # Kokoro generation isn't guaranteed thread-safe; serialize it.
    with _lock:
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio")

    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
