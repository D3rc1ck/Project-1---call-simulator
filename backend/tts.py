"""Text-to-speech powered by Kokoro (hexgrad/Kokoro-82M).

Pipelines are created lazily (one per language code) so the server starts
instantly and model weights are only downloaded on first use.

For air-gapped / offline deployments the weights can be supplied locally
instead of downloading from Hugging Face:

    KOKORO_CONFIG      path to config.json
    KOKORO_MODEL       path to kokoro-v1_0.pth
    KOKORO_VOICES_DIR  directory containing <voice>.pt files
"""

import io
import logging
import os
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
_model = None
_lock = threading.Lock()


def _get_local_model():
    """Build a KModel from local files when KOKORO_CONFIG/KOKORO_MODEL are set
    (offline deployments); otherwise return None to download from HF."""
    global _model
    config = os.environ.get("KOKORO_CONFIG")
    model = os.environ.get("KOKORO_MODEL")
    if not (config and model):
        return None
    if _model is None:
        logger.info("Loading local Kokoro model from %s", model)
        from kokoro import KModel

        _model = KModel(
            repo_id="hexgrad/Kokoro-82M", config=config, model=model
        ).eval()
    return _model


def _resolve_voice(voice: str) -> str:
    """Map a voice id to a local .pt path when KOKORO_VOICES_DIR is set."""
    voices_dir = os.environ.get("KOKORO_VOICES_DIR")
    if voices_dir:
        return os.path.join(voices_dir, f"{voice}.pt")
    return voice


def _get_pipeline(lang_code: str):
    with _lock:
        if lang_code not in _pipelines:
            logger.info("Loading Kokoro pipeline for lang_code=%s ...", lang_code)
            from kokoro import KPipeline

            local_model = _get_local_model()
            _pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
                model=local_model if local_model is not None else True,
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
        for _graphemes, _phonemes, audio in pipeline(
            text, voice=_resolve_voice(voice), speed=speed
        ):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio")

    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
