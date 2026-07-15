"""Speech-to-text powered by Whisper (via faster-whisper / CTranslate2).

The model is loaded lazily on first request. Size, device and compute type
are configurable through environment variables:

    WHISPER_MODEL    tiny | base | small | medium | large-v3   (default: base)
    WHISPER_DEVICE   cpu | cuda | auto                          (default: cpu)
    WHISPER_COMPUTE  int8 | int8_float16 | float16 | float32    (default: int8)
"""

import io
import logging
import os
import threading

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            size = os.environ.get("WHISPER_MODEL", "base")
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            compute = os.environ.get("WHISPER_COMPUTE", "int8")
            logger.info(
                "Loading Whisper model %s (device=%s, compute=%s) ...",
                size, device, compute,
            )
            _model = WhisperModel(size, device=device, compute_type=compute)
            logger.info("Whisper model ready")
        return _model


def transcribe(data: bytes, language: str | None = None) -> dict:
    """Transcribe an audio file (wav/webm/ogg/mp3/...) given as raw bytes."""
    if not data:
        raise ValueError("audio payload is empty")

    model = _get_model()
    # A single model instance is not safe for concurrent transcriptions on CPU
    # with shared workers; serialize requests (fine for local, single-user use).
    with _lock:
        segments, info = model.transcribe(
            io.BytesIO(data),
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

    return {
        "text": text,
        "language": info.language,
        "duration": round(info.duration, 2),
    }
