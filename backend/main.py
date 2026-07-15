"""Call Simulator — FastAPI backend for local deployment.

Speech synthesis: Kokoro TTS (hexgrad/Kokoro-82M)
Speech recognition: Whisper (faster-whisper)

Run locally with:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import base64
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import stt, tts
from .scenarios import FALLBACK_LINE, SCENARIOS, SessionStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Call Simulator API",
    description="Practice phone calls locally: Kokoro TTS + Whisper STT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = SessionStore()

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str = tts.DEFAULT_VOICE
    speed: float = Field(1.0, ge=0.5, le=2.0)


class StartCallRequest(BaseModel):
    scenario_id: str
    voice: str | None = None


class TextTurnRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, max_length=2000)


class EndCallRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/voices")
def voices() -> list[dict]:
    return tts.list_voices()


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [
        {
            "id": s.id,
            "title": s.title,
            "description": s.description,
            "role": s.role,
            "voice": s.voice,
        }
        for s in SCENARIOS.values()
    ]


# ---------------------------------------------------------------------------
# Standalone speech endpoints
# ---------------------------------------------------------------------------

@app.post("/api/tts")
def text_to_speech(req: TTSRequest) -> Response:
    """Synthesize speech with Kokoro; returns a WAV file."""
    try:
        wav = tts.synthesize(req.text, voice=req.voice, speed=req.speed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=wav, media_type="audio/wav")


@app.post("/api/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
) -> dict:
    """Transcribe an uploaded audio file (wav/webm/ogg/mp3/...) with Whisper."""
    data = await audio.read()
    try:
        return stt.transcribe(data, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Call simulation
# ---------------------------------------------------------------------------

def _speak(session, text: str) -> str:
    """Synthesize `text` with the session's voice, return base64 WAV."""
    wav = tts.synthesize(text, voice=session.voice)
    return base64.b64encode(wav).decode("ascii")


@app.post("/api/call/start")
def start_call(req: StartCallRequest) -> dict:
    try:
        session = sessions.create(req.scenario_id, voice=req.voice)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown scenario")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    greeting = session.scenario.greeting
    return {
        "session_id": session.id,
        "scenario": session.scenario.title,
        "role": session.scenario.role,
        "reply_text": greeting,
        "audio_b64": _speak(session, greeting),
        "ended": False,
    }


def _run_turn(session, user_text: str) -> dict:
    if not user_text.strip():
        return {
            "transcript": "",
            "reply_text": FALLBACK_LINE,
            "audio_b64": _speak(session, FALLBACK_LINE),
            "ended": False,
        }
    reply, ended = session.next_reply(user_text)
    return {
        "transcript": user_text,
        "reply_text": reply,
        "audio_b64": _speak(session, reply),
        "ended": ended,
    }


@app.post("/api/call/turn")
async def call_turn(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    """One spoken turn: Whisper transcribes the user, the scenario replies,
    Kokoro speaks the answer."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")

    data = await audio.read()
    try:
        result = stt.transcribe(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = _run_turn(session, result["text"])
    if response["ended"]:
        sessions.drop(session_id)
    return response


@app.post("/api/call/text_turn")
def call_text_turn(req: TextTurnRequest) -> dict:
    """Same as /api/call/turn but with typed text instead of audio."""
    session = sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")

    response = _run_turn(session, req.text)
    if response["ended"]:
        sessions.drop(req.session_id)
    return response


@app.post("/api/call/end")
def end_call(req: EndCallRequest) -> dict:
    sessions.drop(req.session_id)
    return {"status": "ended"}


# Serve the frontend (mounted last so /api/* wins).
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
