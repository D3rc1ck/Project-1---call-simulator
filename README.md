# 📞 Call Simulator

Practice phone calls entirely on your own machine. The simulator plays the
person on the other end of the line (an angry customer, a recruiter, a
confused tech-support caller) — you talk, it talks back.

- **Speech synthesis:** [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) (82M-parameter local model, 11 voices)
- **Speech recognition:** [Whisper](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
- **Backend:** FastAPI, served locally with uvicorn
- **Frontend:** vanilla HTML/JS single page (mic recording + audio playback)

Everything runs locally — no cloud APIs, no keys. Model weights are
downloaded from Hugging Face on first use and cached afterwards.

## How a call works

```
🎤 your voice ──► Whisper (STT) ──► scenario engine ──► Kokoro (TTS) ──► 🔊 reply
```

Each scenario is a small offline state machine: keyword rules react to what
you say (e.g. offering a *refund* to the billing customer), otherwise the
call advances through its script until the caller wraps up and hangs up.
You can also type replies instead of speaking.

## Requirements

- Python 3.10 – 3.12
- `espeak-ng` (Kokoro's fallback phonemizer)
  - Debian/Ubuntu: `sudo apt-get install espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
- ~3 GB disk for PyTorch + model weights

## Local deployment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**, pick a scenario, and answer the call.
The first turn is slow while Kokoro and Whisper load; after that it's fast.

> Browsers only expose the microphone on `localhost` or HTTPS — access the
> app via `127.0.0.1`/`localhost`, not a bare LAN IP, or use the typed-reply
> box instead.

### Docker

```bash
docker build -t call-simulator .
docker run -p 8000:8000 call-simulator
```

### Configuration

| Env var | Default | Options |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, `auto` |
| `WHISPER_COMPUTE` | `int8` | `int8`, `int8_float16`, `float16`, `float32` |

Example with a bigger model on GPU:

```bash
WHISPER_MODEL=small WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 \
  uvicorn backend.main:app --port 8000
```

## API

Interactive docs at **http://127.0.0.1:8000/docs**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/voices` | Available Kokoro voices |
| `GET` | `/api/scenarios` | Available call scenarios |
| `POST` | `/api/tts` | `{text, voice, speed}` → WAV audio |
| `POST` | `/api/stt` | multipart audio file → `{text, language, duration}` |
| `POST` | `/api/call/start` | `{scenario_id, voice?}` → greeting text + audio, session id |
| `POST` | `/api/call/turn` | multipart `session_id` + audio → transcript, reply text + audio |
| `POST` | `/api/call/text_turn` | `{session_id, text}` → reply text + audio |
| `POST` | `/api/call/end` | `{session_id}` → hang up |

Standalone examples:

```bash
# Text to speech
curl -s http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from Kokoro!", "voice": "af_heart"}' \
  -o hello.wav

# Speech to text
curl -s http://127.0.0.1:8000/api/stt -F audio=@hello.wav
```

## Project layout

```
backend/
  main.py        FastAPI app + call session endpoints
  tts.py         Kokoro synthesis (lazy-loaded pipelines)
  stt.py         faster-whisper transcription (lazy-loaded model)
  scenarios.py   Call scenarios + session engine
frontend/
  index.html     Single-page UI
  app.js         Mic recording, turn handling, audio playback
  style.css
```
