# Emotion Aware Voice Assistant Voice Companion Backend

This package contains the backend Voice Companion slice plus a SOVA-Bench-inspired research extension for **emotion-aware speech interaction** and **interruptible/barged-in conversation**.

## Implemented

- Authenticated caller-owned voice sessions.
- Transcript runtime processing through the existing companion graph.
- Groq Whisper STT adapter.
- ElevenLabs TTS adapter with per-response emotion-conditioned voice settings.
- Optional Wav2Vec2 acoustic speech-emotion recognition.
- Hybrid acoustic + transcript emotion fusion.
- Emotion-aware response policy in the LLM prompt.
- Constrained companion expression/gesture/gaze directives.
- Streaming TTS with backend cancellation for barge-in.
- Browser VAD that stops assistant playback when the user interrupts.
- Research/demo endpoints for emotion-model status/warm-up and session interruption metrics.
- Local research demo at `/api/v1/demo`.

## Research extension

The supplied SOVA-Bench paper treats emotion as a paralinguistic speech-understanding dimension and evaluates speech generation using semantic and acoustic perspectives. This project extends the existing voice-companion slice by integrating an acoustic SER path, fusing emotion with transcript cues, conditioning the response strategy and TTS delivery, and measuring interruption behavior.

Read the full design and scientific boundary in `docs/RESEARCH_ADDITION.md`.

## Local run

```powershell
Copy-Item .env.example .env
# Configure the existing Groq, ElevenLabs, and JWT values in .env.
uv sync --extra dev
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or use:

```powershell
./start_demo.ps1
```

Open:

```text
http://127.0.0.1:8000/api/v1/demo
```

For the professor-facing flow, see `DEMO_GUIDE.md`.

## Acoustic emotion model

The default checkpoint is:

```text
superb/wav2vec2-base-superb-er
```

It is loaded lazily. Use **Warm-up Acoustic SER** in the demo before presenting the emotion panel. The model expects 16 kHz speech. If it cannot be loaded, the API explicitly reports the fallback source instead of presenting a text prediction as an acoustic prediction.

## Main voice API

- `POST /api/v1/voice/sessions`
- `POST /api/v1/voice/sessions/{session_id}/turns`
- `POST /api/v1/voice/sessions/{session_id}/audio-turn`
- `POST /api/v1/voice/sessions/{session_id}/audio-stream-turn`
- `POST /api/v1/voice/sessions/{session_id}/cancel`
- `GET /api/v1/voice/sessions/{session_id}/metrics`
- `GET /api/v1/voice/emotion/status`
- `POST /api/v1/voice/emotion/warmup`
- `POST /api/v1/voice/transcriptions`
- `POST /api/v1/voice/synthesis`
- `POST /api/v1/voice/synthesis/stream`

## Security / privacy

- Voice endpoints require a valid bearer JWT and the existing voice permission.
- Session ownership is enforced server side.
- Raw audio is bounded and processed in memory.
- The demo is development-only.
- `.env` is local configuration and should not be committed.

## Validation

The source tree is syntax-checked with Python `compileall`, and the browser demo JavaScript is checked with Node's parser.

Full pytest execution in the current sandbox is blocked by missing project dependencies in the sandbox image (for example the Anthropic SDK). The project dependency manifests have been updated for the research extension; run `uv sync --extra dev` locally before the full suite.
