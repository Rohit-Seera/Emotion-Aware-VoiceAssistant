# Emotion Aware Voice Assistant Aiko — 2-Minute Research Demo

## Before the demo

Use the existing `.env` for your local provider keys. Do not commit or share `.env`.

Install the research dependencies:

```powershell
uv sync --extra dev
```

Start:

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or:

```powershell
./start_demo.ps1
```

Open:

`http://127.0.0.1:8000/api/v1/demo`

## Demo flow

### A. Show acoustic emotion recognition

Click **Warm-up Acoustic SER**.

The right panel should show one of:

- `ready` — Wav2Vec2 acoustic model is loaded;
- `lazy` — model is configured but has not been loaded;
- `load-failed` — the optional model could not be loaded, so the system will expose a text fallback rather than pretending an acoustic result exists.

Then click play and say a clearly emotional sentence.

The panel will show:

`Detected emotion` → `Confidence` → `Source` → top acoustic predictions.

### B. Show emotion-aware response

Watch how the response plan changes:

`User emotion` → `Response style` → `Voice style` → `Expression/Gesture/Gaze`.

For sad/anxious/angry/frustrated input, the LLM prompt contains a response policy that asks for validating, calm, context-appropriate language.

### C. Show barge-in

Wait until Aiko is speaking.

Start speaking over her.

The demo should:

1. detect speech with browser VAD;
2. stop local playback;
3. call `/cancel` on the active session;
4. capture the interrupting utterance;
5. start a new STT → emotion → LLM → TTS turn.

The **Barge-in lab** panel shows the interruption count and the browser-measured latency from first detected voice activity to playback stop.

## What to say to the professor

> "We extended the SOVA-Bench-inspired voice pipeline in two directions. First, we added an acoustic speech-emotion path using Wav2Vec2 and fused it with transcript cues, so the detected user emotion changes the response policy, companion expression, and per-response TTS delivery. Second, we made the response interruptible: the user can barge in while the assistant is speaking, the frontend stops playback, the backend task is cancelled, and the new user turn is processed immediately."

Then show the metrics panel and the live event log.
