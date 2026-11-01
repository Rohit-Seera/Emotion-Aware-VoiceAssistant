# Emotion Aware Voice Assistant — Professor Demonstration Sheet

## Research extension built on SOVA-Bench

### SOVA-Bench connection
SOVA-Bench treats emotion as a paralinguistic property of spoken input and evaluates speech understanding/generation. Its ideal-assistant example connects a user's emotional state with an appropriate response tone.

### What Emotion Aware Voice Assistant adds in this prototype

**1. Advanced emotion-aware interaction**

`Microphone audio -> ASR + acoustic SER -> hybrid emotion -> emotion-aware prompt -> expression plan -> emotion-conditioned TTS`

- Acoustic emotion recognition uses the public `superb/wav2vec2-base-superb-er` checkpoint.
- Transcript/keyword cues are fused with the acoustic prediction.
- The detected emotion is included in the LLM response policy.
- Companion expression/gesture/gaze is adapted to the emotional context.
- ElevenLabs receives per-response voice settings for warm / empathetic / excited delivery.
- The system exposes emotion confidence, source, and top predictions for inspection.

**2. Barge-in / interruptible conversation**

`Assistant speaks -> browser VAD detects user speech -> stop playback -> cancel active backend task -> capture new turn -> STT -> emotion -> LLM -> TTS`

- User can interrupt while the assistant is speaking.
- Playback is stopped immediately by the browser.
- The active server-side turn is cancelled.
- The interrupting utterance is processed as a new turn.
- Session interruption count is tracked.
- The demo measures local VAD-to-playback-stop latency.

## What is genuinely implemented vs. planned

**Implemented now:** acoustic SER integration, hybrid emotion fusion, emotion-aware prompt policy, expression/voice-style conditioning, interruptible streaming flow, cancellation, interruption metrics, research demo dashboard and APIs.

**Not claimed:** a Weings-trained emotion model or a full custom end-to-end duplex speech model. The acoustic SER model is a public pretrained component integrated into the Weings pipeline.

## 2-minute demo script

1. Start the backend and open `/api/v1/demo`.
2. Click **Warm-up Acoustic SER** and show `ready` in the status panel.
3. Start the companion and say a clearly emotional sentence.
4. Show `Detected emotion`, confidence, source, top predictions, response style and voice style.
5. Let the assistant begin speaking.
6. Interrupt it mid-response.
7. Show the playback stop, cancellation event, new-turn processing and interruption count.
8. Click **Refresh Metrics** and show the session metrics.

## One-line research statement

> We extend an SOVA-Bench-inspired voice pipeline with acoustic emotion-aware response adaptation and interruptible barge-in handling, then expose measurable interaction signals for future controlled evaluation.
