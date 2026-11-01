# Emotion Aware Voice Assistant — Research Addition for SOVA-Bench-Inspired Voice Interaction

## What was added

This implementation extends the existing Emotion Aware Voice Assistant Voice Companion in two research-facing directions:

### 1. Emotion-aware interaction

The paper **SOVA-Bench: Benchmarking the Speech Conversation Ability for LLM-based Voice Assistant** treats emotion as a paralinguistic speech-understanding dimension and evaluates speech generation from semantic and acoustic perspectives. Its ideal-assistant example connects a user's emotional state to an appropriate response tone. The paper's paralinguistic tasks include emotion, gender, and age recognition, and its generation tasks include consistency, semantic quality, and acoustic quality.

Weings now uses a **hybrid emotion path**:

```text
User audio
   ├──> Groq Whisper STT ──────────────> transcript
   └──> Wav2Vec2 acoustic SER ─────────> acoustic emotion
                       \                 /
                        \               /
                         +-- lexical --+
                                |
                                v
                         fused user emotion
                                |
                                v
                       emotion-aware prompt
                                |
                                v
                               LLM
                                |
                                v
                     response expression plan
                                |
                                v
                    emotion-conditioned TTS
```

The default acoustic checkpoint is `superb/wav2vec2-base-superb-er`. It is loaded lazily, expects 16 kHz speech, and is used through the Transformers audio-classification pipeline.

If the optional acoustic stack is unavailable, the system does **not** claim an acoustic prediction: it falls back to the existing transcript/keyword detector and exposes that source in the API and demo UI.

### 2. Barge-in interaction

The existing streaming voice route is retained and strengthened as a research demo:

```text
Aiko speaking
     |
     +--> browser VAD detects user speech
              |
              +--> stop local playback
              +--> cancel active backend task
              +--> capture interruption audio
              +--> new STT/LLM/TTS turn
```

The demo measures the local VAD-to-playback-stop latency and the backend stores a per-session interruption count.

## Emotion-conditioned speech

The ElevenLabs adapter now accepts a per-response `emotion`/delivery style and converts it to request-level voice settings. The stored voice itself is not changed. The research demo currently uses these delivery styles:

| Assistant style | Intent | Voice controls |
|---|---|---|
| `warm` | neutral conversational delivery | balanced stability/style/speed |
| `empathetic` | supportive response to sad/anxious/angry/frustrated input | lower stability, slightly slower pace, higher style |
| `excited` | naturally positive response | lower stability, more style, slightly faster pace |

These settings are applied only to the individual generation request.

## Endpoints added/extended

### Existing

- `POST /api/v1/voice/sessions`
- `POST /api/v1/voice/sessions/{session_id}/turns`
- `POST /api/v1/voice/sessions/{session_id}/audio-turn`
- `POST /api/v1/voice/sessions/{session_id}/audio-stream-turn`
- `POST /api/v1/voice/sessions/{session_id}/cancel`
- `POST /api/v1/voice/synthesis`
- `POST /api/v1/voice/synthesis/stream`

### Research/demo additions

- `GET /api/v1/voice/emotion/status`
- `POST /api/v1/voice/emotion/warmup`
- `GET /api/v1/voice/sessions/{session_id}/metrics`

Audio turn responses now expose these headers:

- `X-Voice-User-Emotion`
- `X-Voice-Emotion-Confidence`
- `X-Voice-Emotion-Source`
- `X-Voice-Emotion-Top`
- `X-Voice-Voice-Style`

## Demo procedure for a professor

1. Configure the existing Groq and ElevenLabs credentials in `.env`.
2. Install the project with the research dependencies so `transformers`, `torch`, and `numpy` are available.
3. Run the project with `uvicorn app.main:app --reload` or use `start_demo.ps1`.
4. Open `http://127.0.0.1:8000/api/v1/demo`.
5. Click **Warm-up Acoustic SER** once. On the first run, Transformers may download the model from Hugging Face.
6. Start the conversation.
7. Speak an emotional utterance such as a frustrated or sad sentence and let Aiko respond. The right panel shows the detected emotion, confidence, model source, top predictions, response style, and voice style.
8. While Aiko is speaking, interrupt her. The UI shows the measured interruption latency and the backend cancellation result.
9. Click **Refresh Metrics** to show the backend interruption count for the session.

## Current scientific boundary

The acoustic checkpoint is a general speech-emotion classifier and is not presented as a custom Weings-trained model. The Weings research contribution in this prototype is the **end-to-end integration and fusion layer**: acoustic emotion + textual cues -> response policy -> companion expression -> per-response speech delivery, combined with interruptible streaming conversation.

For a publication, the next step should be a controlled experiment comparing:

- baseline voice assistant;
- emotion-aware voice assistant;
- emotion-aware + barge-in voice assistant.

Suggested metrics include emotion recognition accuracy, response appropriateness, acoustic speech quality, interruption success rate, and VAD-to-playback-stop latency.

## References

- SOVA-Bench paper included in the supplied project materials: `2506.02457v1`, arXiv, June 2025.
- Hugging Face model card: https://huggingface.co/superb/wav2vec2-base-superb-er
- ElevenLabs Text-to-Speech API: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- ElevenLabs voice settings: https://elevenlabs.io/docs/api-reference/voices/settings/get

## SOVA-Bench evaluator and result hygiene

The project now includes a benchmark scorer and report builder in `benchmark/`. It calculates and reports standard normalized WER (with S/D/I counts), accuracy, answer rate, conditioned accuracy, raw UTMOSv2, and GPTScore only if a field explicitly contains a GPTScore result. It groups results by model, leaves unrun tasks visible, tracks execution/dataset coverage, and can treat expected-but-unrecorded cases as missing failures. `benchmark/paper_reference_table4.json` stores the paper's Table 4 reference results.

The currently saved runs do **not** constitute a full SOVA-Bench reproduction. In particular, the 150-item pilot is text-only; the saved LibriSpeech run calls Whisper directly and its legacy WER is punctuation-sensitive; Spoken SQuAD has 16 saved result rows against a 500-case run target; and the 20-prompt generation run uses a custom C/R/H/I rubric rather than the paper's GPTScore protocol. See `docs/SOVA_BENCH_AUDIT.md` for the exact audit and reproducible commands.

For the claimed Weings contribution, the proposed controlled ablation is: baseline -> acoustic SER -> hybrid acoustic/text emotion policy -> hybrid emotion plus barge-in. Log each sample and condition using `benchmark/ablation_template.csv`. Report measured improvements only after running all variants with the same audio, model/TTS settings, label mapping, evaluation prompts, and metric normalization.
