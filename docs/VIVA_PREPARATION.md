# Viva Preparation

## Explain the system

| Component | What it does and uses | Input → output | Evidence and failure modes |
|---|---|---|---|
| API and demo | FastAPI serves the local page and authenticated voice routes. | Text/audio request → JSON or streamed audio | `app/main.py`, `app/api/router.py`, `app/api/v1/voice.py`; check `/api/v1/health/live`. Provider outages or invalid credentials return errors or labelled fallback. |
| Speech recognition | Hosted Groq Whisper `whisper-large-v3-turbo`. | Audio bytes → transcript | `app/services/speech/groq_stt.py::GroqWhisperSTT.transcribe`; needs credentials, supported audio, internet, and quota. |
| Emotion detection | Public pretrained SUPERB Wav2Vec2 plus keyword rules; no training or fine-tuning. | WAV and/or transcript → label, score, source | `app/engines/emotion/acoustic.py`, `engine.py`, `hybrid.py`; model download/runtime can fail; compressed input uses text fallback. Scores are uncalibrated. |
| Fusion and policy | Explicit rules select acoustic/lexical signals; prompt builder conditions tone. | Predictions → final label/source → policy text | `app/engines/emotion/hybrid.py`, `app/runtime/nodes/prompt_builder.py`; policy output is not accuracy evidence. |
| LLM | Configured Groq GPT-OSS 120B via hosted API, with configured fallback behavior. | Conditioned prompt → response | `app/runtime/nodes/llm.py`, `app/providers/manager/provider_manager.py`; response reports actual provider/model, including development fallback. |
| Speech generation | Hosted ElevenLabs TTS with style settings; browser speech is an explicit fallback. | Assistant text/style → audio stream or browser speech | `app/providers/elevenlabs/provider.py`, `app/services/speech/service.py`; network and quota matter. UI logs actual playback provider. |
| Barge-in | Browser VAD stops playback, aborts streaming, and asks backend to cancel the active session task. | Speech during playback → cancelled output and next turn | `app/api/v1/demo.html`, `app/api/v1/voice.py`, `app/services/speech/service.py::VoiceSessionRegistry.cancel`; automated tests exist, actual mic interruption still needs a manual browser check. |
| SOVA evaluation | Scorers calculate defined metrics and preserve task/sample provenance. | Predictions plus labels/references → reports | `benchmark/sova_metrics.py`, `benchmark/score_sova_results.py`, `benchmark/run_sova_model.py`; local runs differ in coverage and are not a full paper reproduction. |

## Common questions

**Which model detects emotion?** The optional local pretrained checkpoint `superb/wav2vec2-base-superb-er`. If it is unavailable, or compressed audio cannot be decoded by that path, the system identifies the lexical text-proxy fallback.

**Did you train it?** No. This phase uses inference and integration. The checkpoint was pretrained by its source authors; Weings has not trained or fine-tuned it.

**What classes does it output?** The code maps anger, happiness, neutral, and sadness. Keyword rules also recognize anxious and frustrated; those are not additional acoustic classes.

**How are predictions fused?** A high-confidence lexical cue may replace neutral acoustics; agreeing labels get a small capped confidence adjustment; otherwise a sufficiently confident acoustic label wins; weak or missing audio uses lexical output. Branch tests are in `tests/runtime/test_hybrid_emotion.py`.

**Where does emotion affect the answer?** `app/runtime/nodes/prompt_builder.py` adds the detected label, source, score, and a short response policy. The expression planner selects constrained voice/animation settings.

**How does voice generation work?** ElevenLabs receives text and the configured voice/style; audio streams from the backend. If unavailable, the response identifies a browser speech fallback.

**How does barge-in work?** Browser VAD watches microphone energy during speech, stops playback, aborts the browser request, then calls the authenticated cancellation route. The server tracks the active streamed task per session. Automated tests cover cancellation; actual mic behavior depends on browser permissions and still needs a manual check.

**Where is WER calculated?** `benchmark/sova_metrics.py::wer_stats` aligns normalized words and counts substitutions, deletions, and insertions. `corpus_wer` sums edit counts before dividing by reference words. The old LibriSpeech script's `calculate_wer` instead reports a mean of per-item WER values.

**Why can WER exceed 100%?** Insertions add errors while reference words stay in the denominator. Three errors against a one-word reference yields 300%.

**Why is emotion confidence not accuracy?** Confidence is one model's score for one prediction. Accuracy is the fraction matching ground-truth labels across a labelled test set. No valid labelled audio test set was used to measure this model's accuracy here.

**How many examples were evaluated?** Counts differ by saved experiment: LibriSpeech 500 transcription cases; Spoken SQuAD 16 successes and 484 failures of 500; generation UTMOS 20 audio samples; custom text emotion pilot 50 cases. These are separate runs and test sets.

**What is GPTScore?** A semantic judge score is not proof of factual correctness. The saved 20-example semantic pilot uses a custom 1–5 rubric, not the paper's exact GPTScore protocol.

**What does UTMOSv2 show?** It predicts speech naturalness on a MOS-like 1–5 scale. It is not human listening MOS or accuracy. The stored historical mean is 2.677 over 20 audio samples.

**Which values are published?** `benchmark/paper_reference_table4.json` contains the supplied paper's Table 4 values. Local measurements are stored separately; a published number is not a result from this computer.

**Can the results be reproduced?** The scorer and manifests are repeatable. Exact reproduction still requires matching data, model revisions/checkpoints, splits, audio preparation, judge prompts, and settings. These are not all present, so exact paper reproduction is not claimed.

**What demonstrates the app works?** The live report records health/demo/OpenAPI checks, a real Groq text turn, Groq Whisper transcription, and real ElevenLabs audio output. Unit tests cover authentication, emotion fusion, prompt building, metrics, and cancellation. Browser microphone permission and physical interruption need a presenter-side check.

## Important distinctions

- Training changes weights; inference uses existing weights; evaluation compares outputs with references or a declared judge.
- STT WER compares incoming speech with a human transcript. Consistency WER compares assistant text with ASR of generated speech.
- Confidence is per prediction; accuracy is an aggregate over labelled examples.
- Accuracy, precision, recall, and F1 have different denominators. This scorer reports accuracy/answer rate/conditional accuracy, not per-class F1.
- Semantic judge scores assess a rubric and can miss factual errors.
- UTMOSv2 is a learned acoustic predictor; human MOS requires listening ratings and a human-test protocol.
- Unit tests isolate functions; integration tests exercise connected components; live tests call configured providers. Mock tests do not prove a live provider works.
- A working emotional response policy shows the prompt can change; it does not establish emotion-classification accuracy.
