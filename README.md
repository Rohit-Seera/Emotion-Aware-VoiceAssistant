# Emotion-Aware Voice Assistant

## Presentation and report

- Editable report: project_materials/Emotional_Aware_Voice_Assistant_Report.docx
- Readable report copy: project_materials/Emotional_Aware_Voice_Assistant_Report.pdf
- Editable 9-slide PowerPoint: project_materials/Emotion_Aware_Voice_Assistant_Presentation.pptx
- Explained workflow animation: project_materials/Assistant_Workflow_Animation.mp4
- Canva editor deck (9 slides, includes the animated workflow): https://www.canva.com/d/_ZNOQGYHebgtdrv

The report and slides summarize the saved pilot evaluation and state the protocol limits. Dataset materials are included only where redistribution is permitted; some source datasets, model weights, and large case-level artifacts remain local or linked as described in SOVA_Evaluation/README.md. Never commit .env or provider keys.

An emotion-aware voice assistant built around a FastAPI backend, speech processing, configurable LLM and TTS providers, emotion handling, and a browser-based demo.

## Run on Windows

1. Install Python 3.13 and Git.
2. Dependencies are declared in pyproject.toml and uv.lock. For manual setup, run uv sync --extra dev --locked from the project root. The BAT launcher installs dependencies automatically (using uv when available, otherwise pip).
3. Copy .env.example to .env and add your own provider credentials. Keep .env private.
4. Run START_EMOTION_AWARE_VOICE_ASSISTANT.bat from the project folder.
5. When the server is ready, open http://127.0.0.1:8000/api/v1/demo?visual=particles if the browser does not open automatically.

Provider-backed features require valid local credentials and the relevant provider services. Never commit .env or paste API keys into source files.

## Evaluation results

The evaluation package reports 1,220 cases across five task families:

| Task | Result |
|---|---:|
| LibriSpeech ASR | 3.55% corpus WER across 1,000 utterances |
| TriviaQA | 35.58% mean token-F1; 0.00% exact-match proxy across 55 cases |
| LibriSQA Part II | 74.55% accuracy across 55 cases; 78.85% among 52 valid options |
| Spoken SQuAD | 25.54% mean token-F1; 0.00% exact-match proxy across 55 cases |
| Alpaca | 20.93% ROUGE-L F1 and 26.66% token-F1 across 55 prompts |

See SOVA_Evaluation/README.md and SOVA_Evaluation/outputs/SOVA_FINAL_RESULTS_AND_LIMITATIONS.md for detailed metrics, formulas, comparison references, and limitations.

These are fixed-seed, limited samples, not a full SOVA-Bench reproduction. Published baseline values are references from the paper and were not rerun locally. Exact-match, token-F1, and ROUGE-L are lexical proxy metrics; they are not the paper's GPT-evaluator accuracy or GPTScore. IEMOCAP emotion accuracy, Common Voice age/gender metrics, and speech-generation quality metrics were not measured. No foundation-model fine-tuning is claimed.

## Project structure

- app/ — FastAPI backend, voice pipeline, providers, emotion handling, and browser demo.
- benchmark/ — model adapters and benchmark utilities.
- tests/ — automated tests.
- docs/ — project and research documentation.
- SOVA_Evaluation/ — evaluation code, aggregate results, formulas, and limitations.
- START_EMOTION_AWARE_VOICE_ASSISTANT.bat — Windows launcher.
- pyproject.toml and uv.lock — Python project dependencies.

## Public repository scope

Aggregate evaluation reports and summaries are included. Full source datasets, raw per-case transcripts, case-level JSONL outputs, evaluation audio, and model weights are not included in this public repository. The .env file is local configuration and must never be committed.