# Emotion Aware Voice Assistant

This is the clean, runnable project copy. The original project is preserved separately as a backup. The browser demo, API backend, model configuration, evaluation scripts, selected datasets, and final result files are kept here; temporary caches and superseded benchmark outputs were removed.

## Run the assistant on Windows

1. Install Python 3.13 and make sure internet access is available for dependency setup and configured speech/LLM providers.
2. Check that the local `.env` has valid provider credentials. Do not share or commit that file; `.env.example` is the safe template.
3. Double-click `START_EMOTION_AWARE_VOICE_ASSISTANT.bat`. It checks the dependencies, starts the backend in a minimized terminal, waits for the API to be ready, and opens the demo browser tab.
4. If needed, open `http://127.0.0.1:8000/api/v1/demo?visual=particles` manually. The server terminal remains minimized while the demo runs.

## Supervisor evaluation files

Open `SOVA_Evaluation/README.md` first. The final measured results and auditable calculation files are in `SOVA_Evaluation/outputs/`; evaluation code is in `SOVA_Evaluation/code/`. The package reports 1,000 LibriSpeech ASR utterances plus 55 cases from each of four task datasets. These are fixed-seed samples, not a full SOVA-Bench reproduction. Published baseline rows are paper references, not locally rerun models. The report lists all unmeasured metrics and protocol differences.

## Important project files

- `app/` — FastAPI backend, voice pipeline, providers, emotion handling, and browser UI.
- `benchmark/` — SOVA-compatible adapter code; older generated outputs are kept out of this final copy to avoid mixing them with the final result set.
- `tests/` — automated backend and evaluation tests.
- `runtime_models/` — local model assets used by configured components.
- `requirements.txt`, `pyproject.toml`, and `uv.lock` — Python dependencies and lock file.
- `START_EMOTION_AWARE_VOICE_ASSISTANT.bat` — one-click Windows launcher.
- `SOVA_Evaluation/` — final result tables, formulas, case-level outputs, manifests, and reproducibility code.

The complete LibriSpeech and Spoken SQuAD sources are linked from `SOVA_Evaluation/inputs/` to the preserved local dataset archive; the junction targets are recorded in that folder's README. No result in the report is described as model training: the assistant was evaluated through inference, and no fine-tuning run was performed.
