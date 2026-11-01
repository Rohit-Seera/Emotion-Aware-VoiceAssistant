# Next steps to obtain measured SOVA-Bench calculations

This checklist is deliberately staged. A registered adapter is not a completed measurement: its official model/runtime, audio inputs, credentials, and successful inference run are still required.

## 1. Extract and verify the project

Open PowerShell in this extracted project root. Activate the virtual environment already used for this project. If it is not present, create one and install the project dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests\benchmark -q
python -m pytest tests\runtime\test_personality.py -q
python -m benchmark.run_sova_model --model all --list-models
```

Do not install the official repositories for all 7 baselines into the same virtual environment. Keep their potentially incompatible Python/CUDA stacks separate.

## 2. Configure Emotion Aware Voice Assistant and API credentials

Copy `.env.example` to `.env`. For local development, set a fresh random `SECURITY_JWT_SECRET_KEY` meeting the configured minimum length and set actual provider credentials. The Weings backend reads its project settings from `.env`.

The live benchmark registry intentionally excludes the paper Cascade because it requires paid GPT-4o-mini. The judge step uses the user’s existing `GROQ_API_KEY` with `openai/gpt-oss-120b`; local Whisper-large-v3 is used for generated-audio transcription. Never send secret values to the assistant or commit `.env` to Git.

Start the backend in a separate terminal:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/status
Invoke-RestMethod http://127.0.0.1:8000/api/v1/demo/diagnostics
```

Do not start Weings benchmark runs unless the server has successfully started and the diagnostics show that the expected STT/LLM/TTS providers are configured. If this fails, resolve the backend startup/configuration error first.

## 3. Prepare the actual audio-backed manifest

The model runner does not synthesize audio or download the official datasets for you. Every input record needs an existing `audio_path` (or supported equivalent). For a first test, prepare **five actual LibriSpeech test-clean examples** with their matching audio and references. Create `benchmark\manifests\librispeech_5.json` as a list of records like this; replace the example values with actual downloaded paths and references:

```json
[
  {
    "case_id": "actual-librispeech-utterance-id",
    "task": "speech_recognition",
    "dataset_split": "test-clean",
    "audio_path": "D:/datasets/LibriSpeech/test-clean/REPLACE_WITH_REAL_AUDIO.flac",
    "reference_text": "REPLACE WITH THE OFFICIAL REFERENCE TRANSCRIPT"
  }
]
```

Include five real rows before using `--limit 5`. Keep the same case IDs/audio/reference rows for every model in this comparison. The manifest preparation utility can attach local audio when its filename stem matches the case ID:

```powershell
python -m benchmark.prepare_sova_audio_manifest --input benchmark\manifests\librispeech_5_without_audio.json --audio-dir D:\datasets\LibriSpeech\test-clean --output benchmark\manifests\librispeech_5.json
```

The input JSON must already contain the task labels and reference text. This command only matches audio paths; it does not synthesize or fetch audio.

## 4. Run a small comparable smoke test first

Run a supported open-source SOVA baseline and Emotion Aware Voice Assistant on the same five audio cases. The paid Cascade is not run; its paper values remain reference-only:

```powershell
python -m benchmark.run_sova_model --model cascade --input benchmark\manifests\librispeech_5.json --limit 5 --output-dir benchmark\runs_smoke
python -m benchmark.run_sova_model --model weings_ai --input benchmark\manifests\librispeech_5.json --limit 5 --output-dir benchmark\runs_smoke
python -m benchmark.compare_sova_table4 --runs-root benchmark\runs_smoke --output benchmark\runs_smoke\Table4_comparison.csv
```

Inspect `benchmark\runs_smoke\cascade\summary.json` and `benchmark\runs_smoke\weings_ai\summary.json`. Stop and fix any failed cases before scaling. The smoke test checks wiring only; it is not a publishable full SOVA-Bench score.

## 5. Expand by SOVA-Bench task, preserving modality and splits

The supplied paper's Table 2 targets are: TriviaQA 6,840; LibriSQA 2,619; Spoken SQuAD 3,243; IEMOCAP emotion 5,819; Common Voice gender 2,469; Common Voice age 2,511; and Alpaca generation 3,745. LibriSpeech must preserve dev-clean, dev-other, test-clean, and test-other splits. Some model/task cells are absent in Table 4; leave those as not reported, not zero.

The JSON included for TriviaQA contains text questions/answers, and the included generation prompt file has only 20 custom prompts. These are not the paper's full spoken inputs. The supplied paper describes synthesizing the text-only TriviaQA and Alpaca questions with CosyVoice, then filtering audio using Whisper-large-v3. Reproduce that audio preparation (or document any deviation) before claiming same-protocol speech evaluation. IEMOCAP/Common Voice source data and labels also need to be obtained and prepared.

Run each task on its own correct manifest, with matching audio/labels/references and task-appropriate split names. For each model and manifest, use a distinct `--output-dir` or `--output` path to avoid overwriting prior results. Check the saved `summary.json`, successful count, errors, and coverage before running the scorer/post-processing.

## 6. Configure the paper models one by one

Use `benchmark/adapters/README.md` and `benchmark/sova_model_registry.json` for each model's official checkpoint/runtime and bridge variables. Mini-Omni, Mini-Omni2, Moshi, LLaMA-Omni, Freeze-Omni, and GLM-4-Voice need their own upstream code/weights and compatible runtimes; the ZIP only provides connectors/bridge contracts, not the large checkpoints. Run a smoke test for each model first. If a bridge/runtime is absent, its rows will be failures and are not measurements.

## 7. Compute and report the Table 4 metrics

For QA/semantic judging, use the declared judge script and record its prompt/version; the preprint does not provide every judge implementation detail. For generation rows, keep response-text vs generated-audio transcript WER, GPTScore/judge score, and raw UTMOSv2 score separate. Run acoustic scoring only on the Alpaca/generation task. Finally export a Table-4-shaped CSV with `benchmark.compare_sova_table4` and report measured rows separately from published paper-reference rows.

Do not describe the paper as fully reproduced until the required task counts and model runs are complete, comparable, and audited. Do not report missing or failed runs as zero, and do not present published reference scores as new measurements.
