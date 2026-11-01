# SOVA-Bench model runners (Table 3 / Table 4)

This folder contains batch bridges for the open-source models whose official inference stacks are not exposed through one common HTTP API. The wrappers read **one JSON object per line** from stdin and write **one JSON object per line** to stdout. They process a batch in one process so model weights can stay loaded across examples.

## The six open-source paper baselines plus Emotional Voice Assistant

| `--model` id | Official implementation path | What this ZIP wires up | Additional runtime needed |
|---|---|---|---|
| `mini_omni` | [gpt-omni/mini-omni](https://github.com/gpt-omni/mini-omni) | `mini_omni_batch.py` calls the official `OmniInference` + `A1_A2` path and preserves native text and generated audio | Official checkout/checkpoint and the model's own Python/CUDA environment |
| `mini_omni2` | [gpt-omni/mini-omni2](https://github.com/gpt-omni/mini-omni2) | Native `/chat` multipart text+audio connector | Official server and checkpoint |
| `moshi` | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | `moshi_batch.py` uses the official PyTorch `LMGen` + Mimi API; applies the paper's 2.5-second silence prefix | Official `moshi` Python environment and checkpoint; GPU recommended |
| `llama_omni` | [ictnlp/LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) | Native batch invocation of the official `infer.py`, then the official unit-to-waveform vocoder when configured | Official checkout, weights, Whisper encoder and HiFi-GAN vocoder |
| `freeze_omni` | [VITA-MLLM/Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) | `freeze_omni_batch.py` imports and reuses the official inference pipeline across a batch | Official Freeze-Omni environment/checkpoints |
| `glm4_voice` | [zai-org/GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice) | `glm4_voice_batch.py` tokenizes input speech, calls the official model server, and uses the official audio decoder | Official GLM tokenizer/model/decoder, plus `model_server.py` |
| `weings_ai` (displayed as Emotional Voice Assistant) | This repository's `/api/v1/voice/*` endpoints | Native authenticated audio-turn connector; records transcript, response text, emotion/source/confidence headers and generated audio | Running Weings backend, configured STT/LLM/TTS, development demo token or `SOVA_WEINGS_API_TOKEN` |

**This is the executable model-adapter layer, not a claim that seven large models or their checkpoints ship inside the ZIP.** Model weights are many GB, and upstream projects require separate, often incompatible Python/CUDA environments. Don't install every upstream `requirements.txt` into the Weings virtual environment. Keep each official environment isolated and call its bridge using the environment's Python executable.

**Paid baseline excluded:** Table 3's Cascade = Whisper-large-v3 + GPT-4o-mini. The live runner is intentionally omitted because GPT-4o-mini requires the paid OpenAI API. Published Cascade scores remain in `benchmark/paper_reference_table4.json` as reference only. No other LLM is substituted and reported as the paper's Cascade.

## Standard input format

Every case must contain an existing audio file path for speech-in/speech-out models. Use WAV at a common rate (normally 16 kHz for upstream speech inputs; Moshi resamples to its 24 kHz codec rate). Useful optional fields include `id`, `case_id`, `task`, `prompt`, `question`, `reference_text`, `answers`, `expected_label`, `labels`, and `dataset_split`.

```json
[
  {
    "case_id": "knowledge_0001",
    "task": "knowledge_qa",
    "audio_path": "D:/datasets/sova_audio/knowledge_0001.wav",
    "prompt": "What is the capital of France?",
    "answers": ["Paris"]
  }
]
```

The `prompt`/`question` fields are metadata for evaluation unless a model explicitly accepts simultaneous text+audio. Do **not** use the text answer/target as a model input. Audio-only models must receive a recording containing the benchmark's intended spoken input. A text-only TriviaQA or Alpaca file is not by itself an audio benchmark; it needs the declared speech-synthesis preparation first.

## Official runtime setup

### Mini-Omni

The Table-4 runner uses the official batch inference path rather than `/chat`, because Table 4 requires text/audio consistency and the HTTP endpoint may expose audio without a native text stream. Install the official repo in its own supported environment, download the checkpoint (including `lit_model.pth`), and configure the bridge command from the Weings project root:

```powershell
$env:SOVA_MINI_OMNI_REPO = "D:\models\mini-omni"
$env:SOVA_MINI_OMNI_CHECKPOINT = "D:\models\mini-omni\checkpoint"
$env:SOVA_MINI_OMNI_DEVICE = "cuda:0"
$env:SOVA_MINI_OMNI_BRIDGE_CMD = '"D:\miniconda\envs\mini-omni\python.exe" "D:\AIML\WEINGS_AI_SOVA_FIXED\benchmark\adapters\mini_omni_batch.py"'
```

The wrapper loads model weights once for the whole JSONL batch and writes one WAV plus native model text per case. Keep this environment isolated from Weings and other baselines.

### Mini-Omni2

Use its own official environment and start its server on a **different port** if Mini-Omni is also running:

```powershell
python server.py --ip 127.0.0.1 --port 60809
```

```powershell
$env:SOVA_MINI_OMNI2_URL = "http://127.0.0.1:60809"
```

### LLaMA-Omni

Clone the official repository, install its documented dependencies and download the model/Whisper/vocoder assets. Point the runner at that checkout:

```powershell
$env:SOVA_LLAMA_OMNI_REPO = "D:\models\LLaMA-Omni"
$env:SOVA_LLAMA_OMNI_MODEL = "D:\models\LLaMA-Omni\Llama-3.1-8B-Omni"
$env:SOVA_LLAMA_OMNI_VOCODER = "D:\models\LLaMA-Omni\vocoder\g_00500000"
$env:SOVA_LLAMA_OMNI_VOCODER_CFG = "D:\models\LLaMA-Omni\vocoder\config.json"
```

The upstream repository documents Linux/CUDA-centric install steps (including `fairseq` and `flash-attn`); running from its own supported environment is recommended. The wrapper runs inference once for the batch, rather than reloading an 8B model for every case.

### Moshi

Use the official PyTorch Moshi environment. The official project currently notes significant GPU-memory requirements (24 GB in its PyTorch README) and does not promise full Windows support. In the PowerShell/runner environment, point to the upstream environment's Python:

```powershell
$env:SOVA_MOSHI_BRIDGE_CMD = '"D:\miniconda\envs\moshi\python.exe" "D:\AIML\WEINGS_AI_SOVA_FIXED\benchmark\adapters\moshi_batch.py"'
$env:SOVA_MOSHI_HF_REPO = "kyutai/moshiko-pytorch-bf16"
```

The bridge honors `SOVA_MOSHI_DEVICE`, `SOVA_MOSHI_TEMP`, `SOVA_MOSHI_TEXT_TEMP`, and `SOVA_MOSHI_SILENCE_PREFIX_SECONDS` (defaults to 2.5 seconds, matching the supplied paper's Moshi note).

### Freeze-Omni

Follow the official upstream installation and download its Freeze-Omni plus Qwen2-7B-Instruct checkpoints. Example:

```powershell
$env:SOVA_FREEZE_OMNI_REPO = "D:\models\Freeze-Omni"
$env:SOVA_FREEZE_OMNI_MODEL_PATH = "D:\models\Freeze-Omni\checkpoints"
$env:SOVA_FREEZE_OMNI_LLM_PATH = "D:\models\Freeze-Omni\Qwen2-7B-Instruct"
$env:SOVA_FREEZE_OMNI_BRIDGE_CMD = '"D:\miniconda\envs\freeze-omni\python.exe" "D:\AIML\WEINGS_AI_SOVA_FIXED\benchmark\adapters\freeze_omni_batch.py"'
```

### GLM-4-Voice

Follow the official upstream README to download its tokenizer, 9B model and decoder. Start the official model server in the GLM environment first. The upstream README currently documents the server on port 10000; its decoder assets are separate from the Transformer model.

```powershell
$env:SOVA_GLM4_VOICE_REPO = "D:\models\GLM-4-Voice"
$env:SOVA_GLM4_VOICE_MODEL_PATH = "D:\models\GLM-4-Voice\glm-4-voice-9b"
$env:SOVA_GLM4_VOICE_TOKENIZER_PATH = "D:\models\GLM-4-Voice\glm-4-voice-tokenizer"
$env:SOVA_GLM4_VOICE_DECODER_PATH = "D:\models\GLM-4-Voice\glm-4-voice-decoder"
$env:SOVA_GLM4_VOICE_SERVER_URL = "http://127.0.0.1:10000"
$env:SOVA_GLM4_VOICE_BRIDGE_CMD = '"D:\miniconda\envs\glm4-voice\python.exe" "D:\AIML\WEINGS_AI_SOVA_FIXED\benchmark\adapters\glm4_voice_batch.py"'
```

### Emotion Aware Voice Assistant

Start the backend in development mode. The runner calls `GET /api/v1/demo/token` and `POST /api/v1/voice/sessions`, then sends each WAV to `POST /api/v1/voice/sessions/{session_id}/audio-stream-turn`. In development only, the demo-token endpoint can mint its scoped short-lived token. In a non-development setup, provide a suitable JWT in `SOVA_WEINGS_API_TOKEN`. The model's STT/LLM/TTS and acoustic SER must actually be configured for the measurement to represent the complete proposed pipeline.

## Commands

Run from the extracted Weings project root in PowerShell:

```powershell
# list all baseline and proposed-model runner IDs
python -m benchmark.run_sova_model --model mini_omni --input path\to\audio_manifest.json --list-models

# smoke test one model on five real audio cases
python -m benchmark.run_sova_model --model mini_omni --input path\to\audio_manifest.json --limit 5

# run all configured adapters; unconfigured bridges are recorded as explicit failures
python -m benchmark.run_sova_model --model all --input path\to\audio_manifest.json

# GPT-OSS 120B judgement for QA and semantic generation; uses the existing GROQ_API_KEY
$env:SOVA_JUDGE_MODEL = "openai/gpt-oss-120b"
python -m benchmark.judge_sova_outputs --input benchmark\runs\weings_ai\results.json

# transcribe only generation-task response audio and add consistency rows; uses GROQ_API_KEY
python -m benchmark.transcribe_sova_outputs --input benchmark\runs\weings_ai\results.json
# (add --all-audio-tasks only for a separate diagnostic, not Table 4 consistency)

# score ONLY Alpaca/generation-task WAVs with UTMOSv2; requires utmosv2 installation
python -m benchmark.acoustic_eval --audio-dir benchmark\runs\weings_ai --input benchmark\runs\weings_ai\results.json --generation-only --output benchmark\runs\weings_ai\acoustic

# attach the UTMOSv2 results as separate generation_acoustic records (makes a backup)
python -m benchmark.attach_sova_metrics --results benchmark\runs\weings_ai\results.json --acoustic-results benchmark\runs\weings_ai\acoustic\acoustic_results.json

# one-command post-processing for all completed model runs (API keys/UTMOSv2 only if configured)
powershell -ExecutionPolicy Bypass -File benchmark\postprocess_sova_runs.ps1

# build a Table-4-shaped comparison; reference values and local measurements are separate rows
python -m benchmark.compare_sova_table4 --runs-root benchmark\runs --output benchmark\Table4_comparison.csv
```

`--list-models` can be used without model inference. `--input` is required before an actual run. Always do a five-case smoke test before the full set and check `summary.json` coverage/errors.

## Metric definitions / interpretation

- **Accuracy:** correct target cases / all target cases × 100. QA correctness can be adjudicated by `judge_sova_outputs.py`; it records the exact judge prompt version. Without that step, any exact-label/stored-correctness result is a diagnostic, not a reproduction of the authors' `GPTEval` prompt.
- **Answer rate (AR):** answered / all target cases × 100. **Conditioned accuracy (CA):** correct / answered × 100. The existing `score_sova_results.py` reports these separately for the paralinguistic tasks where records include the required fields.
- **WER:** corpus WER = (total substitutions + deletions + insertions) / total reference words × 100. Split-specific `dataset_split` must be present (`dev-clean`, `dev-other`, `test-clean`, `test-other`).
- **Consistency WER:** requires two separate outputs: the model's text response and ASR transcript of its generated audio. Audio-only Mini-Omni output alone is insufficient to produce that consistency metric without a separate model text stream.
- **GPTScore:** operationalized in `judge_sova_outputs.py` as a 1–5 semantic-quality rating with a recorded prompt version. The five-page paper does not specify its exact judge prompt, so report this as a declared compatible implementation, not byte-identical author reproduction.
- **UTMOSv2:** install the official predictor in the benchmark environment with `python -m pip install git+https://github.com/sarulab-speech/UTMOSv2.git`. Use `benchmark/acoustic_eval.py --generation-only` so spoken-QA response audio is not mixed into the Alpaca generation acoustic score. Don't average values from failed audio outputs; report the number of scored clips.

## Reproduction limitation that must be stated in a paper

The source preprint doesn't release a complete official evaluation implementation. A benchmark-compatible implementation can reproduce the named tasks and reported metric types, but cannot truthfully claim exact author-level numerical reproduction unless the dataset filtering, TTS prompt-speaker set, judge prompts, class mappings, model revisions and inference settings also match. `benchmark/sova_model_registry.json` describes each model connector; `benchmark/sova_protocol.json` lists source datasets and remaining source-data limitations; `benchmark/paper_reference_table4.json` is only the published comparison reference.
