# SOVA-Bench: paper models only, no paid baseline APIs

This setup follows the exact model list in Table 3 of the supplied SOVA-Bench preprint. It does not add a substitute model to the baseline list.

## Runnable registry

The live registry contains the six non-Cascade speech models named in the paper, plus the user's own Emotion Aware Voice Assistant system:

1. **Mini-Omni** — official repository/checkpoint, 0.5B.
2. **Mini-Omni2** — official repository/checkpoint, 0.5B.
3. **Moshi** — official repository/checkpoint, 7B.
4. **LLaMA-Omni** — official repository/checkpoint, 8B.
5. **Freeze-Omni** — official repository/checkpoint, 7B; its official setup includes the Qwen2-7B-Instruct component named in the paper setup, not an added baseline.
6. **GLM-4-Voice** — official repository/checkpoint; the paper's architecture and parameter-size entries are not fully consistent, so the table discrepancy is documented rather than silently corrected.
7. **Emotion Aware Voice Assistant** — the user's proposed system, using the already-configured GPT-OSS 120B where enabled.

The runner adapters are included, but model checkpoints and upstream Python environments are not bundled in the ZIP. A wrapper is not evidence that a model has run: each official environment/checkpoint must be installed and its smoke test must succeed.

## Excluded from live runs: the paper Cascade

SOVA-Bench's Cascade is **Whisper-large-v3 + GPT-4o-mini**. Because GPT-4o-mini calls the paid OpenAI API, the Cascade runner has been removed from live runs. The published Table 4 values remain in `benchmark/paper_reference_table4.json` and the audit CSV as **paper reference only**. We do not replace GPT-4o-mini with another LLM and call that the same baseline.

The user's existing Groq GPT-OSS 120B may remain the Emotion Aware Voice Assistant LLM and may judge QA/semantic outputs via the already-configured Groq access. When used as a judge, its name and provider are recorded. It is **not** one of the paper's speech-model baselines, nor proof of byte-identical reproduction of the authors' unspecified `GPTEval` prompts.

## Groq access note

The model ID is `openai/gpt-oss-120b`. Use the user's current `GROQ_API_KEY` and available account quota; do not add a payment method. Free-tier access/limits are account-specific and can change. If the model is unavailable or rate-limited, stop and report the error; do not silently fall back to another LLM.

## Step 1 — List allowed runners

From the project root:

```powershell
python -m benchmark.run_sova_model --model all --list-models
```

The runnable list should show the six SOVA open-source baselines and Emotion Aware Voice Assistant. Cascade should appear only with an `EXCLUDED` note, not as a runnable model.

## Step 2 — Keep official model environments isolated

Each upstream model has its own runtime/dependencies. Follow the official repo links in `benchmark/adapters/README.md`, download the corresponding official checkpoint, and configure the bridge command. Do not install all models' requirements into the Weings environment. An unconfigured bridge must fail explicitly instead of emitting a made-up score.

Your RTX 3050 6GB will not fit every paper baseline in its recommended setup. Larger models may require an actually available free GPU runtime; do not assume a notebook has a guaranteed GPU. Only report a model after its real run completes.

## Step 3 — Prepare actual audio inputs

Every speech model needs real audio paths and correct reference/label metadata. Follow the source dataset and splits in `benchmark/README.md`; never feed clean target text to an audio-only model. The paper synthesizes TriviaQA and Alpaca audio with CosyVoice and filters audio with Whisper-large-v3, so text-only JSON is not enough for these speech tasks.

Example manifest shape (replace with actual existing audio and reference transcript):

```json
[
  {
    "case_id": "actual-utterance-id",
    "task": "speech_recognition",
    "dataset_split": "test-clean",
    "audio_path": "D:/datasets/LibriSpeech/actual-audio.flac",
    "reference_text": "official reference transcript"
  }
]
```

## Step 4 — Run the same cases through a paper model and Weings

```powershell
python -m pip install faster-whisper
$env:SOVA_OUTPUT_ASR_MODEL = "large-v3"
$env:SOVA_OUTPUT_ASR_DEVICE = "cpu"
$env:SOVA_OUTPUT_ASR_COMPUTE_TYPE = "int8"

python -m benchmark.run_sova_model --model mini_omni --input benchmark\manifests\librispeech_5.json --limit 5 --output-dir benchmark\runs
python -m benchmark.run_sova_model --model weings_ai --input benchmark\manifests\librispeech_5.json --limit 5 --output-dir benchmark\runs
```

Repeat for `mini_omni2`, `moshi`, `llama_omni`, `freeze_omni`, and `glm4_voice` only after their official runtime bridges are configured. `--model all` runs only these six paper baselines plus Emotion Aware Voice Assistant; it will not call OpenAI paid APIs.

## Step 5 — Use the existing GPT-OSS 120B for judging

```powershell
$env:SOVA_JUDGE_MODEL = "openai/gpt-oss-120b"
python -m benchmark.judge_sova_outputs --input benchmark\runs\weings_ai\results.json
```

This uses `GROQ_API_KEY`, explicitly requests GPT-OSS 120B, and records the judge model and prompt version. If the model is unavailable or rate-limited, stop and surface the error—do not substitute another model. Since the paper's exact judging prompt/model is not fully specified in the five-page preprint, label this as a GPT-OSS 120B operationalization of the paper's GPTEval/GPTScore metric family, not a byte-identical author reproduction.

## Step 6 — Metric calculations

- **WER:** use the paper's LibriSpeech split definitions and corpus WER formula.
- **Generation consistency:** compare the system's native text response with a separate Whisper-large-v3 transcript of its generated audio. Keep these as separate fields; never force WER to zero by copying a transcript into both fields.
- **QA accuracy / semantic score:** the current GPT-OSS 120B judge operationalizes the same metric family, and records the judge/model. The paper's undisclosed exact judge prompt remains a replication limitation.
- **UTMOSv2:** install the official open evaluator and score only generated Alpaca-task audio. Do not substitute speech-recognition or QA audio.
- **Paralinguistic tasks:** use SOVA's five-choice emotion class mapping plus Common Voice age/gender mappings. Report answer rate and conditioned accuracy where applicable.

Keep published Table 4 reference scores separate from new measurements. Missing tasks remain `not_run`/blank, never 0.00. Record dataset split, attempted/successful cases, model revision, hardware, inference settings, judge model and output files before presenting results.
