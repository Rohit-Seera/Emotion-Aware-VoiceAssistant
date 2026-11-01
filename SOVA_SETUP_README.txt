EMOTIONAL VOICE ASSISTANT — SOVA-BENCH SETUP

This package includes the SOVA-compatible evaluation scripts and report scaffold.
The local .env file is intentionally NOT included. Create/maintain your own local
.env from .env.example and put newly rotated credentials into it.

IMPORTANT SECURITY NOTE
The uploaded source ZIP contained a populated .env with provider credentials.
For safety, rotate/revoke any real keys from that file (LLM provider, speech/TTS,
embedding services) before using them again. Never commit or share .env.

OFFLINE SETUP
  .\.venv\Scripts\Activate.ps1
  powershell -ExecutionPolicy Bypass -File benchmark\run_sova_setup.ps1

The script runs benchmark tests, checks the model registry and data inventory,
and creates a SOVA Table-4-shaped CSV. It does not download weights, synthesize
full datasets, call model APIs, or train anything.

FIRST REAL RUN
  1. Set up one registered model's official runtime/checkpoint.
  2. Prepare a real audio-backed manifest, e.g. LibriSpeech.
  3. Run benchmark\run_sova_smoke.ps1 on five examples.
  4. Inspect raw outputs and scored task_summary.csv before increasing the count.

See benchmark\RESULTS_AND_CALCULATIONS_GUIDE.md for formulas, output paths,
dataset coverage, and the difference between historical pilot results and
new measured Table-4 rows.
