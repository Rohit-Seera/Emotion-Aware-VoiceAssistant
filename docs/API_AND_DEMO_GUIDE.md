# Windows API and Demo Guide

## Start the local application

Open PowerShell:

```powershell
Set-Location 'D:\sova\WEINGS_AI_SOVA_PAPER_MODELS_ONLY (1)\WEINGS_AI_SOVA_PAPER_MODELS_ONLY'
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep that window open. FastAPI serves both the API and demo; there is no separate frontend server. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that window, then activate again. That policy change ends with that PowerShell process.

The local `.env` is already configured. For a fresh copy, duplicate `.env.example` as `.env` and fill credentials locally. Never paste key values into a terminal transcript, screenshot, issue, or report. To inspect selected non-secret settings and credential presence:

```powershell
python -c "from app.core.settings import settings; print('LLM:', settings.ai.default_provider, getattr(getattr(settings.ai, settings.ai.default_provider), 'model', 'unknown')); print('STT:', settings.voice.stt_provider, bool(settings.ai.groq.api_key)); print('TTS:', settings.voice.tts_provider, bool(settings.elevenlabs.api_key), bool(settings.elevenlabs.voice_id))"
```

## Open and verify

- Demo: `http://127.0.0.1:8000/api/v1/demo`
- Liveness: `http://127.0.0.1:8000/api/v1/health/live`
- Readiness: `http://127.0.0.1:8000/api/v1/health/ready` (checks database/cache)
- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Safe diagnostics: `http://127.0.0.1:8000/api/v1/demo/diagnostics`

Liveness should return `{"status":"alive"}` and the demo should return HTTP 200. Diagnostics exposes provider names/readiness, never credentials. The demo token route is development-only and is expected to return 404 in production.

## Present the demo

1. Open the demo and wait for “Ready”; it creates an authenticated process-local session.
2. Type any message and choose **Send text**. Verify the exact input, assistant response, emotion source/confidence, actual LLM provider/model, request ID, and backend time. TTS plays the response and logs its actual provider or browser fallback.
3. Choose **Choose audio file**, select WAV/MP3/OGG/WebM under 10 MiB and three minutes, then click **Upload and process**. Verify actual transcript, emotion result/source/confidence, answer, request ID, LLM, latency, and streamed audio. The API also enforces a 10 MiB byte limit. Acoustic SER expects WAV; compressed audio can correctly report a text-proxy fallback.
4. Click the play button to request microphone access. Speak clearly and pause to end the turn; verify transcript and response. If automatic voice detection misses speech, click **Record voice**, speak, then click **Finish & send**. Manual recording is limited to 90 seconds to stay within the WAV upload limit.
5. With headphones, speak while Aiko is talking. Confirm playback stops, interruption metrics update, backend cancellation reports its result, and the next turn is captured.
6. Click **New Session / Reset** to cancel active work, stop stale playback, clear outputs, and create a new session.

The browser mic and human barge-in have not been verified by unattended HTTP checks. Test them once in Chrome or Edge and on the machine used for the viva. The embedded Codex preview may not provide real audio output; the demo now exposes native audio controls and a download link, but use a full browser for live speech.

## Tests and saved evidence

```powershell
python -m pytest -q
python -m pytest -q tests\api\test_main.py tests\api\test_router.py tests\api\test_voice.py tests\runtime\test_llm.py tests\runtime\test_hybrid_emotion.py tests\runtime\test_prompt_builder.py tests\benchmark tests\services\speech\test_service.py
python -m compileall -q app tests
```

Test outputs are in `results/baseline_pytest.log`, `results/baseline_pytest_importlib.log`, `results/focused_pytest.log`, and `results/final_pytest.log`. Current live provider evidence is `results/live_e2e_smoke.md`. Historical benchmark rows and summaries are in `benchmark/results_*`; new common-runner results belong under `benchmark/runs/<model_id>/`.

Score saved rows without new provider calls:

```powershell
python -m benchmark.score_sova_results --input benchmark\results_generation\sova_generation_merged.json --output benchmark\results_generation\scored_sova
python -m benchmark.compare_sova_table4
```

For new model runs, inspect `benchmark/README.md`, `benchmark/runs/README.md`, `benchmark/sova_model_registry.json`, and `benchmark/manifests/`. Run the five-case smoke before scaling. Official open-model checkpoints, compatible environments, or bridge commands are not configured here.

## Troubleshooting

- **404 on `/api/v1/...`:** confirm the running process comes from this checkout and restart it after code changes. The corrected router mounts `/api/v1`.
- **Demo token 404:** production mode disables the development demo. Use local development settings and restart.
- **Provider failure:** use the safe configuration check above; verify quota, credential validity, network access, and model availability. The development fallback is labelled when selected.
- **Mic permission denied:** grant page permission, check the OS input device, and reload. Use Chrome/Edge on `127.0.0.1` or localhost.
- **Audio rejected:** use decodable WAV/MP3/OGG/WebM under 10 MiB and three minutes. STT providers may enforce smaller limits.
- **Emotion says `text_proxy_fallback`:** that is the real lexical path, not acoustic inference. Use WAV and check the acoustic model status; first load may need internet and memory.
- **Slow first response:** first diagnostics imports ML libraries; first acoustic inference may download/load checkpoint weights. Warm up before presenting.
- **Readiness returns 503:** `/ready` checks PostgreSQL/Redis; the voice demo may still pass `/health/live` without optional services.
- **Barge-in does not trigger:** use headphones, grant mic permission, speak clearly, and verify the selected device. The event log states whether backend cancellation succeeded.

## Known limits

- Voice sessions are in memory and reset when the backend restarts.
- Real browser microphone capture and human interruption still need a manual check.
- Acoustic confidence is uncalibrated; this project has no labelled audio set for SER accuracy.
- Historical SOVA runs differ in data/protocol, so no exact paper reproduction is claimed.
