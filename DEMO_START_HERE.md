# Emotion Aware Voice Assistant — Start Here

1. Put your own `.env` in this folder. Do not publish it.
2. Run `powershell -ExecutionPolicy Bypass -File .\REPAIR_AND_START_VOICE.ps1`.
3. Open `http://127.0.0.1:8000/`.
4. Click **Start Aiko**.
5. Check **Refresh diagnostics**. The page shows whether Groq STT/LLM, ElevenLabs TTS, acoustic SER, and the development fallback are available.

The demo has recovery paths: missing LLM provider -> clearly-labelled local demo reply; missing TTS -> browser speech synthesis; missing Groq STT -> browser speech recognition (Chrome/Edge). These fallbacks are for the local demonstration only and must not be reported as experimental model results.
