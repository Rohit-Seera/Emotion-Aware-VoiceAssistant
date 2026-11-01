# Emotion Aware Voice Assistant SOVA-inspired Speech Generation Evaluation

1. Generate text responses for the 20 generation cases:
   `python -m benchmark --dataset benchmark/datasets/weings_sova_generation_20.json --output benchmark/results_generation`

2. Run TTS -> STT consistency evaluation:
   `python -m benchmark.audio_eval --input benchmark/results_generation/results.json --output benchmark/results_generation`

The evaluator uses the configured ElevenLabs voice for TTS and Groq Whisper Large V3 Turbo for transcription. The LLM response is the reference text and the Whisper transcription of the generated speech is the hypothesis.

Consistency WER:
`WER = (S + D + I) / N * 100`

Lower WER indicates closer text-speech consistency.

The generated MP3 files are saved under `audio/` for reproducibility. Audio evaluation is intentionally separate from the text benchmark so API cost and rate limits can be controlled.
