# Emotion-Aware Voice Assistant — Evaluation Results

## Evaluation summary

The evaluation covers 1,220 cases: 1,000 stratified LibriSpeech ASR utterances and 55 selected cases from each of TriviaQA, LibriSQA Part II, Spoken SQuAD, and Alpaca.

| Task / dataset | Result | Coverage | Metric |
|---|---:|---:|---|
| LibriSpeech ASR | 3.55% corpus WER | 1,000/1,000 | Corpus micro-WER |
| TriviaQA | 0.00% exact-match proxy; 35.58% token-F1 | 55/55 responses | Lexical proxies |
| LibriSQA Part II | 74.55% accuracy; 78.85% conditioned accuracy | 55 cases; 52 valid options | Four-choice accuracy |
| Spoken SQuAD | 0.00% exact-match proxy; 25.54% token-F1 | 55/55 responses | Lexical proxies |
| Alpaca | 20.93% ROUGE-L F1; 26.66% token-F1 | 55/55 responses | Text-generation lexical proxies |

## LibriSpeech ASR breakdown

The ASR evaluation used Groq Whisper large-v3-turbo on a fixed-seed stratified sample of 1,000 clips, with 250 from each official split.

| Split | Cases | Corpus WER |
|---|---:|---:|
| dev-clean | 250 | 2.75% |
| dev-other | 250 | 4.79% |
| test-clean | 250 | 2.51% |
| test-other | 250 | 4.52% |
| Overall | 1,000 | 3.5525% |

The overall corpus WER is calculated as (substitutions + deletions + insertions) divided by reference words: 656 errors / 18,466 reference words. The separately reported macro WER is 5.3213%. All 1,000 selected transcription requests completed successfully. Mean request latency was 0.348 seconds, median 0.332 seconds, and p95 0.479 seconds (successful requests only).

## Task interpretation

- TriviaQA: all 55 backend responses were returned. The exact-match proxy was 0%; mean token-F1 was 35.58%.
- LibriSQA Part II: 41 of 55 selected cases were correct (74.55%). There were 52 valid options; accuracy conditioned on valid options was 41/52 (78.85%). The answer rate was 52/55 (94.55%).
- Spoken SQuAD: all 55 responses were returned. Audio was transcribed by Whisper, then the LLM answered from the transcript. Exact-match proxy was 0%; mean token-F1 was 25.54%.
- Alpaca: the evaluation was text-only. Mean ROUGE-L F1 was 20.93%, and mean token-F1 was 26.66%.

## Protocol limitations

- This is a limited fixed-seed sample evaluation, not a full reproduction of SOVA-Bench.
- Published Mini-Omni, Mini-Omni2, Moshi, LLaMA-Omni, Freeze-Omni, GLM-4-Voice, and Cascade values in the comparison table are published paper references. Their checkpoints were not rerun locally.
- Exact-match, token-F1, and ROUGE-L are transparent lexical proxies; they are not the paper's GPT-evaluator accuracy or GPTScore.
- IEMOCAP emotion accuracy was not measured because the dataset was unavailable for this evaluation.
- Common Voice age and gender metrics were not measured; age and gender are not outputs of this assistant.
- Alpaca speech-consistency WER and UTMOSv2 audio quality were not measured.
- LibriSpeech WER evaluates speech recognition only, not emotion recognition or end-to-end dialogue.
- No foundation-model fine-tuning run is claimed.

Raw per-case transcripts, selected source manifests, dataset audio, and full source datasets are not included in the public repository. Aggregate summaries, formulas, and comparison tables are included for review.