# Emotion-Aware Voice Assistant — Evaluation

This directory contains the evaluation scripts, aggregate result files, metric definitions, and protocol limitations for the assistant's SOVA-Bench-compatible task-family evaluation.

## Aggregate result files

- outputs/SOVA_FINAL_RESULTS_AND_LIMITATIONS.md — readable results, split-level ASR scores, and limitations.
- outputs/SOVA_STYLE_COMPARISON_TABLE.csv — comparison with published SOVA-Bench reference values and local measurements.
- outputs/SOVA_DATASET_COVERAGE_CATALOG.csv — evaluated tasks, sample sizes, and public-data availability.
- outputs/SOVA_METRIC_FORMULAS.csv — metric calculations and denominator notes.
- outputs/task_summary.json — aggregate metrics for the four 55-case task samples.
- outputs/FINAL_EVALUATION_PROVENANCE.json — model, sample, and publication-scope metadata.
- inputs/asr_librispeech_summary_1000.json — aggregate summary for the 1,000-utterance LibriSpeech ASR run.

## Evaluation coverage

The reported total is 1,220 cases:

- LibriSpeech ASR: 1,000 utterances, sampled equally from dev-clean, dev-other, test-clean, and test-other (250 per split).
- TriviaQA: 55 text questions.
- LibriSQA Part II: 55 questions, answered using saved ASR transcripts.
- Spoken SQuAD: 55 audio questions, processed with Whisper transcription followed by LLM answering.
- Alpaca: 55 text prompts.

## Important interpretation notes

This is a limited, fixed-seed evaluation, not a full reproduction of SOVA-Bench. The published Mini-Omni, Mini-Omni2, Moshi, LLaMA-Omni, Freeze-Omni, GLM-4-Voice, and Cascade values are paper references; those baseline checkpoints were not rerun locally.

Exact-match, token-F1, and ROUGE-L are lexical proxy metrics and are not equivalent to the paper's GPT-evaluated accuracy or GPTScore. LibriSpeech WER measures speech recognition only. IEMOCAP emotion accuracy, Common Voice age/gender metrics, Alpaca speech-consistency WER, and UTMOSv2 audio quality were not measured. No foundation-model fine-tuning run is claimed.

## Data and privacy scope

The public repository contains aggregate reports only. Full dataset files, raw per-case output JSONL, selected source manifests, evaluation WAV files, and model weights are not published here. Provider credentials belong in a local .env file; never commit that file.

## Evaluation code

- code/prepare_expanded_manifests.py — prepares fixed-seed task samples.
- code/run_evaluation.py — runs selected cases through the configured assistant backend.
- code/build_final_results.py — calculates aggregate tables and metrics.
- source_backup/sova_metrics.py — metric calculation utilities.