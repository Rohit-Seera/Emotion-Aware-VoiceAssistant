# Metrics and Formulas

The project contains a runnable SOVA-inspired scorer, historical local runs, and paper-reference data. No stored local result is claimed as an exact SOVA-Bench reproduction.

## Classification and answer metrics

For (N) target examples, let (C) be correct answers and (A) be successfully answered examples.

- Accuracy: `100 × C / N`. Four correct of five targets gives 80%. When an expected count is supplied, failed or missing cases stay in the denominator.
- Answer rate (AR): `100 × A / N`. Four answers from five targets gives 80%.
- Conditional accuracy (CA): `100 × C / A`. Three correct among four answered cases gives 75%.
- Execution coverage: `100 × successful executions / expected targets`. This measures completion, not answer quality.
- Emotion label accuracy: `100 × exactly matched reference labels / labelled targets`. It requires ground-truth labels and a declared label mapping; confidence cannot calculate it.

Implemented in `benchmark/sova_metrics.py` (`accuracy`, `answer_rate`, `conditioned_accuracy`, `exact_label_match`) and aggregated by `benchmark/score_sova_results.py::score_rows`. Legacy matching in `benchmark/metrics.py` is a separate token-boundary alias matcher; cite the method recorded in the result.

## Word error rate

For a reference with (N) words, minimum edit alignment counts substitutions (S), deletions (D), and insertions (I):

`WER = (S + D + I) / N`; percent form is `100 × (S + D + I) / N`.

Example: reference `the cat sat`, hypothesis `the dog sat` has one substitution, zero deletions, zero insertions, and three reference words: 1/3 = 33.33%. WER can exceed 100% because insertions add errors: reference `yes`, hypothesis `no maybe certainly` has one substitution and two insertions, so 3/1 = 300%.

The maintained alignment is in `benchmark/sova_metrics.py::wer_stats`; `wer` and `corpus_wer` report per-item and corpus/micro summaries. Corpus WER sums errors before division; macro WER averages per-utterance percentages. They differ. Historical `benchmark/sova_librispeech_500.py::calculate_wer` reports a mean of per-item WER values with simpler lowercase/whitespace normalization. Its `mean_wer` is a fraction; multiply by 100 for percent. Do not call it corpus WER.

STT WER compares incoming speech to a human transcript. Generation-consistency WER compares assistant text to a separate ASR transcript of generated speech. The latter measures text-to-audio consistency, not incoming-speech recognition.

## Precision, recall, and F1

For a class, `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, and `F1 = 2×precision×recall/(precision+recall)`. TP is a correct positive prediction, FP an incorrect positive prediction, and FN a missed positive. These metrics are not calculated by the current SOVA scorer, which reports exact-label accuracy and answer metrics. Do not claim F1 from these files.

## Semantic and acoustic quality

- The custom semantic pilot averages four 1–5 rubric scores (correctness, relevance, coherence, instruction following); normalized value is `100 × mean_score / 5`. See `benchmark/semantic_eval.py` and `benchmark/results_generation/semantic_summary.json`. This is not the paper's exact GPTScore protocol and is not proof of factual correctness.
- UTMOSv2 is a pretrained predictor of perceived naturalness on a MOS-like 1–5 scale. It is not human-listener MOS and is not accuracy. Mean and per-example values are in `benchmark/results_generation/acoustic_summary.json`, `acoustic_results.csv`, and `acoustic_results.json`.
- `100 × score / 5` is only rescaling; use the raw 1–5 UTMOSv2 value in a viva.

## Emotion confidence

Wav2Vec2/SUPERB returns softmax scores for checkpoint classes. A score of 0.80 means that class received 0.80 of the normalized model score for that input; it does not mean 80% empirically correct. Calibration and accuracy require a labelled held-out audio set. Keyword confidence values are fixed rule scores, not probabilities. Fusion is implemented in `app/engines/emotion/hybrid.py`; lexical labels are in `app/engines/emotion/engine.py`.

## Latency and reliability

- API text `processing_ms` is `(perf_counter_end - perf_counter_start) × 1000` around the runtime in `app/services/speech/service.py::VoiceCompanionService.process_turn`.
- Audio UI time uses browser `performance.now()` from request start through playback completion. It includes network and playback and is not LLM-only latency.
- Reliability uses all intended cases as the denominator. Missing and failed calls remain failures. The stored Spoken SQuAD run has 16 successes and 484 failures out of 500 targets.
- Barge-in playback-stop latency is browser measured from speech-candidate detection to playback pause. Cancellation result is reported separately. Neither is a SOVA Table 4 metric.

## Stored measurements and scope

| Artifact | Observed result | Correct interpretation |
|---|---:|---|
| `benchmark/results_sova_librispeech_500/summary.json` | 500/500 success; mean WER 0.146259 (14.63%); mean latency 3834 ms | Historical Groq Whisper `whisper-large-v3-turbo`; mean utterance WER, not corpus WER. Stored config `clean`, split `test`. |
| `benchmark/results_sova_triviaqa_500/summary.json` | 500 targets; 28.0% accuracy; 40.2% answer rate; 69.65% conditional accuracy; 251 failures | Historical Groq GPT-OSS 120B text pilot, not the paper's synthesized spoken questions. |
| `benchmark/results_sova_triviaqa_test_v2/summary.json` | 100 targets; 49.0% accuracy; 82.0% answer rate; 59.76% conditional accuracy; 2 failures | Historical text pilot; not a full SOVA result. |
| `benchmark/results_sova_spoken_squad_500/summary.json` | 16/500 success (3.2% coverage); stored accuracy 50%; 484 failures | Partial run; 50% applies to the saved answered subset, not a 500-case completion. |
| `benchmark/audit/pilot_150/summary.json` | 150 custom text examples; emotion task has 50 cases and 94% exact-label accuracy | Not audio SER accuracy and not a SOVA test-set result. |
| `benchmark/results_generation/semantic_summary.json` | 20/20; custom semantic mean 5.0/5 | Custom rubric; not paper GPTScore or factuality proof. |
| `benchmark/results_generation/acoustic_summary.json` | 20/20; mean UTMOSv2 2.677/5 | Historical generated-audio predictions; not human listening scores. |
| `results/live_e2e_smoke.md` | Current live request evidence | Single local integration smoke test, not a benchmark score. |

Paper values are separately stored in `benchmark/paper_reference_table4.json`. For example, Mini-Omni's published reference row gives test-clean WER 8.68% and UTMOSv2 3.62. Those are not Weings measurements. The local Whisper and generated-audio pilots do not share systems, samples, or protocols, so the values do not establish a ranking.

The six open SOVA speech-model adapters are listed in `benchmark/sova_model_registry.json`. Official weights, compatible environments, or bridge commands are not configured here. An adapter in code does not mean that model was evaluated. See `docs/SOVA_BENCH_AUDIT.md` and `benchmark/runs/README.md`.
