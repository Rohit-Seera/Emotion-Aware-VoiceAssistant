# Emotion Aware Voice Assistant — SOVA Evaluation Pack

This clean pack contains the evaluation code, calculation formulas, selected input manifests, required audio clips, prior 1,000-utterance LibriSpeech ASR evidence, and final tables. No source project files were overwritten.

## Show these files to the supervisor

- `outputs/SOVA_FINAL_RESULTS_AND_LIMITATIONS.md` — readable results and protocol notes.
- `outputs/SOVA_STYLE_COMPARISON_TABLE.csv` — SOVA Table 4-style comparison; published model values are identified as paper references.
- `outputs/SOVA_DATASET_COVERAGE_CATALOG.csv` — dataset availability and sample sizes.
- `outputs/SOVA_METRIC_FORMULAS.csv` — the metric equations and denominators.
- `outputs/task_summary.json` — aggregate machine-readable metrics.
- `outputs/FINAL_SELECTED_CASE_RESULTS.jsonl` — one final row per selected case across four task families.
- `outputs/task_case_results.jsonl` — raw case outputs including retries and failures.
- `outputs/librispeech_asr_1000_case_results.jsonl` — all 1,000 LibriSpeech transcriptions used for WER.
- `outputs/FINAL_EVALUATION_PROVENANCE.json` — sample definitions and SHA-256 hashes.

## Evaluation size

The final count is **1,220 benchmark cases**: 1,000 stratified LibriSpeech ASR utterances, plus 55 questions/prompts each from TriviaQA, LibriSQA, Spoken SQuAD, and Alpaca. The four 55-case subsets include the earlier 25-case pilot; successful responses were reused, and new cases were added with fixed seeds.

The full TriviaQA, LibriSQA, and Alpaca files are included in `inputs/`. The complete Spoken SQuAD source (5,351 rows) is available through `inputs/spoken_squad_full_source/`, and the complete LibriSpeech source is available through `inputs/librispeech_source/`. These are directory junctions to the preserved dataset archive, so the large datasets are accessible here without making duplicate copies. The 55 Spoken SQuAD evaluation WAVs and checksum manifests are stored directly in this pack.

## Code

- `code/prepare_expanded_manifests.py` — creates the fixed-seed 55-case manifests.
- `code/run_evaluation.py` — runs the selected cases through the existing authenticated assistant backend and saves each result.
- `code/build_final_results.py` — calculates the SOVA-style table, coverage catalog, formulas, and provenance file.
- `source_backup/sova_metrics.py` — WER, normalization, and percentage calculations used by the evaluation.

The raw SOVA paper scores in the comparison table are published references; the paper's baseline models were not rerun. Exact match, token-F1, and ROUGE-L are proxy metrics, not the paper's GPT-evaluator scores. IEMOCAP was unavailable, and Common Voice age/gender plus Alpaca audio-quality metrics were not measured for this assistant.
