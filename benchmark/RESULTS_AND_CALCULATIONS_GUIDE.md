# SOVA evaluation: calculations, outputs, and provenance

This file is an operator's guide for the code in this folder. The SOVA paper's Table 2/4 metric names and counts are the reference protocol. The code is a compatible implementation; it does not claim byte-identical reproduction of unpublished author-side prompts or filtering.

## Quick Windows setup (offline; no model/API call)

From the extracted project root, with the project's virtual environment activated:

```powershell
powershell -ExecutionPolicy Bypass -File benchmark\run_sova_setup.ps1
```

It runs the benchmark unit tests, lists adapter IDs, checks available benchmark assets and writes a Table-4-shaped CSV. It does not load a speech model, call Groq/ElevenLabs, download datasets, or train anything.

## SOVA paper task matrix to reproduce

The supplied preprint's Table 2 gives these target instance counts and metric families:

| Table-4 metric row | Dataset | Paper instances/splits | Metric |
|---|---|---:|---|
| Knowledge accuracy | TriviaQA unfiltered dev | 6,840 | GPT-evaluated accuracy |
| ASR WER | LibriSpeech | dev-clean, dev-other, test-clean, test-other | Corpus WER per split |
| LibriSQA | LibriSQA Part II | 2,619 | Accuracy |
| Spoken SQuAD | Spoken SQuAD | 3,243 | GPT-evaluated accuracy |
| Emotion | IEMOCAP | 5,819 | Accuracy, plus AR/CA analysis |
| Gender | Common Voice speaker metadata | 2,469 | Accuracy, plus AR/CA analysis |
| Age | Common Voice speaker metadata | 2,511 | Accuracy, plus AR/CA analysis |
| Generation consistency | Cleaned Alpaca subset | 3,745 | WER between intended response text and ASR transcript of generated speech |
| Generation semantic | Cleaned Alpaca subset | 3,745 | GPTScore |
| Generation acoustic | Cleaned Alpaca subset | 3,745 | Raw UTMOSv2 score |

The CSV exporter uses the same key metric columns as Table 4 and adds run-provenance fields (`source_type`, `measurement_status`, `n_cases`, `n_success`, and `metric_method_notes`). The initial generated table contains seven published paper reference rows plus one blank Emotional Voice Assistant placeholder. No local score is invented.

## Data needed before inference

The ZIP contains text-only TriviaQA records and prior generated-output examples, but does **not** bundle the source audio for SOVA's full evaluation sets or the official baseline checkpoints. A prior LibriSpeech manifest contains absolute paths from a specific Windows machine and must be regenerated if those paths do not exist here.

### Prepare a small LibriSpeech smoke manifest

Obtain/extract LibriSpeech separately, then run:

```powershell
python -m benchmark.prepare_librispeech_manifest --root "D:\datasets\LibriSpeech" --splits test-clean --limit-per-split 5 --output benchmark\manifests\librispeech_smoke.json
```

If the LibriSpeech root already contains the `test-clean` folder, use that root. The script reads each chapter's `*.trans.txt`, attaches an existing audio file, and saves absolute audio paths plus reference transcripts. No model is invoked by this preparation step.

Validate the manifest before spending GPU/API time:

```powershell
python -m benchmark.check_sova_setup --manifest benchmark\manifests\librispeech_smoke.json --strict-manifest --output benchmark\audit\manifest_check.json
```

### Run one configured model (pilot limit defaults to 5)

```powershell
powershell -ExecutionPolicy Bypass -File benchmark\run_sova_smoke.ps1 -Model mini_omni -Manifest benchmark\manifests\librispeech_smoke.json
```

The upstream model repo, compatible Python/CUDA environment, and checkpoint must already be set up. `mini_omni` is only an example model ID; the runner intentionally does not pretend that a checkpoint is included. Use `python -m benchmark.run_sova_model --model all --list-models` to see all registered IDs and their requirements.

To run the full cases in a manifest, pass `-AllCases`. Do this only after a pilot passes and you have checked cost/runtime. QA tasks need a judging step; the `-Judge` switch uses the configured judge model (the current README default is GPT-OSS 120B via Groq). Do not run 6,840 examples before checking five cases.

## Calculations: exact code locations

- `benchmark/sova_metrics.py`
  - `accuracy(correct, total)`: `100 * correct / total`.
  - `answer_rate(answered, total)`: `100 * answered / total`.
  - `conditioned_accuracy(correct, answered)`: `100 * correct / answered`.
  - `wer_stats(reference, hypothesis)`: minimum-edit-distance counts for substitutions (`S`), deletions (`D`), and insertions (`I`).
  - `corpus_wer(pairs)`: aggregate S/D/I/reference words across utterances first (micro/corpus WER) and also report macro utterance WER.
  - Exact label comparison normalizes Unicode NFKC and case; it does not count a substring as the correct class.
- `benchmark/score_sova_results.py`
  - `_task_summary()` groups cases by task and calculates accuracy, answer rate, conditioned accuracy, execution coverage, and WER; it keeps no-result tasks visible.
  - `score_rows()` groups each model separately, so predictions from different models are not pooled.
  - For expected-count runs, missing records count against coverage and all-case accuracy instead of disappearing.
  - `main()` writes the outputs listed below.
- `benchmark/compare_sova_table4.py`
  - `aggregate_model()` computes the Table-4 cells from raw measured records and their saved labels/transcripts/scores.
  - `build_table()` imports the fixed published Table 4 reference values and appends measured local-run rows. The reference and measured rows have different `source_type` values.
  - Missing/uncomputed metrics remain blank; the tool does not turn absent values into zero and does not convert custom C/R/H/I ratings into GPTScore.
- `benchmark/judge_sova_outputs.py`
  - Uses the configured judge model to add correctness/GPTScore-like judgements to appropriate text outputs. The paper identifies GPT evaluation but the provided preprint does not include all judge prompt details. Save the judge model and prompt/protocol used in your methods section.

## Output paths

A fresh model run writes:

- `benchmark/runs/<model_id>/results.json` — raw per-example predictions, errors, latency, emotion output where supplied, and output-audio path.
- `benchmark/runs/<model_id>/summary.json` — attempted/success/failed counts and execution coverage emitted by the runner.
- `benchmark/runs/<model_id>/scored/summary.json` — task metrics from `score_sova_results.py`.
- `benchmark/runs/<model_id>/scored/scored_cases.json` — normalized, case-level fields used in calculations.
- `benchmark/runs/<model_id>/scored/task_summary.csv` — one row per model/task with formulas, denominators, coverage, and caveats.
- `benchmark/Table4_comparison.csv` — SOVA Table-4-shaped columns, including published reference rows and a blank proposed-model placeholder until actual run results exist. After inference, measured rows replace placeholders and contain their measured-case counts.
- `benchmark/sova_reference_comparison.csv` — a supplementary task-wise local/reference export. It deliberately avoids computing improvement deltas unless comparability is established.
- `benchmark/audit/setup_check.json` — environment/data inventory; `benchmark/audit/existing_results_inventory.csv` describes historical saved pilot outputs in this ZIP.

The checked-in `benchmark/results_*` and `benchmark/audit/*` files are historical experiments from before the current common adapter runner. Do not automatically combine them into one Table-4 row: e.g., direct Groq Whisper ASR, GPT-OSS TriviaQA, and a partial Spoken SQuAD run are different systems/protocols. The setup script preserves them and labels their limitations.

## Which source data are present?

- `benchmark/datasets/sova_triviaqa_6840.json` and `benchmark/sova_official/triviaqa_6840.jsonl`: TriviaQA question/answer text records. SOVA's paper turns text-only TriviaQA questions into speech before evaluating speech-input models. These files do not themselves contain audio.
- `benchmark/manifests/librispeech_test_clean_5.json`: a tiny machine-specific manifest sample. Recreate it with `prepare_librispeech_manifest.py`; source audio is not bundled.
- `benchmark/results_generation/audio*`: prior generated assistant outputs, not the SOVA source audio for the Alpaca test inputs.
- IEMOCAP, Common Voice audio, full LibriSQA/Spoken SQuAD processing, and CosyVoice-synthesized TriviaQA/Alpaca audio must be sourced/prepared separately. IEMOCAP has its own access and license conditions.
- `benchmark/paper_reference_table4.json`: values transcribed from SOVA Table 4. They are reference data, never measured results from our assistant.

## Experimental integrity rules

1. Use the same fixed held-out evaluation examples for compared systems. Do not train on examples that remain in the test set.
2. Report smoke/subset runs as subsets; don't label five or 500 samples as the paper's full dataset result.
3. Knowledge and open-ended QA need a declared judge protocol; don't treat keyword overlap or the C/R/H/I custom rubric as the paper's GPTScore.
4. Emotion/age/gender must receive the paper-required audio and label protocol. The app's 4-class pretrained SUPERB emotion model is not automatically equivalent to SOVA's 5-choice IEMOCAP task.
5. UTMOSv2 is a raw naturalness score, not percentage accuracy. Keep the score scale.
6. Barge-in latency, false trigger rate, and cancel success rate are useful extra system metrics, but they are not Table 4 columns in the provided paper.
