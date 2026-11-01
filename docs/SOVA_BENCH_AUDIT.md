# SOVA-Bench implementation audit for Emotion Aware Voice Assistant

**Scope:** static review of the supplied SOVA-Bench preprint and the supplied `testing.zip`, plus local scoring-tool changes. This file distinguishes completed code from experiments that still need to be run.

## What was fixed in the benchmark layer

1. `benchmark/score_sova_results.py` now groups results by **model + task** rather than pooling models into one task score.
2. Accuracy, answer rate, and conditioned accuracy are computed from successful outputs with explicit denominators. Failed rows count as failed attempts; a declared run target can also count missing result rows as failures/unanswered.
3. Unrun SOVA metric rows are retained with status `not_run` instead of disappearing from a summary.
4. Both corpus/micro WER and macro utterance WER are reported. The legacy `wer` values are retained as a separate diagnostic when present because the old LibriSpeech script tokenized by lowercase+whitespace only, while the normalized scorer strips punctuation/normalizes spacing.
5. The custom C/R/H/I Likert rubric is preserved as a custom semantic score. It is **not** relabelled as the paper's GPTScore.
6. `benchmark/merge_sova_generation_results.py` merges the four stored generation result streams by case ID, without combining metrics with incompatible scales.
7. `benchmark/paper_reference_table4.json` records the Table 4 values from the supplied preprint, while `benchmark/compare_sova_reference.py` exports a side-by-side view without silently asserting that non-matching local results are comparable.
8. Eighteen automated benchmark tests cover WER, model separation, failed/missing cases, generation metrics, adapter parsing, reference-vs-measurement separation, and task visibility.

## Audit of result files already in the ZIP

These are diagnostics of the **saved ZIP files**, not new runs against the official full benchmark.

| Saved run | Observed result | Interpretation |
|---|---|---|
| `benchmark/results/results.json` | 150 total records: 50 knowledge, 50 emotion, 50 instruction; knowledge 98% and emotion 94% in the saved text pilot | This is text-only pilot/classification work. The emotion figure is not acoustic SER accuracy and must not be presented as IEMOCAP/SOVA-Bench emotion accuracy. |
| `benchmark/results_sova_triviaqa_500/results.json` | 500 records, 249 successful and 251 failed; scorer reports 49.8% execution coverage, 28.0% all-target accuracy using the saved correctness flags, and 69.65% conditioned accuracy among successfully answered rows | Questions are text prompts rather than the paper's synthesized speech inputs, and correctness flags are not verified as the paper's GPT-evaluator outputs. This is a partial text QA pipeline diagnostic, not SOVA knowledge accuracy. |
| `benchmark/results_sova_librispeech_500/results.json` | 500 successful records; normalized corpus WER recomputed from saved reference/hypothesis text is about 2.77%; the source's saved per-utterance WER averages about 14.63% | The discrepancy is driven by normalization: the original runner treats punctuation differences as token errors; the current scorer normalizes punctuation. The paper's short protocol description does not specify enough to assert which local normalization exactly matches author-side code. State the calculation and do not mix the two values in one table. This is also a direct Groq Whisper run, not the entire Weings voice path. |
| `benchmark/results_sova_spoken_squad_500/summary.json` and `results.json` | 16 result rows against a run target of 500; summary says 484 failures; among the 16 returned records, stored correctness is 50% | In the repaired scoring command, passing `--expected-count spoken_qa_open_ended=500` reports 3.2% execution coverage, 50% conditioned accuracy on the 16 returned answers, and 1.6% all-target accuracy if the 16 stored correctness flags are used. The latter is **not** the paper's GPT-judged score. |
| `benchmark/results_generation/*` | 20 custom prompt cases, with text responses, ASR transcripts, custom C/R/H/I judgements, and UTMOSv2 scores | Useful pilot for pipeline debugging. Twenty custom prompts are not the 3,745 cleaned Alpaca cases in the paper. C/R/H/I scores are not GPTScore. |
| `benchmark/datasets/sova_triviaqa_6840.json` | 6,840 text QA records are present | The records are text questions/answers; they do not themselves reproduce the paper's synthesized speech-input pipeline. |

## Baseline model runner status

`benchmark/sova_model_registry.json` now registers all seven Table 3 baselines and the Emotion Aware Voice Assistant adapter. Each baseline is wired to a repeatable API connector, official-inference CLI, or JSONL bridge wrapper. This is implementation plumbing; it does not mean the baseline checkpoints have been downloaded or that every model has completed every task. `benchmark/run_sova_model.py` writes only actual run outputs, while `benchmark/paper_reference_table4.json` stores the paper values separately. Native external model execution still requires the model's official environment/weights or configured API keys and an audio-backed benchmark manifest.

The added `benchmark/attach_sova_metrics.py` attaches UTMOSv2 results as separate `generation_acoustic` rows. `transcribe_sova_outputs.py` now defaults to generation-task outputs rather than calculating generation consistency from spoken-QA answer audio. `acoustic_eval.py --generation-only` similarly avoids scoring speech responses from non-generation tasks.

## How to run the repaired analysis

From the project root in PowerShell:

```powershell
# 1. Static metric tests (no API key required)
python -m pytest tests/benchmark -q

# 2. Audit the saved TriviaQA subset and count failed attempts
python -m benchmark.score_sova_results `
  --input benchmark/results_sova_triviaqa_500/results.json `
  --output benchmark/audit/triviaqa_500 `
  --model-name "Groq GPT-OSS 120B" --provider groq --model "openai/gpt-oss-120b" `
  --expected-count knowledge_qa=500

# 3. Audit the saved 500-record LibriSpeech run
python -m benchmark.score_sova_results `
  --input benchmark/results_sova_librispeech_500/results.json `
  --output benchmark/audit/librispeech_direct_whisper `
  --model-name "Direct Whisper STT baseline" `
  --provider groq --model "whisper-large-v3-turbo" `
  --expected-count speech_recognition=500

# 4. Audit Spoken SQuAD and count missing cases
python -m benchmark.score_sova_results `
  --input benchmark/results_sova_spoken_squad_500/results.json `
  --output benchmark/audit/spoken_squad `
  --model-name "Whisper STT + GPT-OSS pipeline" `
  --provider groq --model "openai/gpt-oss-120b" `
  --expected-count spoken_qa_open_ended=500

# 5. Merge saved text/TTS/ASR/custom-judge/UTMOSv2 generation runs and score them
python -m benchmark.merge_sova_generation_results
python -m benchmark.score_sova_results `
  --input benchmark/results_generation/sova_generation_merged.json `
  --output benchmark/results_generation/scored_sova

# 6. Export a reference table without claiming unsupported deltas
python -m benchmark.compare_sova_reference `
  --local-summary benchmark/audit/librispeech_direct_whisper/summary.json `
  --output benchmark/audit/local_vs_paper_reference.csv
```

`--expected-count TASK=COUNT` is repeatable. Use it when the raw result file omits failed/unrecorded attempts, as the saved Spoken SQuAD result does. When the input file contains explicit failure rows already, do not count the same failures a second time.

## Research contribution: what is genuinely new in the current Weings prototype?

The uploaded code positions Weings as an **interaction-system contribution inspired by SOVA-Bench**, not as a newly trained speech foundation model and not as a complete reproduction of the SOVA benchmark. The potentially defensible contributions are:

- **Hybrid emotion-informed response control:** combine the optional acoustic SER output with transcript/lexical cues; expose confidence/source; pass the fused estimate to an explicit response policy.
- **Emotion-conditioned spoken delivery:** derive a response/voice style and pass request-level delivery controls to TTS rather than changing only the reply text.
- **Interruptible streaming dialogue:** browser voice activity detection, playback stop, cancellation of the active backend turn, then capture/process the new user turn.
- **Observable interaction diagnostics:** per-session interruption count and measured VAD-to-playback-stop latency, with emotion source and confidence visible in the demo/API.

These are system-design and integration contributions. The current emotion checkpoint is public pretrained SUPERB/Wav2Vec2, and the current 150-case emotion result is text-based. **Do not claim measured improvement over SOVA-Bench Table 4 yet.**

## Experiment needed before claiming improvement

Use exactly the same held-out audio, LLM, TTS voice, generation settings, and scoring procedure across these ablations:

| Variant | Change relative to previous variant |
|---|---|
| A0 — baseline | Emotion policy disabled; same LLM and TTS path |
| A1 — acoustic SER | Add audio emotion prediction |
| A2 — hybrid emotion | Fuse acoustic emotion with transcript cues; enable emotion-aware response policy and delivery style |
| A3 — hybrid + barge-in | Add VAD-triggered playback stop and backend cancellation |

Report per-variant acoustic emotion accuracy / macro-F1 on licensed labelled speech, answer rate and conditioned accuracy, a declared response-appropriateness rubric (do not call it GPTScore unless the GPTScore protocol is implemented), raw UTMOSv2, text-audio consistency WER with fixed normalization, interruption success rate, and median/p95 interruption latency. Use paired statistical analysis or confidence intervals. Report sample counts and failures next to every result.

Only after that controlled experiment should the paper say the Weings method improves a particular metric. At present, the contribution is implemented as a prototype and the saved metrics are pilot diagnostics.
