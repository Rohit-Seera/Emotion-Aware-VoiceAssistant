# SOVA-Bench implementation status — Emotion Aware Voice Assistant

This document is intended for research-review discussion. It separates executable implementation from reference numbers and external setup still required.

## Model coverage

`benchmark/sova_model_registry.json` and `benchmark/run_sova_model.py` register the six open-source speech models from the supplied SOVA-Bench Table 3 plus Emotion Aware Voice Assistant. Cascade is excluded from live runs because it uses paid GPT-4o-mini; its published values remain reference-only:

| System | Executable adapter in this project | What must still be supplied/configured locally |
|---|---|---|
| Mini-Omni | Official `OmniInference` / `A1_A2` JSONL batch bridge | Official checkout, checkpoint (`lit_model.pth`), supported CUDA/Python environment |
| Mini-Omni2 | Native `/chat` multipart HTTP adapter (native text and audio parsed separately) | Official server, checkpoint, supported runtime |
| Moshi | Official PyTorch `LMGen` + Mimi JSONL batch bridge | Official Moshi environment/checkpoint and sufficient GPU memory |
| LLaMA-Omni | Official `omni_speech/infer/infer.py` batch invocation and optional official vocoder conversion | Official checkout, model, Whisper assets, vocoder and dependencies |
| Freeze-Omni | Official `bin/inference.py` batch wrapper | Official checkout, Freeze-Omni and Qwen checkpoints, compatible runtime |
| GLM-4-Voice | Official tokenizer, model-server generation, and decoder wrapper | Official checkout, tokenizer/model/decoder assets, running server |
| Emotion Aware Voice Assistant | Native authenticated adapter to the project's `audio-stream-turn` API | Running/configured backend with STT, LLM, emotion model and TTS actually enabled |

Weights and credentials are not included. The wrappers intentionally invoke each open model in its own environment instead of merging incompatible requirements into Weings' application environment.

## Metrics implemented

| Table 4 column | How the runner derives it |
|---|---|
| Knowledge / spoken-QA / paralinguistic accuracy | Correct cases divided by total target records; failed or unanswered target cases stay in the denominator. QA and the paper's categorical tasks can use the explicitly recorded GPT-OSS 120B judge step via the user's existing Groq access; this is an operationalization, not proof of the authors' undisclosed exact GPTEval prompt/model. |
| ASR WER per LibriSpeech split | Corpus WER: `100 × (S + D + I) / reference_words`, computed separately for `dev-clean`, `dev-other`, `test-clean`, and `test-other` when those splits are present. |
| Generation text-audio consistency | Same metric formula as WER, comparing the model's own text response against a separate ASR transcript of its generated speech. The audio transcript is never copied into the model-text field to force a 0% WER. |
| Generation semantic score | Average explicit `gptscore` values produced by `judge_sova_outputs.py`; prompt version is recorded. This is a declared 1–5 operationalization, not a claim that the paper's unpublished judging prompt is identical. |
| Generation acoustic score | Average raw UTMOSv2 MOS values from successful generated WAVs only. Scores are attached as independent `generation_acoustic` rows. |
| Answer rate / conditioned accuracy | Also supported by the legacy scoring report for paralinguistic tasks: answered / total and correct / answered. The Table-4-shaped CSV contains the Table 4 columns. |

`benchmark/compare_sova_table4.py` exports Table-4-shaped CSV columns. `source_type=published_paper_table4_reference` denotes values transcribed from the supplied paper; `source_type=measured_local_run` denotes outputs recorded from this runner. These are separate rows so paper values cannot be mistaken for our measurements. Empty cells mean not run, unavailable, or unsupported in Table 4—not zero.

## Required run sequence

1. Prepare an audio-backed JSON manifest per benchmark tasks. Each record must point to the actual intended speech input and retain reference text/labels, target answers, case ID, and (for ASR) the exact split.
2. For TriviaQA and Alpaca, create synthesized audio first. The paper describes CosyVoice zero-shot synthesis and filtering through Whisper-large-v3, but the supplied paper does not contain all prompt-speaker IDs and filtering scripts. Use the original speaker set/protocol if available; otherwise explicitly document the adapted protocol.
3. Configure official checkpoints or APIs for every desired baseline. Run a 5-case pilot before any full run.
4. Run `python -m benchmark.run_sova_model --model all --input path\to\audio_manifest.json`. For a strict Table 4 report, use full target manifests and separate runs for each dataset/split where inputs differ.
5. Run `powershell -ExecutionPolicy Bypass -File benchmark\postprocess_sova_runs.ps1` to apply the configured GPT-OSS 120B judge (through the user's existing Groq access, if available), local Whisper-large-v3 generation-only consistency transcription, local UTMOSv2 scoring, and Table 4 export. This workflow does not call the paid OpenAI Cascade API. The Groq judge still depends on the user's existing account access and quota; if unavailable or rate-limited, the judge step must be reported as skipped/failed, not silently replaced.
6. Inspect `summary.json` for each model and the `metric_method_notes`, `n_cases`, and empty cells in `benchmark\Table4_comparison.csv` before describing a result as a measured improvement.

## Exact reproduction limitations

The supplied five-page paper gives the dataset/task families and metrics, but does not specify every preprocessing parameter, model revision, speaker-prompt set, judge prompt, or class-label conversion. Some datasets (including IEMOCAP) require independent access/licensing, and text-only TriviaQA/Alpaca records are not equivalent to the paper's synthesized audio inputs. Thus this project now has a **runner and calculation pipeline for all paper baselines**, but exact author-level numerical reproduction is only established after all models are actually run using matching data/protocols and coverage has been checked. The existing stored pilot results are diagnostics, not full-benchmark results. The paper Cascade is deliberately not run through a substitute model; published Table 4 Cascade values are reference only.
