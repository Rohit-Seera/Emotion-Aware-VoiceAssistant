"""Build a provenance-aware SOVA-style summary for the Emotion Aware Voice Assistant."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "outputs"
ASR = RUN / "inputs" / "asr_librispeech_summary_1000.json"
PAPER = RUN / "reference" / "paper_reference_table4.json"


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "NR"
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    tasks = json.loads((OUT / "task_summary.json").read_text(encoding="utf-8"))
    asr = json.loads(ASR.read_text(encoding="utf-8"))
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    tr = tasks["task_results"]
    splits = asr["by_split"]

    latest = {}
    case_file = OUT / "task_case_results.jsonl"
    if case_file.exists():
        for line in case_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                latest[item["case_id"]] = item
    selected = []
    for task in ("triviaqa", "librisqa", "spoken_squad", "alpaca"):
        manifest = json.loads((RUN / "inputs" / f"{task}_eval_sample_55.json").read_text(encoding="utf-8"))
        for case in manifest:
            case_id = str(case["case_id"])
            selected.append(latest.get(case_id, {"case_id": case_id, "task": task, "success": False, "status": "not_attempted"}))
    (OUT / "FINAL_SELECTED_CASE_RESULTS.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    provenance_files = [
        OUT / "task_summary.json",
        OUT / "FINAL_SELECTED_CASE_RESULTS.jsonl",
        ASR,
        PAPER,
        RUN / "code" / "run_evaluation.py",
        RUN / "code" / "build_final_results.py",
        RUN / "source_backup" / "sova_metrics.py",
        RUN / "inputs" / "spoken_squad_source_hashes.json",
        RUN / "inputs" / "spoken_squad_audio_sha256.json",
        RUN / "code" / "run_evaluation.py",
        RUN / "code" / "build_final_results.py",
        RUN / "source_backup" / "sova_metrics.py",
        *(RUN / "inputs" / f"{task}_eval_sample_55.json" for task in ("triviaqa", "librisqa", "spoken_squad", "alpaca")),
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assistant_model": tasks["model"],
        "sample_sizes": {task: len(json.loads((RUN / "inputs" / f"{task}_eval_sample_55.json").read_text(encoding="utf-8"))) for task in ("triviaqa", "librisqa", "spoken_squad", "alpaca")},
        "sampling_seed_rule": "Retain the original fixed-seed 25-case pilot and add 30 randomly sampled cases per dataset with seed = 20261030 + sum(ord(character) for character in task name).",
        "full_spoken_squad_source": "Available at inputs/spoken_squad_full_source (directory junction to the preserved dataset archive); the folder also includes checksums and the 55 copied evaluation clips.",
        "full_librispeech_source": "Available at inputs/librispeech_source (directory junction to the preserved LibriSpeech dataset archive).",
        "hashed_files": {
            str(path.relative_to(RUN)) if path.is_relative_to(RUN) else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in provenance_files
        },
        "metric_caveat": "SOVA-style task-family pilot; not a full SOVA-Bench reproduction. Published baseline values are references and were not re-run.",
    }
    (OUT / "FINAL_EVALUATION_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for model in paper["models"]:
        rows.append({
            "Model / source": model["model"] + " (published SOVA Table 4)",
            "Knowledge accuracy (%)": fmt(model.get("knowledge_accuracy_pct")),
            "LibriSpeech dev-clean WER (%)": fmt(model.get("dev_clean_wer_pct")),
            "LibriSpeech dev-other WER (%)": fmt(model.get("dev_other_wer_pct")),
            "LibriSpeech test-clean WER (%)": fmt(model.get("test_clean_wer_pct")),
            "LibriSpeech test-other WER (%)": fmt(model.get("test_other_wer_pct")),
            "LibriSQA accuracy (%)": fmt(model.get("librisqa_accuracy_pct")),
            "Spoken SQuAD accuracy (%)": fmt(model.get("spoken_squad_accuracy_pct")),
            "Emotion accuracy (%)": fmt(model.get("emotion_accuracy_pct")),
            "Age accuracy (%)": fmt(model.get("age_accuracy_pct")),
            "Gender accuracy (%)": fmt(model.get("gender_accuracy_pct")),
            "Generation consistency WER (%)": fmt(model.get("generation_consistency_wer_pct")),
            "Generation GPTScore": fmt(model.get("generation_gptscore")),
            "Generation UTMOSv2": fmt(model.get("generation_utmosv2_score")),
        })

    we = {
        "Model / source": "Emotion Aware Voice Assistant (local evaluation)",
        "Knowledge accuracy (%)": f"{fmt(tr['triviaqa'].get('exact_match_accuracy_percent'))} exact-match proxy (n={tr['triviaqa']['target_cases']}; answered={tr['triviaqa']['successful_cases']})",
        "LibriSpeech dev-clean WER (%)": f"{fmt(splits['dev-clean']['corpus_wer']['wer_percent'])} (n={splits['dev-clean']['successful']})",
        "LibriSpeech dev-other WER (%)": f"{fmt(splits['dev-other']['corpus_wer']['wer_percent'])} (n={splits['dev-other']['successful']})",
        "LibriSpeech test-clean WER (%)": f"{fmt(splits['test-clean']['corpus_wer']['wer_percent'])} (n={splits['test-clean']['successful']})",
        "LibriSpeech test-other WER (%)": f"{fmt(splits['test-other']['corpus_wer']['wer_percent'])} (n={splits['test-other']['successful']})",
        "LibriSQA accuracy (%)": f"{fmt(tr['librisqa'].get('accuracy_percent'))} (n={tr['librisqa']['target_cases']}; valid options={sum(bool(x.get('answered')) for x in selected if x.get('task') == 'librisqa')}/55; conditioned={fmt(tr['librisqa'].get('conditioned_accuracy_percent'))}%)",
        "Spoken SQuAD accuracy (%)": f"{fmt(tr['spoken_squad'].get('exact_match_accuracy_percent'))} exact-match proxy; token-F1={fmt(tr['spoken_squad'].get('mean_token_f1_percent'))} (n={tr['spoken_squad']['target_cases']})",
        "Emotion accuracy (%)": "NR (IEMOCAP unavailable)",
        "Age accuracy (%)": "NR (not an assistant output)",
        "Gender accuracy (%)": "NR (not an assistant output)",
        "Generation consistency WER (%)": "NR (speech output not evaluated)",
        "Generation GPTScore": "NR (paper evaluator not run)",
        "Generation UTMOSv2": "NR (audio metric not run)",
    }
    rows.append(we)

    csv_path = OUT / "SOVA_STYLE_COMPARISON_TABLE.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    catalog = [
        {"SOVA task": "Knowledge / TriviaQA", "Paper test size": 6840, "Local data": "Full 6840-row source used to create fixed-seed sample", "Assistant evaluated": tr["triviaqa"]["target_cases"], "Coverage/status": f"{tr['triviaqa']['successful_cases']} successful responses; sample"},
        {"SOVA task": "Speech recognition / LibriSpeech", "Paper test size": "four official splits", "Local data": "Full source at inputs/librispeech_source; final 1000-case outputs included", "Assistant evaluated": asr["successful_cases"], "Coverage/status": "250 stratified utterances each from dev-clean, dev-other, test-clean, test-other"},
        {"SOVA task": "Spoken question answering / LibriSQA Part II", "Paper test size": 2619, "Local data": "Full dataset available in inputs/; 150 ASR-transcript overlap pool", "Assistant evaluated": tr["librisqa"]["target_cases"], "Coverage/status": "sample; uses saved ASR transcripts"},
        {"SOVA task": "Spoken question answering / Spoken SQuAD", "Paper test size": 3243, "Local data": "Full 5351-row source available at inputs/spoken_squad_full_source; 55 evaluation WAVs copied here", "Assistant evaluated": tr["spoken_squad"]["target_cases"], "Coverage/status": "sample; audio hashes recorded"},
        {"SOVA task": "Speech generation / Alpaca", "Paper test size": 3745, "Local data": "Full 3745-row source used to create fixed-seed sample", "Assistant evaluated": tr["alpaca"]["target_cases"], "Coverage/status": "text response only; speech quality not measured"},
        {"SOVA task": "Emotion / IEMOCAP", "Paper test size": 5819, "Local data": "Not available locally", "Assistant evaluated": 0, "Coverage/status": "not run; dataset files were not supplied"},
        {"SOVA task": "Age / Common Voice", "Paper test size": 2511, "Local data": "Authorized Common Voice demographic release not available", "Assistant evaluated": 0, "Coverage/status": "not an output of this assistant"},
        {"SOVA task": "Gender / Common Voice", "Paper test size": 2469, "Local data": "Authorized Common Voice demographic release not available", "Assistant evaluated": 0, "Coverage/status": "not an output of this assistant"},
    ]
    with (OUT / "SOVA_DATASET_COVERAGE_CATALOG.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)

    formulas = [
        {"Metric": "Corpus WER (%)", "Calculation": "100 * (substitutions + deletions + insertions) / total reference words", "Interpretation / caveat": "Lower is better; corpus micro-average, calculated over 1,000 LibriSpeech utterances."},
        {"Metric": "TriviaQA exact-match proxy (%)", "Calculation": "100 * normalized exact matches / 55 selected questions", "Interpretation / caveat": "Does not use SOVA paper's GPT evaluator; failed backend calls count incorrect in denominator."},
        {"Metric": "LibriSQA accuracy (%)", "Calculation": "100 * correct selected options / 55 selected questions", "Interpretation / caveat": "Failed backend calls count incorrect; four-way multiple-choice; transcript comes from prior ASR sample."},
        {"Metric": "Spoken SQuAD exact-match proxy (%)", "Calculation": "100 * normalized exact matches / 55 selected audio questions", "Interpretation / caveat": "Does not use paper's GPT evaluator; failed backend calls count incorrect in denominator."},
        {"Metric": "Token F1 (%)", "Calculation": "100 * 2 * token_precision * token_recall / (token_precision + token_recall)", "Interpretation / caveat": "Average over successful responses only; lexical overlap against provided reference answer(s)."},
        {"Metric": "Alpaca ROUGE-L F1 (%)", "Calculation": "100 * LCS-based ROUGE-L F1 between generated output and reference", "Interpretation / caveat": "Text-only lexical proxy; not GPTScore, consistency WER, or UTMOSv2."},
        {"Metric": "Answer coverage (%)", "Calculation": "100 * successful backend cases / selected case count", "Interpretation / caveat": "Reported separately from conditional response-quality metrics."},
    ]
    with (OUT / "SOVA_METRIC_FORMULAS.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(formulas[0]))
        writer.writeheader()
        writer.writerows(formulas)

    md = [
        "# Emotion Aware Voice Assistant — SOVA-Bench Evaluation",
        "",
        "This table combines published SOVA-Bench Table 4 reference values with local measurements of the Emotion Aware Voice Assistant. Published baseline scores are copied from the paper; those checkpoints were not run here. The assistant was measured on fixed, reproducible samples because the free API allocation cannot support 1,000 additional model calls in this run. This is not a reproduction of the full SOVA-Bench protocol.",
        "",
        "## Assistant results",
        "",
        "| Task / dataset | Assistant result | Coverage | Metric and denominator |",
        "|---|---:|---:|---|",
        f"| LibriSpeech ASR | {fmt(asr['corpus_wer']['wer_percent'])}% corpus WER | {asr['successful_cases']}/{asr['manifest_cases']} | (S+D+I)/reference words = {asr['corpus_wer']['errors']}/{asr['corpus_wer']['reference_words']}; stratified 250/split |",
        f"| TriviaQA knowledge | {fmt(tr['triviaqa'].get('exact_match_accuracy_percent'))}% exact-match proxy | {tr['triviaqa']['successful_cases']}/{tr['triviaqa']['target_cases']} | exact answer / all selected cases; paper uses GPT-evaluated accuracy |",
        f"| LibriSQA Part II | {fmt(tr['librisqa'].get('accuracy_percent'))}% (conditioned: {fmt(tr['librisqa'].get('conditioned_accuracy_percent'))}%) | {tr['librisqa']['successful_cases']}/{tr['librisqa']['target_cases']} replies; {sum(bool(x.get('answered')) for x in selected if x.get('task') == 'librisqa')}/55 valid options | four-choice accuracy / all selected cases; uses saved Whisper transcripts |",
        f"| Spoken SQuAD | {fmt(tr['spoken_squad'].get('exact_match_accuracy_percent'))}% exact match; token-F1 {fmt(tr['spoken_squad'].get('mean_token_f1_percent'))}% | {tr['spoken_squad']['successful_cases']}/{tr['spoken_squad']['target_cases']} | answer exact match over sample; paper uses GPT-evaluated accuracy |",
        f"| Alpaca response generation | ROUGE-L F1 {fmt(tr['alpaca'].get('mean_rouge_l_f1_percent'))}%; token-F1 {fmt(tr['alpaca'].get('mean_token_f1_percent'))}% | {tr['alpaca']['successful_cases']}/{tr['alpaca']['target_cases']} | lexical proxies only; not GPTScore |",
        "",
        "In total, this package reports 1,000 LibriSpeech ASR utterances plus 220 sampled questions/prompts across TriviaQA, LibriSQA, Spoken SQuAD and Alpaca: 1,220 evaluated dataset cases. The complete LibriSpeech and Spoken SQuAD sources are available through the `../inputs/librispeech_source/` and `../inputs/spoken_squad_full_source/` directory links; the 55 evaluation WAV files and source checksums are also included.",
        "",
        "## Protocol gaps",
        "",
        "- IEMOCAP emotion accuracy: not measured; the licensed dataset was not supplied.",
        "- Common Voice age/gender: not measured; age and gender are not outputs of this assistant.",
        "- Alpaca speech consistency WER and UTMOSv2: not measured; speech synthesis was excluded from this free-quota evaluation.",
        "- Published Mini-Omni, Mini-Omni2, Moshi, LLaMA-Omni, Freeze-Omni, and GLM-4-Voice scores are paper references only; their model checkpoints were not rerun.",
        "- Exact-match, token-F1, and ROUGE-L are transparent lexical proxies and should not be presented as the paper's GPT-evaluator accuracy or GPTScore.",
        "",
        "## Reproduction files",
        "",
        "- Raw per-case assistant output: `task_case_results.jsonl`.",
        "- Exact 220-case analysis slice: `FINAL_SELECTED_CASE_RESULTS.jsonl` (four fixed 55-case manifests).",
        "- Dataset/sample inventory: `SOVA_DATASET_COVERAGE_CATALOG.csv`.",
        "- Metric equations and denominator notes: `SOVA_METRIC_FORMULAS.csv`.",
        "- Evaluation manifest/output checksums and run metadata: `FINAL_EVALUATION_PROVENANCE.json`.",
        "- Sample manifests and selected audio: `../inputs/`.",
        "- LibriSpeech case outputs and full calculation: `librispeech_asr_1000_case_results.jsonl` and `../inputs/asr_librispeech_summary_1000.json`.",
        "- Evaluation and calculation code: `../code/run_evaluation.py`, `../code/build_final_results.py`, `../source_backup/sova_metrics.py`.",
        "- Published reference: [SOVA-Bench paper, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/hou25b_interspeech.pdf).",
        "",
        "Generated from saved case outputs by `build_final_results.py`.",
    ]
    (OUT / "SOVA_FINAL_RESULTS_AND_LIMITATIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {OUT / 'SOVA_FINAL_RESULTS_AND_LIMITATIONS.md'}")


if __name__ == "__main__":
    main()
