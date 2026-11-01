"""Score SOVA-Bench-style prediction records without making API calls.

The scorer is deliberately conservative: model results are never pooled, failed
runs are excluded from conditional metrics, unrun paper tasks remain visible,
and custom semantic rubrics are never relabelled as the paper's GPTScore.

Input can be a JSON list or an object containing ``results``, ``records`` or
``items``. Multiple input files may be passed to the CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmark.sova_metrics import (
    accuracy,
    answer_rate,
    conditioned_accuracy,
    corpus_wer,
    exact_label_match,
    mean_or_none,
    safe_float,
    wer_stats,
)

ROOT = Path(__file__).resolve().parent

# SOVA-Bench's 8 task groups expand into 10 metric-bearing rows because speech
# generation has consistency, semantic and acoustic sub-metrics.
PAPER_TASKS: dict[str, dict[str, Any]] = {
    "knowledge_qa": {"paper_instances": 6840, "paper_metric": "GPT-evaluated accuracy", "direction": "higher"},
    "speech_recognition": {"paper_instances": None, "paper_metric": "WER on dev/test clean/other splits", "direction": "lower"},
    "spoken_qa_multiple_choice": {"paper_instances": 2619, "paper_metric": "Accuracy", "direction": "higher"},
    "spoken_qa_open_ended": {"paper_instances": 3243, "paper_metric": "GPT-evaluated accuracy", "direction": "higher"},
    "emotion_recognition": {"paper_instances": 5819, "paper_metric": "GPT-evaluated accuracy; answer rate and conditioned accuracy", "direction": "higher"},
    "gender_recognition": {"paper_instances": 2469, "paper_metric": "GPT-evaluated accuracy; answer rate and conditioned accuracy", "direction": "higher"},
    "age_recognition": {"paper_instances": 2511, "paper_metric": "GPT-evaluated accuracy; answer rate and conditioned accuracy", "direction": "higher"},
    "generation_consistency": {"paper_instances": 3745, "paper_metric": "WER between response text and speech transcript", "direction": "lower"},
    "generation_semantic": {"paper_instances": 3745, "paper_metric": "GPTScore", "direction": "higher"},
    "generation_acoustic": {"paper_instances": 3745, "paper_metric": "UTMOSv2 raw score", "direction": "higher"},
}

TASK_ALIASES = {
    "knowledge": "knowledge_qa",
    "triviaqa": "knowledge_qa",
    "trivia_qa": "knowledge_qa",
    "speech_recognition": "speech_recognition",
    "asr": "speech_recognition",
    "librispeech": "speech_recognition",
    "librisqa": "spoken_qa_multiple_choice",
    "spoken_qa": "spoken_qa_multiple_choice",
    "spoken_qa_multiple_choice": "spoken_qa_multiple_choice",
    "spoken_squad": "spoken_qa_open_ended",
    "spoken_qa_open_ended": "spoken_qa_open_ended",
    "emotion": "emotion_recognition",
    "emotion_recognition": "emotion_recognition",
    "gender": "gender_recognition",
    "gender_recognition": "gender_recognition",
    "age": "age_recognition",
    "age_recognition": "age_recognition",
    "consistency": "generation_consistency",
    "generation_consistency": "generation_consistency",
    "semantic": "generation_semantic",
    "generation_semantic": "generation_semantic",
    "acoustic": "generation_acoustic",
    "generation_acoustic": "generation_acoustic",
    "open_ended": "generation_response",
    "generation": "generation_response",
    "generation_response": "generation_response",
    "instruction": "instruction_following",
    "instruction_following": "instruction_following",
}


def parse_bool(value: Any, default: bool | None = None) -> bool | None:
    """Parse booleans from JSON-friendly bools, 0/1, and common strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        val = value.strip().casefold()
        if val in {"true", "t", "yes", "y", "1", "success", "ok"}:
            return True
        if val in {"false", "f", "no", "n", "0", "failed", "failure", "error"}:
            return False
    return default


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _infer_task(row: dict[str, Any]) -> str:
    """Infer a task only from unambiguous metric-specific fields."""
    if _first(row, "utmosv2_score", "utmosv2", "acoustic_score") is not None:
        return "generation_acoustic"
    if _first(row, "gptscore", "paper_gptscore") is not None:
        return "generation_semantic"
    if _first(row, "generated_transcript") is not None and _first(row, "reference_text") is not None:
        return "generation_consistency" if str(_first(row, "case_id", "id") or "").startswith("generation_") else "speech_recognition"
    if _first(row, "hypothesis") is not None and _first(row, "reference") is not None:
        return "speech_recognition"
    if any(key in row for key in ("mean_score_5", "semantic_score_pct", "correctness", "instruction_following")):
        return "generation_semantic"
    return "unknown"


def normalize_row(
    row: dict[str, Any],
    *,
    defaults: dict[str, str] | None = None,
) -> dict[str, Any]:
    task_raw = str(row.get("task", "")).strip().lower()
    task = TASK_ALIASES.get(task_raw, task_raw)
    if not task or task == "unknown":
        task = _infer_task(row)

    prediction = _first(row, "prediction", "response_text", "output", "model_output", "generated_text")
    transcript = _first(row, "generated_transcript", "transcript", "hypothesis", "recognized_text")
    reference_text = _first(row, "reference_text", "reference", "reference_transcript", "gold_text", "target_text")
    expected_label = _first(row, "expected_label", "ground_truth_label", "label", "gold_label", "expected")
    predicted_label = _first(row, "predicted_label", "prediction_label", "predicted", "label_prediction")

    # Avoid bool("false") == True, a subtle but high-impact evaluation error.
    judge_correct = parse_bool(_first(row, "judge_correct", "gpt_eval_correct", "semantic_correct"))
    correct = parse_bool(row.get("correct"))
    if judge_correct is not None:
        correct = judge_correct
    elif correct is None and expected_label is not None and predicted_label is not None:
        correct = exact_label_match(str(predicted_label), str(expected_label))

    success_value = parse_bool(row.get("success", row.get("ok")))
    if success_value is None:
        success_value = not bool(row.get("error")) and bool(
            prediction or transcript or predicted_label or row.get("correct") is not None
        )
    answered_value = parse_bool(row.get("answered"))
    if answered_value is None:
        answered_value = bool(prediction or transcript or predicted_label)
    answered = bool(success_value and answered_value)

    model_name = str(_first(row, "model_name", "system_name", "model_alias") or (defaults or {}).get("model_name") or "Unspecified model")
    provider = str(_first(row, "provider") or (defaults or {}).get("provider") or "")
    model = str(_first(row, "model", "model_id") or (defaults or {}).get("model") or "")

    if row.get("metric_method"):
        metric_method = str(row["metric_method"])
    elif task in {"knowledge_qa", "spoken_qa_open_ended"}:
        metric_method = "provided_correct_flag_not_verified_GPT_judge" if judge_correct is None else "provided_GPT_judge_correctness"
    elif task in {"emotion_recognition", "gender_recognition", "age_recognition"}:
        metric_method = "exact_label_match_or_provided_correct_flag"
    elif task == "spoken_qa_multiple_choice":
        metric_method = "normal_accuracy_or_provided_correct_flag"
    elif task in {"speech_recognition", "generation_consistency"}:
        metric_method = "normalized_WER_from_reference_and_hypothesis"
    elif task == "generation_semantic":
        metric_method = "custom_or_unspecified_semantic_metric"
    elif task == "generation_acoustic":
        metric_method = "UTMOSv2_if_score_field_is_utmosv2"
    else:
        metric_method = "unspecified"

    normalized = dict(row)
    normalized.update({
        "task": task,
        "model_name": model_name,
        "provider": provider,
        "model": model,
        "success": bool(success_value),
        "answered": answered,
        "case_id": str(_first(row, "case_id", "id", "caseId") or ""),
        "prediction": None if prediction is None else str(prediction),
        "transcript": None if transcript is None else str(transcript),
        "reference_text": None if reference_text is None else str(reference_text),
        "expected_label": None if expected_label is None else str(expected_label),
        "predicted_label": None if predicted_label is None else str(predicted_label),
        "correct_normalized": correct,
        "judge_correct": judge_correct,
        "metric_method": metric_method,
    })
    if normalized.get("gptscore") is None and normalized.get("paper_gptscore") is not None:
        normalized["gptscore"] = normalized["paper_gptscore"]
    if normalized.get("utmosv2_score") is None and normalized.get("utmosv2") is not None:
        normalized["utmosv2_score"] = normalized["utmosv2"]
    if normalized.get("custom_crhi_mean_score_5") is None and normalized.get("mean_score_5") is not None:
        normalized["custom_crhi_mean_score_5"] = normalized["mean_score_5"]
    if normalized.get("custom_crhi_score_pct") is None and normalized.get("semantic_score_pct") is not None:
        normalized["custom_crhi_score_pct"] = normalized["semantic_score_pct"]
    return normalized


def _task_summary(task: str, group: list[dict[str, Any]], expected_count: int | None = None) -> dict[str, Any]:
    meta = PAPER_TASKS.get(task)
    if meta is None:
        successful = [r for r in group if r["success"]]
        return {
            "status": "auxiliary_or_unmapped",
            "cases_in_file": len(group),
            "successful_cases": len(successful),
            "failed_cases": len(group) - len(successful),
            "coverage_percent": accuracy(len(successful), len(group)),
            "metric_method": sorted({str(row["metric_method"]) for row in group}),
            "note": "This record type is useful for the project audit but is not itself one of the paper's named metric rows.",
        }

    successful = [row for row in group if row["success"]]
    case_ids = [row["case_id"] for row in group if row["case_id"]]
    duplicate_case_count = len(case_ids) - len(set(case_ids))
    instances = meta["paper_instances"]
    run_denominator = max(len(group), int(expected_count or 0))
    unrecorded_expected = max(0, run_denominator - len(group)) if expected_count is not None else 0
    effective_failures = (len(group) - len(successful)) + unrecorded_expected
    result: dict[str, Any] = {
        "status": "not_run" if not group else ("partial_subset" if instances and len(group) < instances else "records_present"),
        "paper_instances": instances,
        "paper_metric": meta["paper_metric"],
        "direction": meta["direction"],
        "cases_in_file": len(group),
        "expected_cases_for_this_run": expected_count,
        "unrecorded_expected_cases": unrecorded_expected,
        "effective_run_denominator": run_denominator,
        "unique_case_ids": len(set(case_ids)),
        "duplicate_case_ids": duplicate_case_count,
        "successful_cases": len(successful),
        "failed_cases_in_file": len(group) - len(successful),
        "effective_failed_or_missing_cases": effective_failures,
        "execution_coverage_percent": accuracy(len(successful), run_denominator),
        "dataset_coverage_vs_paper_percent": accuracy(len(set(case_ids)), instances) if instances else None,
        "metric_method": sorted({str(row["metric_method"]) for row in group}),
    }

    if not group:
        result["note"] = "No prediction records found. This task is not evaluated; no score has been imputed."
        if expected_count is not None:
            result["note"] += f" The declared run target was {expected_count}; all cases are currently missing."
        if task in {"knowledge_qa", "spoken_qa_multiple_choice", "spoken_qa_open_ended", "emotion_recognition", "gender_recognition", "age_recognition"}:
            result.update({
                "judged_successful_cases": 0,
                "answered_successful_cases": 0,
                "correct_successful_cases": 0,
                "accuracy_percent_over_judged_successful_cases": None,
                "accuracy_percent_over_all_cases": accuracy(0, run_denominator),
                "answer_rate_percent": accuracy(0, run_denominator),
                "conditioned_accuracy_percent": None,
            })
        elif task in {"speech_recognition", "generation_consistency"}:
            result.update({
                "instances_with_reference_and_hypothesis": 0,
                "corpus_wer_percent": None,
                "macro_utterance_wer_percent": None,
            })
        elif task == "generation_semantic":
            result.update({"mean_gptscore": None, "gptscore_scored_cases": 0, "mean_custom_crhi_score_5": None, "mean_custom_crhi_percent": None})
        elif task == "generation_acoustic":
            result.update({"mean_utmosv2_score": None, "utmosv2_scored_cases": 0})
        return result

    if task in {"speech_recognition", "generation_consistency"}:
        pairs: list[tuple[str, str]] = []
        for row in successful:
            ref = row.get("reference_text")
            hyp = row.get("transcript")
            if ref is not None and hyp is not None:
                pairs.append((str(ref), str(hyp)))
        wer_result = corpus_wer(pairs)
        result.update({
            "instances_with_reference_and_hypothesis": len(pairs),
            "corpus_wer_percent": wer_result["wer_percent"],
            "macro_utterance_wer_percent": wer_result["macro_wer_percent"],
            "wer_substitutions": wer_result["substitutions"],
            "wer_deletions": wer_result["deletions"],
            "wer_insertions": wer_result["insertions"],
            "reference_words": wer_result["reference_words"],
            "wer_formula": "100 * (substitutions + deletions + insertions) / reference_words",
            "wer_normalization": "NFKC + casefold + punctuation/spacing normalization; corpus micro-WER",
        })
        # Preserve old-script WER values as a diagnostic, not as the primary
        # normalized score. The included LibriSpeech runner saved `wer` as a
        # fraction using lowercasing and whitespace splitting only.
        saved_values: list[float] = []
        for row in successful:
            val = safe_float(row.get("wer_percent"))
            if val is not None:
                saved_values.append(val)
                continue
            val = safe_float(row.get("wer_fraction"))
            if val is not None:
                saved_values.append(val * 100.0)
                continue
            val = safe_float(row.get("wer"))
            if val is not None:
                # The included benchmark/sova_librispeech_500.py saves WER as a fraction.
                saved_values.append(val * 100.0)
        result["saved_utterance_mean_wer_percent"] = mean_or_none(saved_values)
        result["saved_wer_rows"] = len(saved_values)
        if saved_values and wer_result["wer_percent"] is not None:
            result["wer_normalization_note"] = (
                "The saved per-utterance WER uses the source script's lowercase+whitespace tokenization; "
                "corpus_wer_percent is recomputed with punctuation/spacing normalization. Differences are expected; "
                "state the selected normalization in any paper table."
            )
    elif task in {
        "knowledge_qa", "spoken_qa_multiple_choice", "spoken_qa_open_ended",
        "emotion_recognition", "gender_recognition", "age_recognition",
    }:
        answered_rows = [row for row in successful if row["answered"]]
        judged_rows = [row for row in successful if row["correct_normalized"] is not None]
        correct_rows = [row for row in judged_rows if bool(row["correct_normalized"])]
        result.update({
            "judged_successful_cases": len(judged_rows),
            "answered_successful_cases": len(answered_rows),
            "correct_successful_cases": len(correct_rows),
            "accuracy_percent_over_judged_successful_cases": accuracy(len(correct_rows), len(judged_rows)),
            # Failed/unanswered attempts and missing expected rows stay in this denominator.
            "accuracy_percent_over_all_cases": accuracy(len(correct_rows), run_denominator),
            "answer_rate_percent": answer_rate(len(answered_rows), run_denominator),
            "conditioned_accuracy_percent": conditioned_accuracy(len(correct_rows), len(answered_rows)),
        })
        if task in {"knowledge_qa", "spoken_qa_open_ended"} and any(
            row.get("judge_correct") is None and row.get("correct_normalized") is not None for row in successful
        ):
            result["paper_metric_warning"] = "The paper names GPT-evaluated accuracy. Provided correctness flags are not verified as the same GPT judge/prompt."
        if task in {"emotion_recognition", "gender_recognition", "age_recognition"} and any(
            str(row.get("input_modality", "")).casefold() == "text" for row in successful
        ):
            result["input_modality_warning"] = "Text-only classification does not measure the paper's speech-based paralinguistic recognition task."
    elif task == "generation_semantic":
        paper_scores = [safe_float(_first(row, "gptscore", "paper_gptscore")) for row in successful]
        custom_scores = [safe_float(_first(row, "custom_crhi_mean_score_5", "mean_score_5")) for row in successful]
        custom_percent = [safe_float(_first(row, "custom_crhi_score_pct", "semantic_score_pct")) for row in successful]
        result.update({
            "mean_gptscore": mean_or_none(paper_scores),
            "gptscore_scored_cases": sum(v is not None for v in paper_scores),
            "mean_custom_crhi_score_5": mean_or_none(custom_scores),
            "mean_custom_crhi_percent": mean_or_none(custom_percent),
            "custom_crhi_scored_cases": sum(v is not None for v in custom_scores),
            "paper_metric_warning": "The paper reports GPTScore. The custom C/R/H/I Likert rubric is not the paper's GPTScore and is shown separately." if any(v is not None for v in custom_scores) and not any(v is not None for v in paper_scores) else None,
        })
    elif task == "generation_acoustic":
        values = [safe_float(_first(row, "utmosv2_score", "utmosv2")) for row in successful]
        result.update({
            "mean_utmosv2_score": mean_or_none(values),
            "utmosv2_scored_cases": sum(value is not None for value in values),
            "score_scale": "raw UTMOSv2 model score; no percentage conversion for paper comparison",
        })

    if instances and len(group) < instances:
        result["note"] = f"Only {len(group)} records are present against {instances} paper instances. Treat as a subset, not a full benchmark score."
    return result

def score_rows(
    rows: list[dict[str, Any]],
    *,
    defaults: dict[str, str] | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    normalized = [normalize_row(row, defaults=defaults) for row in rows]
    models: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        models[(row["model_name"], row["provider"], row["model"])].append(row)

    model_reports: list[dict[str, Any]] = []
    for (model_name, provider, model), model_rows in sorted(models.items()):
        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in model_rows:
            task_groups[row["task"] or "unknown"].append(row)
        task_names = set(PAPER_TASKS) | set(task_groups)
        task_summaries = {
            task: _task_summary(task, task_groups.get(task, []), (expected_counts or {}).get(task))
            for task in sorted(task_names)
        }
        model_reports.append({
            "model_name": model_name,
            "provider": provider,
            "model": model,
            "input_rows": len(model_rows),
            "failed_rows": sum(not row["success"] for row in model_rows),
            "tasks": task_summaries,
        })

    summary: dict[str, Any] = {
        "benchmark": "Emotion Aware Voice Assistant SOVA-compatible evaluation",
        "paper_reference": "SOVA-Bench, arXiv:2506.02457v1, Table 2 and Table 4",
        "strict_official_reproduction_claimed": False,
        "paper_top_level_task_groups": 8,
        "paper_metric_rows": len(PAPER_TASKS),
        "total_rows": len(normalized),
        "model_count": len(model_reports),
        "models": model_reports,
        "method_notes": [
            "Model results are grouped separately; scores from different models are never pooled.",
            "Failed cases are excluded from conditional/answer metrics and remain in the all-case denominator; execution coverage is reported separately.",
            "Knowledge QA and open-ended spoken QA use paper-compatible accuracy only when judge_correct came from a declared GPT evaluator. Other correctness flags are explicitly marked as non-identical.",
            "Emotion/age/gender classification from text alone is not an acoustic paralinguistic evaluation; paper-style results require the relevant speech inputs and label protocol.",
            "Generation consistency WER is recomputed from reference text and an ASR transcript of generated audio.",
            "A custom C/R/H/I Likert rubric is not the paper's GPTScore and is not substituted for it.",
            "The original 5-page paper does not fully specify all evaluator prompts and implementation details; this report therefore does not claim exact author-side reproduction.",
        ],
        "normalized_rows": normalized,
    }
    # Backward-compatible convenience for a single-model run. For multiple models
    # this remains null so consumers cannot accidentally treat pooled metrics as one model.
    summary["tasks"] = model_reports[0]["tasks"] if len(model_reports) == 1 else None
    return summary


def write_csv(path: Path, models: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for model_report in models:
        for task, task_summary in model_report["tasks"].items():
            row = {
                "model_name": model_report["model_name"],
                "provider": model_report["provider"],
                "model": model_report["model"],
                "task": task,
            }
            for key, value in task_summary.items():
                row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            flat.append(row)
    if not flat:
        path.write_text("model_name,provider,model,task\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def _load_records(path: Path) -> list[dict[str, Any]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, dict):
        content = content.get("results", content.get("records", content.get("items", [])))
    if not isinstance(content, list):
        raise ValueError(f"Input must be a JSON list or object containing results/records/items: {path}")
    return [row for row in content if isinstance(row, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="One or more JSON prediction/result files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model-name", default=None, help="Default model name for imported rows missing model metadata")
    parser.add_argument("--provider", default=None, help="Default provider for imported rows missing provider metadata")
    parser.add_argument("--model", default=None, help="Default model ID for imported rows missing model metadata")
    parser.add_argument(
        "--expected-count", action="append", default=[], metavar="TASK=COUNT",
        help="Declared target size for this run (repeatable), e.g. spoken_qa_open_ended=500. Missing records count as failures/unanswered.",
    )
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for source_path in args.input:
        rows.extend(_load_records(source_path))
    defaults = {k: v for k, v in {"model_name": args.model_name, "provider": args.provider, "model": args.model}.items() if v}
    expected_counts: dict[str, int] = {}
    for entry in args.expected_count:
        if "=" not in entry:
            parser.error(f"Invalid --expected-count {entry!r}; use TASK=COUNT")
        task_name, count_text = entry.split("=", 1)
        task_name = TASK_ALIASES.get(task_name.strip().lower(), task_name.strip().lower())
        try:
            count = int(count_text)
        except ValueError:
            parser.error(f"Invalid count in --expected-count {entry!r}")
        if count < 0:
            parser.error("Expected counts must be non-negative")
        expected_counts[task_name] = count
    args.output.mkdir(parents=True, exist_ok=True)
    summary = score_rows(rows, defaults=defaults, expected_counts=expected_counts)
    json_summary = {key: value for key, value in summary.items() if key != "normalized_rows"}
    (args.output / "summary.json").write_text(json.dumps(json_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "scored_cases.json").write_text(json.dumps(summary["normalized_rows"], indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output / "task_summary.csv", summary["models"])
    print(json.dumps({
        "model_count": summary["model_count"],
        "total_rows": summary["total_rows"],
        "models": [
            {"model_name": item["model_name"], "provider": item["provider"], "model": item["model"],
             "tasks_run": [name for name, task in item["tasks"].items() if task.get("cases_in_file", 0)]}
            for item in summary["models"]
        ],
    }, indent=2, ensure_ascii=False))
    print(f"Wrote model-wise SOVA-compatible metrics to {args.output}")


if __name__ == "__main__":
    main()
