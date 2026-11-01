"""Create a Table-4-shaped CSV with clearly separated paper references and runs.

It never copies published values into measured rows and never turns custom CRHI
ratings into GPTScore. Unrun/unsupported tasks are blank.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from benchmark.sova_metrics import corpus_wer

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "paper_reference_table4.json"
TASK_ALIASES = {
    "knowledge": "knowledge_qa", "triviaqa": "knowledge_qa", "trivia_qa": "knowledge_qa",
    "asr": "speech_recognition", "librispeech": "speech_recognition",
    "librisqa": "spoken_qa_multiple_choice", "spoken_qa": "spoken_qa_multiple_choice",
    "spoken_squad": "spoken_qa_open_ended", "open_ended_qa": "spoken_qa_open_ended",
    "emotion": "emotion_recognition", "age": "age_recognition", "gender": "gender_recognition",
    "consistency": "generation_consistency", "semantic": "generation_semantic", "acoustic": "generation_acoustic",
    "open_ended": "generation_semantic", "generation": "generation_semantic", "generation_response": "generation_semantic",
}
TABLE_COLUMNS = [
    "model", "llm_size", "source_type", "measurement_status", "n_cases", "n_success",
    "knowledge_accuracy_pct", "dev_clean_wer_pct", "dev_other_wer_pct", "test_clean_wer_pct", "test_other_wer_pct",
    "librisqa_accuracy_pct", "spoken_squad_accuracy_pct", "emotion_accuracy_pct", "age_accuracy_pct", "gender_accuracy_pct",
    "generation_consistency_wer_pct", "generation_gptscore", "generation_utmosv2_score", "metric_method_notes",
]
METRIC_KEYS = [c for c in TABLE_COLUMNS if c.endswith("_pct") or c.endswith("_wer_pct") or c.endswith("_score")]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("results", "records", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    return data if isinstance(data, list) and all(isinstance(x, dict) for x in data) else []


def _task(row: dict[str, Any]) -> str:
    raw = str(row.get("task", "")).strip().lower()
    return TASK_ALIASES.get(raw, raw)


def _correct(row: dict[str, Any]) -> bool | None:
    # Prefer the declared GPTEval judgement where available.
    for key in ("judge_correct", "gpt_eval_correct", "semantic_correct", "correct", "correct_normalized"):
        value = row.get(key)
        if isinstance(value, bool):
            return value if row.get("success", True) is not False else False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value) if row.get("success", True) is not False else False
        if isinstance(value, str):
            val = value.strip().lower()
            if val in {"true", "yes", "correct", "1"}:
                return row.get("success", True) is not False
            if val in {"false", "no", "incorrect", "0"}:
                return False
    expected = row.get("expected_label") or row.get("gold_label") or row.get("label")
    predicted = row.get("predicted_label") or row.get("prediction_label")
    if expected is not None and predicted is not None:
        return str(expected).strip().casefold() == str(predicted).strip().casefold() and row.get("success", True) is not False
    return None


def _accuracy(rows: list[dict[str, Any]]) -> tuple[float | None, str]:
    if not rows:
        return None, "not_run"
    if all(row.get("judge_correct") is not None or row.get("gpt_eval_correct") is not None or row.get("semantic_correct") is not None for row in rows):
        method = "GPT-OSS-120B-judged correctness fields (user-configured Groq access; not exact hidden author prompt)"
    elif any(row.get("expected_label") is not None for row in rows):
        method = "label match / stored correctness; check prompt and denominator"
    else:
        method = "stored correctness or inferred exact label; not verified as authors' GPTEval"
    vals = [_correct(row) for row in rows]
    # Failed attempts and no-answer cases remain in the denominator as incorrect.
    correct_count = sum(v is True for v in vals)
    return 100.0 * correct_count / len(rows), method


def _pick_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        val = row.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _wer(rows: list[dict[str, Any]], *, generated: bool = False) -> tuple[float | None, int]:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if row.get("success", True) is False:
            continue
        if generated:
            reference = _pick_text(row, ("model_text_output", "response_text", "text_output"))
            hypothesis = _pick_text(row, ("generated_transcript", "audio_transcript"))
        else:
            reference = _pick_text(row, ("reference_text", "reference_transcript", "reference", "target_text", "transcript_reference"))
            hypothesis = _pick_text(row, ("hypothesis", "recognized_text", "transcript", "prediction", "output"))
        if reference and hypothesis:
            pairs.append((reference, hypothesis))
    if not pairs:
        return None, 0
    result = corpus_wer(pairs)
    return result.get("wer_percent"), len(pairs)


def _mean_score(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> tuple[float | None, int]:
    values: list[float] = []
    for row in rows:
        if row.get("success", True) is False:
            continue
        value = None
        for key in keys:
            candidate = row.get(key)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                value = float(candidate)
                break
        if value is not None:
            values.append(value)
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def _split(row: dict[str, Any]) -> str:
    raw = str(row.get("dataset_split") or row.get("split") or row.get("subset") or "").lower().replace("_", "-")
    for split in ("dev-clean", "dev-other", "test-clean", "test-other"):
        if split in raw:
            return split
    return raw


def aggregate_model(model_name: str, model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {c: None for c in TABLE_COLUMNS}
    unique_cases = {str(r.get("case_id") or r.get("id") or f"row:{i}") for i, r in enumerate(model_rows)}
    successful_cases = {str(r.get("case_id") or r.get("id") or f"row:{i}") for i, r in enumerate(model_rows) if r.get("success") is True}
    model_id = next((str(r.get("model_id") or r.get("model")) for r in model_rows if r.get("model_id") or r.get("model")), "")
    llm_size = "unspecified"
    try:
        from benchmark.sova_model_adapters import load_registry
        registered = {str(item["id"]): item for item in load_registry().get("models", [])}
        if model_id in registered:
            llm_size = str(registered[model_id].get("llm_size", "unspecified"))
    except Exception:
        pass
    result.update(model=model_name, llm_size=llm_size, source_type="measured_local_run", measurement_status="no_comparable_metrics", n_cases=len(unique_cases), n_success=len(successful_cases), metric_method_notes="; ".join(sorted({str(r.get("metric_method")) for r in model_rows if r.get("metric_method")})))
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in model_rows:
        by_task.setdefault(_task(row), []).append(row)
    notes: list[str] = []
    for task, out_key in [
        ("knowledge_qa", "knowledge_accuracy_pct"), ("spoken_qa_multiple_choice", "librisqa_accuracy_pct"),
        ("spoken_qa_open_ended", "spoken_squad_accuracy_pct"), ("emotion_recognition", "emotion_accuracy_pct"),
        ("age_recognition", "age_accuracy_pct"), ("gender_recognition", "gender_accuracy_pct"),
    ]:
        if task in by_task:
            task_rows = by_task[task]
            result[out_key], method = _accuracy(task_rows)
            task_success = sum(row.get("success") is True for row in task_rows)
            notes.append(f"{out_key}: {method}; target_records={len(task_rows)}; successful_inferences={task_success}")
    if "speech_recognition" in by_task:
        for split, key in [("dev-clean", "dev_clean_wer_pct"), ("dev-other", "dev_other_wer_pct"), ("test-clean", "test_clean_wer_pct"), ("test-other", "test_other_wer_pct")]:
            subset = [r for r in by_task["speech_recognition"] if _split(r) == split]
            result[key], n = _wer(subset)
            if subset:
                notes.append(f"{key}: corpus WER from {n} matched reference/hypothesis pairs")
    if "generation_consistency" in by_task:
        result["generation_consistency_wer_pct"], n = _wer(by_task["generation_consistency"], generated=True)
        if n:
            notes.append(f"generation consistency: corpus WER from {n} paired text/audio outputs")
        elif any(r.get("generated_transcript") for r in by_task["generation_consistency"]):
            notes.append("generation consistency unavailable because separate model text and generated-audio transcript were not both present")
    semantic_rows = by_task.get("generation_semantic", []) + by_task.get("generation_response", [])
    result["generation_gptscore"], n = _mean_score(semantic_rows, ("gptscore", "paper_gptscore"))
    if n:
        notes.append(f"GPTScore: mean from {n} records; confirm evaluation protocol against paper")
    if any("custom_crhi_mean_score_5" in r or "mean_score_5" in r for r in semantic_rows):
        notes.append("custom C/R/H/I rubric ignored for GPTScore; not the same metric")
    acoustic_rows = by_task.get("generation_acoustic", [])
    result["generation_utmosv2_score"], n = _mean_score(acoustic_rows, ("utmosv2_score", "utmosv2"))
    if n:
        notes.append(f"UTMOSv2: mean from {n} records")
    populated = [c for c in METRIC_KEYS if result.get(c) is not None]
    result["measurement_status"] = "measured_subset" if populated else "no_comparable_metrics"
    result["metric_method_notes"] = "; ".join(notes + ([result["metric_method_notes"]] if result["metric_method_notes"] else []))
    return result


def build_table(runs_root: Path, *, include_reference: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_reference:
        paper = json.loads(REFERENCE.read_text(encoding="utf-8"))
        for model in paper.get("models", []):
            row = {c: None for c in TABLE_COLUMNS}
            row.update({key: model.get(key) for key in row if key in model})
            row.update(model=model.get("model"), llm_size=model.get("llm_size"), source_type="published_paper_table4_reference", measurement_status="published_reference_not_reproduced", n_cases=None, n_success=None, metric_method_notes="Transcribed from supplied SOVA-Bench Table 4; this is not a new run in this project.")
            rows.append(row)
    measured_model_ids: set[str] = set()
    if runs_root.exists():
        for path in sorted(runs_root.glob("*/results.json")):
            try:
                records = _read_rows(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not records:
                continue
            inferred = str(records[0].get("model_name") or path.parent.name)
            measured_model_ids.add(str(records[0].get("model_id") or records[0].get("model") or path.parent.name))
            rows.append(aggregate_model(inferred, records))

    # Always show the proposed system in the initial Table-4-shaped scaffold.
    # Once an actual results.json is present for this registry ID, the measured
    # row above replaces this placeholder. Empty metrics stay blank, never zero.
    try:
        from benchmark.sova_model_adapters import load_registry
        registry = load_registry()
        for model in registry.get("models", []):
            if model.get("id") != "weings_ai" or model.get("id") in measured_model_ids:
                continue
            row = {c: None for c in TABLE_COLUMNS}
            row.update(
                model=model.get("name", "Emotional Voice Assistant"),
                llm_size=model.get("llm_size", "configured runtime"),
                source_type="proposed_model_not_run",
                measurement_status="not_run",
                n_cases=0,
                n_success=0,
                metric_method_notes=(
                    "Placeholder only. Run the Emotional Voice Assistant through the audio-backed "
                    "SOVA manifest to populate measured metrics. No scores have been imputed."
                ),
            )
            rows.append(row)
    except Exception:
        pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "Table4_comparison.csv")
    parser.add_argument("--without-paper-reference", action="store_true")
    args = parser.parse_args()
    rows = build_table(args.runs_root, include_reference=not args.without_paper_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote Table-4-shaped comparison: {args.output}")
    print(f"Rows: {len(rows)} ({sum(r['source_type']=='published_paper_table4_reference' for r in rows)} paper references, {sum(r['source_type']=='measured_local_run' for r in rows)} measured local runs)")
    print("Blank values mean not run or not computable from available records; custom C/R/H/I is never mapped to GPTScore.")


if __name__ == "__main__":
    main()
