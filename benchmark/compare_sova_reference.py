"""Export a side-by-side view of local Weings scores and SOVA Table 4 references.

This tool intentionally does not calculate improvement/deltas by default. Scores
are comparable only when dataset, modality, metric implementation, and evaluation
protocol match; sharing a metric name alone is insufficient.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "paper_reference_table4.json"
LOCAL_METRIC_MAP = {
    "knowledge_qa": ("accuracy_percent_over_all_cases", "knowledge_accuracy_pct"),
    "speech_recognition": ("corpus_wer_percent", None),
    "spoken_qa_multiple_choice": ("accuracy_percent_over_all_cases", "librisqa_accuracy_pct"),
    "spoken_qa_open_ended": ("accuracy_percent_over_all_cases", "spoken_squad_accuracy_pct"),
    "emotion_recognition": ("accuracy_percent_over_all_cases", "emotion_accuracy_pct"),
    "age_recognition": ("accuracy_percent_over_all_cases", "age_accuracy_pct"),
    "gender_recognition": ("accuracy_percent_over_all_cases", "gender_accuracy_pct"),
    "generation_consistency": ("corpus_wer_percent", "generation_consistency_wer_pct"),
    "generation_semantic": ("mean_gptscore", "generation_gptscore"),
    "generation_acoustic": ("mean_utmosv2_score", "generation_utmosv2_score"),
}


def build_rows(local_summary: dict[str, Any] | None, paper: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paper_models = paper.get("models", [])
    local_models = (local_summary or {}).get("models", [])
    for local_model in local_models:
        local_tasks = local_model.get("tasks", {})
        for task, (local_key, _) in LOCAL_METRIC_MAP.items():
            stats = local_tasks.get(task, {})
            rows.append({
                "row_type": "local_vs_paper_reference",
                "local_model": local_model.get("model_name", "Unspecified model"),
                "local_provider": local_model.get("provider", ""),
                "local_model_id": local_model.get("model", ""),
                "task": task,
                "local_cases": stats.get("cases_in_file", 0),
                "local_successful_cases": stats.get("successful_cases", 0),
                "local_dataset_coverage_vs_paper_pct": stats.get("dataset_coverage_vs_paper_percent"),
                "local_metric": local_key,
                "local_value": stats.get(local_key),
                "paper_model": "See paper_reference_table4.json",
                "paper_values": json.dumps({p.get("model"): p.get(LOCAL_METRIC_MAP[task][1]) for p in paper_models}, ensure_ascii=False) if LOCAL_METRIC_MAP[task][1] else "Separate dev/test clean/other split values are listed in paper_reference_table4.json",
                "delta_computed": False,
                "comparability_warning": "Descriptive only. Do not claim improvement over Table 4 without matching dataset split, modality, scoring rule, and protocol.",
            })
    if not rows:
        # Still produce a usable reference-only export before a local result file exists.
        for paper_model in paper_models:
            for task, (_, paper_key) in LOCAL_METRIC_MAP.items():
                rows.append({
                    "row_type": "paper_reference_only",
                    "local_model": "",
                    "task": task,
                    "paper_model": paper_model.get("model"),
                    "paper_value": paper_model.get(paper_key) if paper_key else None,
                    "delta_computed": False,
                    "comparability_warning": "This is a published reference value, not a Weings measurement.",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-summary", type=Path, default=None, help="summary.json emitted by score_sova_results.py")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "sova_reference_comparison.csv")
    args = parser.parse_args()
    paper = json.loads(args.reference.read_text(encoding="utf-8"))
    local = json.loads(args.local_summary.read_text(encoding="utf-8")) if args.local_summary else None
    rows = build_rows(local, paper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote side-by-side SOVA reference export: {args.output}")
    print("No improvement deltas were calculated; comparability must be established experimentally.")


if __name__ == "__main__":
    main()
