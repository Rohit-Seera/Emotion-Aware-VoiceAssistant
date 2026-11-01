"""Run one registered SOVA-Bench model on an audio-backed JSON dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.sova_model_adapters import ROOT, load_registry, run_model


def load_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Source-fetch helpers save JSONL; accept the format explicitly rather than
        # forcing users to hand-convert files before manifest validation.
        data = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON/JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object.")
            data.append(row)
    if isinstance(data, dict):
        for key in ("items", "records", "results", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError("Input JSON must be a list of case objects, JSONL objects, or an object containing items/records/results/data.")
    return data


def main() -> None:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model id from benchmark/sova_model_registry.json, or 'all'.")
    parser.add_argument("--input", type=Path, default=None, help="Audio-backed cases JSON. Every speech input requires audio_path or speech; not needed with --list-models.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark" / "runs")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path; default is output-dir/<model-id>/results.json")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional pilot limit for test runs; omit for all input cases.")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        for model in registry["models"]:
            status = "user model" if model['id'] == "weings_ai" else "paper open baseline"
            print(f"{model['id']:16} | {model['name']} | adapter={model['adapter']} | {status}")
        for model in registry.get("excluded_models", []):
            print(f"{model['id']:16} | {model['name']} | EXCLUDED: {model.get('exclusion_reason', 'not in free runner registry')}")
        return
    if args.input is None:
        parser.error("--input is required unless --list-models is used.")
    items = load_items(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        items = items[:args.limit]
    model_ids = [m["id"] for m in registry["models"]] if args.model == "all" else [args.model]
    overall: dict[str, Any] = {"input": str(args.input.resolve()), "num_input_cases": len(items), "models": []}
    for model_id in model_ids:
        print(f"Running {model_id} on {len(items)} case(s)...", flush=True)
        rows = run_model(model_id, items, args.output_dir, timeout_s=args.timeout)
        good = sum(bool(x.get("success")) for x in rows)
        excluded = sum(x.get("measurement_status") == "not_evaluated_by_paper" for x in rows)
        failed = sum((not bool(x.get("success"))) and x.get("measurement_status") != "not_evaluated_by_paper" for x in rows)
        model_dir = args.output_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        out = args.output if args.output and len(model_ids) == 1 else model_dir / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "model_id": model_id,
            "model_name": rows[0].get("model_name") if rows else model_id,
            "attempted_cases": len(rows) - excluded,
            "successful_cases": good,
            "failed_cases": failed,
            "excluded_not_reported_in_paper": excluded,
            "input_cases_seen": len(rows),
            "execution_coverage_percent": round(100 * good / (len(rows) - excluded), 4) if (len(rows) - excluded) else None,
            "results_path": str(out.resolve()),
            "note": "These are new measurements only for records with actual success=true; reference values are stored separately.",
        }
        (out.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        overall["models"].append(summary)
        print(f"  {good}/{len(rows)-excluded} success, {failed} failed, {excluded} excluded (not in paper Table 4) -> {out}", flush=True)
    if len(model_ids) > 1:
        index_path = args.output_dir / "run_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Run index: {index_path}")


if __name__ == "__main__":
    main()
