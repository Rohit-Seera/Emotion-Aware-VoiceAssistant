"""Attach separate metric outputs (currently UTMOSv2) to a model's SOVA run.

This creates one generation_acoustic record per acoustic prediction and never
turns missing/failed clips into a zero. In-place writes create a one-time backup.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for key in ("results", "records", "items", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list) or any(not isinstance(row, dict) for row in obj):
        raise ValueError(f"Expected a JSON array of records: {path}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Model run results.json")
    parser.add_argument("--acoustic-results", required=True, type=Path, help="acoustic_results.json from benchmark.acoustic_eval")
    parser.add_argument("--output", type=Path, default=None, help="Output path; defaults to --results with a backup")
    args = parser.parse_args()

    source = args.results.resolve()
    acoustic_path = args.acoustic_results.resolve()
    output = (args.output or source).resolve()
    base_rows = load_rows(source)
    acoustic_rows = load_rows(acoustic_path)

    base_by_case: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        cid = str(row.get("case_id") or row.get("id") or "")
        if cid and row.get("task") != "generation_acoustic":
            base_by_case.setdefault(cid, row)

    existing = {
        (str(r.get("model") or r.get("model_id") or ""), str(r.get("case_id") or ""), str(r.get("task") or ""))
        for r in base_rows
    }
    added = 0
    skipped = 0
    for audio in acoustic_rows:
        cid = str(audio.get("case_id") or "")
        if not cid:
            skipped += 1
            continue
        base = base_by_case.get(cid, {})
        model_id = str(audio.get("model") or audio.get("model_id") or base.get("model") or base.get("model_id") or "")
        model_name = str(audio.get("model_name") or base.get("model_name") or model_id or "Unknown model")
        key = (model_id, cid, "generation_acoustic")
        if key in existing:
            # Update a previously attached acoustic row in place when rerunning.
            for row in base_rows:
                row_key = (str(row.get("model") or row.get("model_id") or ""), str(row.get("case_id") or ""), str(row.get("task") or ""))
                if row_key == key:
                    row.update({
                        "success": bool(audio.get("success")) and audio.get("utmosv2") is not None,
                        "utmosv2_score": audio.get("utmosv2") if audio.get("success") else None,
                        "utmosv2_percent": audio.get("utmosv2_percent") if audio.get("success") else None,
                        "generated_audio_path": str(audio.get("audio_file") or ""),
                        "metric_method": "UTMOSv2_pretrained_model; score attached from acoustic_eval.py",
                        "error": audio.get("error"),
                    })
                    break
            skipped += 1
            continue
        ok = bool(audio.get("success")) and isinstance(audio.get("utmosv2"), (int, float))
        record = {
            "case_id": cid,
            "model": model_id,
            "model_id": model_id,
            "model_name": model_name,
            "task": "generation_acoustic",
            "success": ok,
            "answered": ok,
            "utmosv2_score": float(audio["utmosv2"]) if ok else None,
            "utmosv2_percent": float(audio["utmosv2_percent"]) if ok and audio.get("utmosv2_percent") is not None else None,
            "generated_audio_path": str(audio.get("audio_file") or ""),
            "metric_method": "UTMOSv2_pretrained_model; score attached from acoustic_eval.py",
            "error": audio.get("error"),
        }
        base_split = base.get("dataset_split") or base.get("split") or base.get("subset")
        if base_split:
            record["dataset_split"] = base_split
        base_rows.append(record)
        existing.add(key)
        added += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if output == source:
        backup = source.with_name(source.stem + ".pre_acoustic_attach" + source.suffix)
        if not backup.exists():
            shutil.copy2(source, backup)
        print(f"Backup: {backup}")
    output.write_text(json.dumps(base_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    successful = sum(r.get("task") == "generation_acoustic" and r.get("success") is True and r.get("utmosv2_score") is not None for r in base_rows)
    print(f"Wrote {output}; added={added}, updated/skipped-existing={skipped}, acoustic_success_rows={successful}")


if __name__ == "__main__":
    main()
