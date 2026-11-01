"""Offline preflight for the SOVA-compatible evaluation package.

No model is loaded, no API is called, and no credentials are printed.
Use --manifest to verify that every audio path in a proposed inference manifest exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("items", "records", "results", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def count_records(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return len(read_records(path))
    except Exception:
        return None


def inspect_manifest(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "records": 0,
            "audio_paths_checked": 0,
            "missing_audio_paths": [],
            "ok": False,
            "error": "Manifest file does not exist.",
        }
    try:
        rows = read_records(path)
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "records": 0,
            "audio_paths_checked": 0,
            "missing_audio_paths": [],
            "ok": False,
            "error": f"Cannot parse manifest: {type(exc).__name__}: {exc}",
        }
    missing: list[dict[str, str]] = []
    checked = 0
    for row in rows:
        raw = row.get("audio_path") or row.get("audio_file") or row.get("speech_path") or row.get("speech")
        if not raw or not isinstance(raw, str):
            missing.append({
                "case_id": str(row.get("case_id") or row.get("id") or ""),
                "path": "",
            })
            continue
        checked += 1
        audio = Path(raw)
        if not audio.is_absolute():
            # The adapter resolves relative paths against the project root.
            audio = (ROOT / audio).resolve()
        if not audio.is_file():
            missing.append({
                "case_id": str(row.get("case_id") or row.get("id") or ""),
                "path": str(audio),
            })
    return {
        "path": str(path),
        "exists": True,
        "records": len(rows),
        "audio_paths_checked": checked,
        "missing_audio_count": len(missing),
        "missing_audio_paths": missing[:50],
        "ok": bool(rows) and not missing,
        "error": None if rows and not missing else "Every input case for this runner needs an existing audio_path/audio_file/speech_path/speech file.",
    }


def existing_results_inventory() -> list[dict[str, Any]]:
    candidates = [
        ("benchmark/results_sova_triviaqa_500/results.json", "TriviaQA custom 500-case pilot; GPT-OSS; prior run"),
        ("benchmark/results_sova_triviaqa_test_v2/results.json", "TriviaQA custom 100-case pilot v2; GPT-OSS; prior run"),
        ("benchmark/results_sova_librispeech_500/results.json", "LibriSpeech 500-sample direct Groq Whisper STT pilot"),
        ("benchmark/results_sova_spoken_squad_500/results.json", "Spoken SQuAD partial pilot; prior summary reports only 16/500 successful"),
        ("benchmark/results_generation/results.json", "20 custom speech-generation prompts; not the 3745-item SOVA subset"),
        ("benchmark/results_generation/sova_generation_merged.json", "Merged historical generation pilot; inspect task labels before using"),
        ("benchmark/audit/pilot_150/scored_cases.json", "Custom 150-example language-task pilot; not acoustic emotion recognition"),
    ]
    out = []
    for rel, note in candidates:
        path = ROOT / rel
        rows = count_records(path)
        summary_path = path.with_name("summary.json")
        summary_exists = summary_path.exists()
        out.append({
            "relative_path": rel,
            "exists": path.is_file(),
            "record_count": rows,
            "summary_path": str(summary_path.relative_to(ROOT)) if summary_exists else "",
            "note": note,
        })
    return out


def write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["relative_path", "exists", "record_count", "summary_path", "note"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional proposed audio manifest to validate.")
    parser.add_argument("--output", type=Path, default=BENCH / "audit" / "setup_check.json")
    parser.add_argument("--strict-manifest", action="store_true", help="Exit code 2 if the requested manifest is incomplete/missing.")
    args = parser.parse_args()

    registry = json.loads((BENCH / "sova_model_registry.json").read_text(encoding="utf-8"))
    protocol = json.loads((BENCH / "sova_protocol.json").read_text(encoding="utf-8"))
    reference = json.loads((BENCH / "paper_reference_table4.json").read_text(encoding="utf-8"))

    triv_json = BENCH / "datasets" / "sova_triviaqa_6840.json"
    triv_jsonl = BENCH / "sova_official" / "triviaqa_6840.jsonl"
    manifest_path = BENCH / "manifests" / "librispeech_test_clean_5.json"

    manifest_check = inspect_manifest(args.manifest) if args.manifest else None
    models = [
        {
            "id": model.get("id"),
            "name": model.get("name"),
            "adapter": model.get("adapter"),
            "is_registered": True,
            "weights_bundled": False,
            "note": "Registered adapter; official checkpoint/runtime must be installed separately.",
        }
        for model in registry.get("models", [])
    ]
    excluded = registry.get("excluded_models", [])
    inventory = existing_results_inventory()
    inventory_path = BENCH / "audit" / "existing_results_inventory.csv"
    write_inventory_csv(inventory_path, inventory)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "mode": "offline_preflight_only_no_model_inference_no_api_calls",
        "benchmark_paper": protocol.get("paper"),
        "paper_top_level_task_groups": protocol.get("paper_total_tasks"),
        "paper_metric_rows": len(protocol.get("tasks", [])) + 2,  # ASR split breakdown and 3 gen metrics are represented in task map.
        "published_table4_reference_model_rows": len(reference.get("models", [])),
        "model_registry": models,
        "excluded_reference_models": excluded,
        "source_dataset_assets": {
            "triviaqa_6840_json": {
                "path": str(triv_json.relative_to(ROOT)),
                "exists": triv_json.is_file(),
                "records": count_records(triv_json),
                "modality_note": "Text-only source rows. Paper converts TriviaQA questions to speech; these rows are not directly valid inputs for speech-only model adapters.",
            },
            "triviaqa_6840_jsonl": {
                "path": str(triv_jsonl.relative_to(ROOT)),
                "exists": triv_jsonl.is_file(),
                "records": count_records(triv_jsonl),
                "modality_note": "Text-only source rows. Speech synthesis/audio validation still required.",
            },
            "bundled_source_dataset_audio_files": 0,
            "bundled_model_checkpoints": False,
            "bundled_older_librispeech_manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "exists": manifest_path.is_file(),
                "path_status": "The manifest may contain machine-specific absolute paths; regenerate it on this computer.",
            },
        },
        "existing_result_inventory": inventory,
        "existing_result_inventory_csv": str(inventory_path.relative_to(ROOT)),
        "proposed_manifest_check": manifest_check,
        "important_limitations": [
            "This ZIP does not bundle the SOVA speech datasets' audio files or the six paper models' weights.",
            "Do not use the text-only TriviaQA JSON directly for audio-input baseline scoring. Synthesize question audio using a declared protocol first.",
            "IEMOCAP access/licensing must be obtained separately; map its labels to the paper's five labels before scoring.",
            "The previous Groq Whisper / GPT-OSS pilot scripts are not the same as running all Table-4 paper baselines.",
            "The paper's official Cascade uses Whisper-large-v3 + GPT-4o-mini and remains reference-only; no substitute is labelled as the official Cascade.",
            "Run smoke tests first and preserve held-out evaluation cases; do not train on the final test split.",
        ],
    }

    out = args.output
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Offline SOVA setup check saved: {out}")
    print(f"Registered runnable adapters: {len(models)}")
    print(f"Published reference rows: {len(reference.get('models', []))}")
    for key in ("triviaqa_6840_json", "triviaqa_6840_jsonl"):
        item = report["source_dataset_assets"][key]
        print(f"{item['path']}: exists={item['exists']}, records={item['records']}")
    print(f"Historical result inventory saved: {inventory_path}")
    if manifest_check is not None:
        print(f"Manifest records={manifest_check['records']}; missing audio paths={manifest_check.get('missing_audio_count', len(manifest_check.get('missing_audio_paths', [])))}")
        if args.strict_manifest and not manifest_check["ok"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
