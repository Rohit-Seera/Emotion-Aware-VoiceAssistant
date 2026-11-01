"""Run the complete four-split SOVA LibriSpeech ASR set through WEINGS' STT.

This runner never truncates or samples the manifest. It uses the running
assistant's authenticated transcription endpoint, saves resumable JSONL case
records, and calculates corpus WER and split coverage.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from benchmark.prepare_librispeech_manifest import build_manifest
from benchmark.sova_metrics import corpus_wer


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "SOVA_BENCH_SUPERVISOR_PACK" / "Datasets" / "LibriSpeech"
RUN_DIR = (
    ROOT
    / "SOVA_BENCH_SUPERVISOR_PACK"
    / "Results"
    / "SOVA_BENCH_FULL_20261030"
    / "weings_ai"
    / "speech_recognition"
)
SPLITS = ["dev-clean", "dev-other", "test-clean", "test-other"]
PAUSE_SECONDS = float(os.getenv("SOVA_ASR_PAUSE_SECONDS", "3.1"))
SAVE_EVERY = 25
MAX_CONSECUTIVE_PROVIDER_FAILURES = 3


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cases
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid results JSONL at line {line_number}: {exc}") from exc
            cases[str(row["case_id"])] = row
    return cases


def summarize(
    manifest: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    *,
    finished: bool,
    backend_url: str,
    stt_model: str,
) -> dict[str, Any]:
    success_rows = [r for r in rows.values() if r.get("success")]
    failed_rows = [r for r in rows.values() if not r.get("success")]
    targets = {str(x["case_id"]): x for x in manifest}
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        split_targets = [x for x in manifest if x["dataset_split"] == split]
        split_results = [rows[str(x["case_id"])] for x in split_targets if str(x["case_id"]) in rows]
        split_success = [x for x in split_results if x.get("success")]
        split_summary[split] = {
            "target_cases": len(split_targets),
            "attempted_cases": len(split_results),
            "successful_cases": len(split_success),
            "failed_cases": len(split_results) - len(split_success),
            "corpus_wer": corpus_wer(
                (targets[str(x["case_id"])]["reference_text"], x.get("prediction", ""))
                for x in split_success
            ),
        }
    return {
        "benchmark": "SOVA-Bench compatible full LibriSpeech recognition run",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "weings_ai",
        "component_measured": "configured speech-to-text component only",
        "stt_model": stt_model,
        "backend_url": backend_url,
        "splits": SPLITS,
        "sampled": False,
        "manifest_cases": len(manifest),
        "attempted_cases": len(rows),
        "successful_cases": len(success_rows),
        "failed_cases": len(failed_rows),
        "all_targets_attempted": len(rows) == len(manifest),
        "all_targets_successful": len(success_rows) == len(manifest),
        "run_finished": finished,
        "coverage_percent": round(100 * len(success_rows) / len(manifest), 4) if manifest else None,
        "corpus_wer": corpus_wer(
            (targets[str(x["case_id"])]["reference_text"], x.get("prediction", ""))
            for x in success_rows
        ),
        "by_split": split_summary,
        "metric_formula": "corpus WER (%) = 100 * (S + D + I) / reference words; failures and missing cases are reported separately",
        "notes": [
            "No case limit or random sampling is applied by this runner.",
            "The user requested a full-data run; requests are issued sequentially to respect provider rate limits.",
            "This measures WEINGS' configured STT endpoint, not end-to-end assistant response generation or TTS.",
        ],
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    backend_url = os.getenv("SOVA_WEINGS_URL", "http://127.0.0.1:8000").rstrip("/")
    stt_model = os.getenv("GROQ_STT_MODEL", "configured-by-backend")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_DIR / "librispeech_full_manifest.json"
    results_path = RUN_DIR / "case_results.jsonl"
    summary_path = RUN_DIR / "summary.json"
    manifest, manifest_summary = build_manifest(DATA_ROOT, SPLITS, limit_per_split=None)
    absent = [s for s in SPLITS if not (Path(manifest_summary["dataset_root"]) / s).is_dir()]
    if absent:
        raise SystemExit(f"Full four-split run refused; split directories missing: {absent}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN_DIR / "manifest_summary.json").write_text(
        json.dumps(manifest_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = load_cases(results_path)
    print(
        f"Full manifest: {len(manifest)} utterances across {', '.join(SPLITS)}; "
        f"already successful: {sum(bool(r.get('success')) for r in results.values())}.",
        flush=True,
    )

    with httpx.Client(base_url=backend_url, timeout=100.0) as client:
        token_response = client.get("/api/v1/demo/token")
        token_response.raise_for_status()
        token = token_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        health = client.get("/api/v1/health/ready")
        health.raise_for_status()
        with results_path.open("a", encoding="utf-8", buffering=1) as output:
            n_new = 0
            consecutive_provider_failures = 0
            run_finished = True
            for index, item in enumerate(manifest, 1):
                case_id = str(item["case_id"])
                if results.get(case_id, {}).get("success"):
                    continue
                audio_path = Path(item["audio_path"])
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "model_id": "weings_ai",
                    "task": "speech_recognition",
                    "dataset": "LibriSpeech",
                    "dataset_split": item["dataset_split"],
                    "reference_text": item["reference_text"],
                    "audio_path": str(audio_path),
                    "success": False,
                    "prediction": "",
                }
                try:
                    response = client.post(
                        "/api/v1/voice/transcriptions?language=en",
                        headers={**headers, "Content-Type": "audio/flac"},
                        content=audio_path.read_bytes(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    row.update(
                        success=True,
                        prediction=str(payload.get("transcript", "")).strip(),
                        detected_language=payload.get("language"),
                        audio_duration_seconds=payload.get("duration_seconds"),
                    )
                    consecutive_provider_failures = 0
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 429 or (status is not None and status >= 500):
                        consecutive_provider_failures += 1
                    else:
                        consecutive_provider_failures = 0
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                results[case_id] = row
                n_new += 1
                if n_new % SAVE_EVERY == 0 or index == len(manifest):
                    summary = summarize(
                        manifest, results, finished=False,
                        backend_url=backend_url, stt_model=stt_model,
                    )
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(
                        f"Progress {index}/{len(manifest)}; success {summary['successful_cases']}; "
                        f"failed {summary['failed_cases']}; corpus WER "
                        f"{summary['corpus_wer'].get('wer_percent')}",
                        flush=True,
                    )
                if consecutive_provider_failures >= MAX_CONSECUTIVE_PROVIDER_FAILURES:
                    run_finished = False
                    print(
                        "Stopping after repeated provider/server failures; saved cases can be resumed later.",
                        flush=True,
                    )
                    break
                time.sleep(PAUSE_SECONDS)

    summary = summarize(
        manifest, results, finished=run_finished,
        backend_url=backend_url, stt_model=stt_model,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Case records: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Successful cases are saved; rerun to resume.", file=sys.stderr)
        raise SystemExit(130)
