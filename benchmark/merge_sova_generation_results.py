"""Merge Weings text, generated-audio, semantic-judge and UTMOSv2 outputs.

This creates one normalized JSON file for ``score_sova_results.py``. The custom
C/R/H/I evaluator is kept as a separate custom metric; it is not called GPTScore.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "responses": ROOT / "results_generation" / "results.json",
    "audio": ROOT / "results_generation" / "audio_results.json",
    "semantic": ROOT / "results_generation" / "semantic_results.json",
    "acoustic": ROOT / "results_generation" / "acoustic_results.json",
    "output": ROOT / "results_generation" / "sova_generation_merged.json",
}


def _read(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if optional:
            return []
        raise FileNotFoundError(f"Missing input file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [row for row in value if isinstance(row, dict)]


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id") or row.get("id") or ""): row for row in rows if row.get("case_id") or row.get("id")}


def merge(
    responses: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    acoustic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audio_by_id, semantic_by_id, acoustic_by_id = map(_index, (audio, semantic, acoustic))
    merged: list[dict[str, Any]] = []
    for response in responses:
        case_id = str(response.get("case_id") or response.get("id") or "")
        if not case_id:
            continue
        common = {
            "case_id": case_id,
            "model_name": response.get("model_name", "Unspecified model"),
            "provider": response.get("provider", ""),
            "model": response.get("model", ""),
        }
        merged.append({
            **common,
            "task": "generation_response",
            "prompt": response.get("prompt"),
            "prediction": response.get("output", response.get("prediction", "")),
            "success": response.get("success", False),
            "answered": response.get("answered", bool(response.get("output", "").strip() if isinstance(response.get("output"), str) else response.get("output"))),
            "metric_method": "custom_20_prompt_generation_response_not_a_standalone_SOVA_metric",
        })

        audio_row = audio_by_id.get(case_id, {})
        if audio_row:
            merged.append({
                **common,
                "task": "generation_consistency",
                "reference_text": audio_row.get("reference_text", response.get("output", "")),
                "generated_transcript": audio_row.get("generated_transcript", ""),
                "success": bool(response.get("success", False) and audio_row.get("success", False)),
                "metric_method": "WER_between_response_text_and_ASR_transcription_of_generated_speech",
                "error": audio_row.get("error"),
            })

        semantic_row = semantic_by_id.get(case_id, {})
        if semantic_row:
            merged.append({
                **common,
                "task": "generation_semantic",
                "success": semantic_row.get("success", False),
                "custom_crhi_mean_score_5": semantic_row.get("mean_score_5"),
                "custom_crhi_score_pct": semantic_row.get("semantic_score_pct"),
                "correctness": semantic_row.get("correctness"),
                "relevance": semantic_row.get("relevance"),
                "coherence": semantic_row.get("coherence"),
                "instruction_following": semantic_row.get("instruction_following"),
                "metric_method": "custom_CRHI_Likert_1_to_5_not_paper_GPTScore",
                "error": semantic_row.get("error"),
            })

        acoustic_row = acoustic_by_id.get(case_id, {})
        if acoustic_row:
            merged.append({
                **common,
                "task": "generation_acoustic",
                "success": acoustic_row.get("success", False),
                "utmosv2_score": acoustic_row.get("utmosv2", acoustic_row.get("utmosv2_score")),
                "audio_file": acoustic_row.get("audio_file"),
                "metric_method": "UTMOSv2_raw_score",
                "error": acoustic_row.get("error"),
            })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=DEFAULTS["responses"])
    parser.add_argument("--audio", type=Path, default=DEFAULTS["audio"])
    parser.add_argument("--semantic", type=Path, default=DEFAULTS["semantic"])
    parser.add_argument("--acoustic", type=Path, default=DEFAULTS["acoustic"])
    parser.add_argument("--output", type=Path, default=DEFAULTS["output"])
    args = parser.parse_args()
    result = merge(
        _read(args.responses), _read(args.audio, optional=True),
        _read(args.semantic, optional=True), _read(args.acoustic, optional=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in result:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    print(json.dumps({"output": str(args.output), "rows": len(result), "task_rows": counts}, indent=2))


if __name__ == "__main__":
    main()
