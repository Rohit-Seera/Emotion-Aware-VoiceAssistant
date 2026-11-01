"""Free local Whisper transcription and consistency-WER for generated audio.

Requires `pip install faster-whisper`. Results go to a separate JSON and use
local_* keys; they are not confused with a paper's Groq Whisper-large-v3 run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.sova_metrics import corpus_wer

GENERATION_TASKS = {"generation", "generation_response", "generation_semantic", "speech_generation", "open_ended"}


def resolve_audio(raw: Any, source: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    options = [path] if path.is_absolute() else [source.parent / path, Path.cwd() / path]
    for option in options:
        if option.is_file():
            return option.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Model results JSON list")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <input>.local_asr.json; source is not overwritten")
    parser.add_argument("--model", default="large-v3", help="Use paper-named Whisper-large-v3; may be slow on CPU")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default="int8", help="e.g. int8 for CPU, float16 for compatible CUDA")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language", default="en", help="Use 'auto' to let Whisper detect language")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Missing faster-whisper. Install it with: python -m pip install faster-whisper") from exc

    source = args.input.resolve()
    output = (args.output or source.with_name(source.stem + ".local_asr.json")).resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit("Input must be a JSON list of result objects.")

    print(f"Loading local faster-whisper model={args.model}, device={args.device}, compute_type={args.compute_type} ...", flush=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    paired: list[tuple[str, str]] = []
    out_rows: list[dict[str, Any]] = []
    attempted = succeeded = 0
    for row in rows:
        task = str(row.get("task", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if task not in GENERATION_TASKS or row.get("success") is False:
            continue
        audio = resolve_audio(row.get("response_audio_path") or row.get("generated_audio_path") or row.get("audio_file"), source)
        text_response = str(row.get("model_text_output") or row.get("response_text") or row.get("prediction") or "").strip()
        if audio is None:
            continue
        attempted += 1
        try:
            language = None if args.language.lower() == "auto" else args.language
            segments, info = model.transcribe(str(audio), language=language, vad_filter=True)
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            result_row = {
                "case_id": row.get("case_id") or row.get("id"),
                "model_id": row.get("model_id") or row.get("model") or row.get("model_name", "unknown"),
                "task": "generation_consistency_local",
                "success": bool(transcript and text_response),
                "model_text_output": text_response,
                "generated_transcript_local": transcript,
                "local_asr_model": args.model,
                "local_asr_device": args.device,
                "local_asr_compute_type": args.compute_type,
                "local_detected_language": getattr(info, "language", None),
                "audio_path": str(audio),
                "metric_method": "local_faster_whisper_transcript; consistency_WER_is_proxy_not_paper_provider_exact",
            }
            if transcript and text_response:
                result_row["local_consistency_wer_percent"] = corpus_wer([(text_response, transcript)])["wer_percent"]
                paired.append((text_response, transcript))
            else:
                result_row["error"] = "Missing response text or ASR transcript; WER not computed."
            out_rows.append(result_row)
            succeeded += int(bool(transcript))
            print(f"[{attempted}] {result_row['case_id']}: {'ok' if transcript else 'empty transcript'}", flush=True)
        except Exception as exc:
            out_rows.append({
                "case_id": row.get("case_id") or row.get("id"), "task": "generation_consistency_local",
                "success": False, "audio_path": str(audio), "local_asr_model": args.model,
                "error": f"{type(exc).__name__}: {exc}",
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = corpus_wer(paired) if paired else {"wer_percent": None, "utterances": 0}
    summary = {
        "source": str(source), "output": str(output), "asr_model": args.model,
        "device": args.device, "compute_type": args.compute_type,
        "attempted_audio_files": attempted, "successful_transcriptions": succeeded,
        "paired_text_audio_cases": len(paired), "local_consistency_corpus_wer_percent": stats.get("wer_percent"),
        "wer_aggregation": stats.get("aggregation"),
        "note": "This uses local Whisper inference. With --model large-v3 the checkpoint name matches the paper; runtime/provider parity with the authors is not established.",
    }
    summary_path = output.with_name(output.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved local transcripts/consistency: {output}")
    print(f"Summary: attempted={attempted}, transcribed={succeeded}, paired={len(paired)}, local corpus WER={stats.get('wer_percent')}")


if __name__ == "__main__":
    main()
