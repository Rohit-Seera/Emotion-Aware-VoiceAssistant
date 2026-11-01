"""Build an audio-backed LibriSpeech manifest for SOVA-style ASR evaluation.

This script does NOT download data or call a model. It reads the standard
LibriSpeech directory layout containing <speaker>-<chapter>.trans.txt files,
then writes a JSON list with absolute audio paths for the registered adapters.

For a quick smoke manifest use --limit-per-split 5 (default).
For a full four-split run, pass --all-examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SPLITS = ("test-clean",)
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg", ".m4a")


def _find_dataset_root(raw: Path) -> Path:
    root = raw.expanduser().resolve()
    candidates = [
        root,
        root / "LibriSpeech",
        root / "data" / "LibriSpeech",
    ]
    for candidate in candidates:
        if any((candidate / split).is_dir() for split in (
            "dev-clean", "dev-other", "test-clean", "test-other"
        )):
            return candidate
    raise SystemExit(
        f"Could not find LibriSpeech split folders under {root}. "
        "Expected dev-clean/dev-other/test-clean/test-other directly inside "
        "the root, or under a child named LibriSpeech. Example: "
        "--root D:\\datasets\\LibriSpeech"
    )


def _audio_for_utterance(transcript_file: Path, utterance_id: str) -> Path | None:
    # Standard layout puts each transcript and audio files in the same chapter folder.
    for suffix in AUDIO_SUFFIXES:
        candidate = transcript_file.parent / f"{utterance_id}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_manifest(
    root: Path,
    splits: list[str],
    limit_per_split: int | None = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = _find_dataset_root(root)
    records: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, int]] = {}
    seen: set[str] = set()

    for split in splits:
        split_root = dataset_root / split
        if not split_root.is_dir():
            split_stats[split] = {"transcript_lines": 0, "audio_found": 0, "audio_missing": 0}
            continue

        transcript_files = sorted(split_root.rglob("*.trans.txt"))
        transcript_lines = 0
        found = 0
        missing = 0
        for transcript_file in transcript_files:
            for raw_line in transcript_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utterance_id, reference = parts
                transcript_lines += 1
                if utterance_id in seen:
                    continue
                audio_path = _audio_for_utterance(transcript_file, utterance_id)
                if audio_path is None:
                    missing += 1
                    continue
                records.append({
                    "case_id": utterance_id,
                    "task": "speech_recognition",
                    "dataset": "LibriSpeech",
                    "dataset_split": split,
                    "audio_path": str(audio_path),
                    "reference_text": reference.strip(),
                    "input_modality": "speech",
                    "metric_method": "LibriSpeech reference transcript; corpus WER after inference",
                })
                seen.add(utterance_id)
                found += 1
                if limit_per_split is not None and found >= limit_per_split:
                    break
            if limit_per_split is not None and found >= limit_per_split:
                break
        split_stats[split] = {
            "transcript_lines_read": transcript_lines,
            "audio_found": found,
            "audio_missing": missing,
            "transcript_files": len(transcript_files),
        }

    summary = {
        "dataset_root": str(dataset_root),
        "requested_splits": splits,
        "limit_per_split": limit_per_split,
        "records_written": len(records),
        "split_stats": split_stats,
        "note": (
            "This is a deterministic smoke subset." if limit_per_split is not None
            else "This manifest includes every transcript line with an audio file found; verify exact official split release before reporting."
        ),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="LibriSpeech root or its parent folder.")
    parser.add_argument("--splits", nargs="+", choices=["dev-clean", "dev-other", "test-clean", "test-other"], default=list(DEFAULT_SPLITS))
    parser.add_argument("--limit-per-split", type=int, default=5, help="Smoke-test limit per split (default: 5).")
    parser.add_argument("--all-examples", action="store_true", help="Use all valid audio/transcript pairs in selected splits.")
    parser.add_argument("--output", type=Path, default=Path("benchmark/manifests/librispeech_smoke.json"))
    args = parser.parse_args()

    if args.limit_per_split < 1 and not args.all_examples:
        parser.error("--limit-per-split must be >= 1; use --all-examples for a full split.")
    limit = None if args.all_examples else args.limit_per_split
    records, summary = build_manifest(args.root, args.splits, limit)
    if not records:
        raise SystemExit(
            "No audio/transcript pairs were found. Check that .flac/.wav files and matching *.trans.txt files exist."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output.resolve()}")
    print(f"Manifest summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
