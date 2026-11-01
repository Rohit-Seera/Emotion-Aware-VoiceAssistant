"""Attach paths to pre-synthesized/source audio files and report missing cases.

This script does not synthesize audio. If the source dataset is text-only (e.g.
TriviaQA/Alpaca), first synthesize its speech inputs using a declared TTS/voice
protocol and then point this tool at that audio directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def _read(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("items", "records", "data", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("Input must contain a JSON list of records.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-key", default="audio_path", help="Field to populate (default audio_path).")
    parser.add_argument("--keep-existing", action="store_true", help="Keep an existing valid audio path.")
    args = parser.parse_args()
    items = _read(args.input)
    root = args.audio_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Audio directory does not exist: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            index.setdefault(path.stem.casefold(), path.resolve())
    attached = missing = 0
    for item in items:
        cid = str(item.get("case_id") or item.get("id") or item.get("question_id") or item.get("example_id") or "")
        if args.keep_existing:
            raw = item.get(args.audio_key) or item.get("speech") or item.get("audio_file")
            if raw:
                path = Path(str(raw))
                if not path.is_absolute():
                    path = (args.input.parent / path).resolve()
                if path.is_file():
                    item[args.audio_key] = str(path)
                    attached += 1
                    continue
        candidates = [cid, cid + ".wav", f"{cid}_audio", f"{cid}-audio", f"{cid}_speech"]
        found = next((index[key.casefold()] for key in candidates if key.casefold() in index), None)
        if found:
            item[args.audio_key] = str(found)
            attached += 1
        else:
            item[args.audio_key] = ""
            missing += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    print(f"Records={len(items)}, audio attached={attached}, missing={missing}")
    if missing:
        print("Missing audio rows stay in the manifest so execution coverage can be reported honestly.")


if __name__ == "__main__":
    main()
