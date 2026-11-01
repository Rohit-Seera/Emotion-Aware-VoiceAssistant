"""Expand the saved pilot to reproducible 55-case task manifests."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import wave
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
INPUT = RUN / "inputs"
SOURCE = Path(os.environ["SOVA_SOURCE_RESULTS_DIR"])
SOURCE_INPUT = SOURCE / "inputs"
PILOT_INPUT = SOURCE_INPUT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_extra(task: str, base: list[dict], pool: list[dict], count: int = 30) -> list[dict]:
    existing = {str(row["case_id"]) for row in base}
    candidates = [row for row in pool if str(row["case_id"]) not in existing]
    if len(candidates) < count:
        raise RuntimeError(f"{task}: need {count} new cases; found {len(candidates)}")
    seed = 20261030 + sum(map(ord, task))
    return random.Random(seed).sample(candidates, count)


def save_manifest(task: str, rows: list[dict]) -> None:
    path = INPUT / f"{task}_eval_sample_55.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{task}: {len(rows)} cases -> {path.name}")


def copy_spoken_audio(rows: list[dict]) -> list[dict]:
    result = []
    audio_hashes = {}
    dest_root = INPUT / "audio" / "spoken_squad"
    for original in rows:
        row = dict(original)
        source_path = Path(row["audio_path"])
        if not source_path.is_absolute():
            source_path = SOURCE_INPUT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        name = f"{row['case_id']}.wav"
        dest = dest_root / name
        shutil.copy2(source_path, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        expected = row.get("audio_sha256")
        if expected and digest.lower() != str(expected).lower():
            raise RuntimeError(f"Audio checksum mismatch for {row['case_id']}")
        row["audio_path"] = f"audio/spoken_squad/{name}"
        row["audio_bytes"] = dest.stat().st_size
        row["audio_sha256"] = digest
        with wave.open(str(dest), "rb") as wav:
            row["audio_duration_seconds"] = round(wav.getnframes() / wav.getframerate(), 3)
        audio_hashes[name] = digest
        result.append(row)
    (INPUT / "spoken_squad_audio_sha256.json").write_text(
        json.dumps(audio_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    trivia_base = read_json(PILOT_INPUT / "triviaqa_eval_sample_25.json")
    trivia_pool = read_json(SOURCE_INPUT / "sova_triviaqa_6840.json")
    save_manifest("triviaqa", trivia_base + pick_extra("triviaqa", trivia_base, trivia_pool))

    libri_base = read_json(PILOT_INPUT / "librisqa_eval_sample_25.json")
    libri_pool = read_json(PILOT_INPUT / "librisqa_asr_overlap_sample_150.json")
    save_manifest("librisqa", libri_base + pick_extra("librisqa", libri_base, libri_pool))

    spoken_base = read_json(PILOT_INPUT / "spoken_squad_eval_sample_25.json")
    spoken_pool = read_json(SOURCE_INPUT / "spoken_squad_audio_sample_500.json")
    expanded_spoken = spoken_base + pick_extra("spoken_squad", spoken_base, spoken_pool)
    save_manifest("spoken_squad", copy_spoken_audio(expanded_spoken))

    alpaca_base = read_json(PILOT_INPUT / "alpaca_eval_sample_25.json")
    alpaca_pool = read_jsonl(SOURCE_INPUT / "alpaca_sova_full_3745.jsonl")
    save_manifest("alpaca", alpaca_base + pick_extra("alpaca", alpaca_base, alpaca_pool))

    source_hashes = read_json(SOURCE_INPUT / "spoken_squad_full" / "download_manifest_sha256.json")
    (INPUT / "spoken_squad_source_hashes.json").write_text(
        json.dumps(source_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    card = SOURCE_INPUT / "spoken_squad_full" / "DATASET_CARD_README.md"
    if card.is_file():
        shutil.copy2(card, INPUT / "spoken_squad_source_card.md")


if __name__ == "__main__":
    main()
