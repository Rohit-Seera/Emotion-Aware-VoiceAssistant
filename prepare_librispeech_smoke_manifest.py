import json
from pathlib import Path

possible_roots = [
    Path("data/LibriSpeech/LibriSpeech/test-clean"),
    Path("data/LibriSpeech/test-clean"),
]

root = next((p for p in possible_roots if p.exists()), None)

if root is None:
    raise SystemExit(
        "LibriSpeech test-clean folder not found. "
        "Check that the archive extracted correctly."
    )

items = []

for transcript_file in sorted(root.rglob("*.trans.txt")):
    for line in transcript_file.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue

        case_id, reference = parts
        audio_path = transcript_file.parent / f"{case_id}.flac"

        if not audio_path.exists():
            continue

        items.append({
            "case_id": case_id,
            "task": "speech_recognition",
            "dataset_split": "test-clean",
            "audio_path": str(audio_path.resolve()),
            "reference_text": reference,
        })

        if len(items) == 5:
            break

    if len(items) == 5:
        break

if not items:
    raise SystemExit("No matching audio/transcript examples found.")

output = Path(
    "benchmark/manifests/librispeech_test_clean_5.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(items, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Created: {output.resolve()}")
print(f"Records: {len(items)}")
print("Dataset split: test-clean")