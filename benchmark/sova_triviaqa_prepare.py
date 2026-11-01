from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("benchmark/sova_official")
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT / "triviaqa_6840.jsonl"

ds = load_dataset(
    "mandarjoshi/trivia_qa",
    "unfiltered",
    split="validation",
    streaming=True,
)

count = 0

with OUTPUT.open("w", encoding="utf-8") as f:
    for row in ds:
        # Keep only fields needed for the benchmark.
        item = {
            "case_id": f"triviaqa_{count+1:05d}",
            "task": "knowledge",
            "question": row["question"],
            "answer": row["answer"],
        }

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            ) + "\n"
        )

        count += 1

        if count % 500 == 0:
            print(f"Saved {count}/6840")

        if count >= 6840:
            break

print(f"\nDONE: {count} cases")
print(f"Saved to: {OUTPUT}")