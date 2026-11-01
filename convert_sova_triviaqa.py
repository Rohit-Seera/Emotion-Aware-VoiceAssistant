
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("benchmark/sova_official/triviaqa_6840.jsonl")
DST = Path("benchmark/datasets/sova_triviaqa_6840.json")

rows = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        raw = json.loads(line)
        ans = raw.get("answer")

        aliases = []
        value = None

        if isinstance(ans, dict):
            value = ans.get("value")

            for key in ("aliases", "normalized_aliases"):
                vals = ans.get(key)
                if isinstance(vals, list):
                    aliases.extend(str(x) for x in vals)

        elif isinstance(ans, list):
            aliases = [str(x) for x in ans]
            value = aliases[0] if aliases else None

        elif ans is not None:
            value = str(ans)

        if value and str(value) not in aliases:
            aliases.insert(0, str(value))

        rows.append({
            "case_id": raw["case_id"],
            "task": "knowledge",
            "prompt": raw["question"],
            "expected": aliases,
            "references": aliases,
        })

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(
    json.dumps(rows, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Converted: {len(rows)} cases")
print(f"Saved: {DST}")

if rows:
    print("\nExample:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
