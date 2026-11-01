"""Evaluate the local text emotion detector on the labeled pilot subset."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

from app.core.enums import Workflow
from app.engines.emotion.engine import KeywordEmotionDetector
from app.runtime.state import RuntimeState


ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"anxious": "fearful"}


async def evaluate(dataset_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [row for row in records if row.get("task") == "emotion"]
    detector = KeywordEmotionDetector()
    results: list[dict[str, object]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for case in cases:
        state = RuntimeState(
            request_id=uuid4(),
            workflow=Workflow.CHAT,
            input=str(case["prompt"]),
        )
        detected = await detector.detect_emotion(state)
        predicted = str(detected["emotion"])
        expected = str(case["expected_label"])
        normalized = LABEL_MAP.get(predicted, predicted)
        correct = normalized == expected
        confusion[expected][normalized] += 1
        results.append(
            {
                "case_id": str(case["id"]),
                "task": "emotion_recognition",
                "model_name": "Emotional Voice Assistant text emotion module",
                "model": "keyword_text_proxy",
                "success": True,
                "answered": True,
                "prompt": str(case["prompt"]),
                "expected_label": expected,
                "predicted_label": normalized,
                "raw_predicted_label": predicted,
                "correct": correct,
                "confidence": float(detected.get("confidence", 0.0)),
                "matched_keyword": detected.get("matched_keyword"),
                "metric_method": "exact label accuracy; anxious is mapped to the five-class fearful label",
            }
        )

    correct_count = sum(bool(row["correct"]) for row in results)
    total = len(results)
    summary: dict[str, object] = {
        "benchmark": "Emotion Aware Voice Assistant SOVA pilot text emotion subset",
        "dataset": str(dataset_path),
        "modality": "text only; not IEMOCAP or acoustic SER",
        "metric": "exact label accuracy",
        "samples": total,
        "answered": total,
        "correct": correct_count,
        "accuracy_percent": round(100 * correct_count / total, 2) if total else None,
        "answer_rate_percent": 100.0 if total else None,
        "conditioned_accuracy_percent": round(100 * correct_count / total, 2) if total else None,
        "label_mapping": LABEL_MAP,
        "confusion_matrix": {label: dict(counts) for label, counts in sorted(confusion.items())},
        "strict_official_reproduction_claimed": False,
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "weings_sova_pilot_150.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark" / "runs" / "emotion_text_pilot")
    args = parser.parse_args()
    results, summary = asyncio.run(evaluate(args.dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ("samples", "correct", "accuracy_percent", "answer_rate_percent")}))


if __name__ == "__main__":
    main()
