"""Turn benchmark JSON results into paper-ready CSV tables and figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(summary: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out_dir / "FIGURES_README.txt").write_text(
            "Install matplotlib, then rerun benchmark.report to create PNG figures.\n",
            encoding="utf-8",
        )
        return

    names = [row["model_name"] for row in summary]
    knowledge = [row["knowledge_accuracy"] for row in summary]
    latency = [row["mean_latency_ms"] for row in summary]
    wer = [row["consistency_wer"] or 0 for row in summary]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(names, knowledge)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Knowledge Accuracy")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "knowledge_accuracy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(names, latency)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Mean End-to-End Runtime Latency")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=200)
    plt.close(fig)

    if any(row.get("consistency_wer") is not None for row in summary):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(names, wer)
        ax.set_ylabel("WER (%)")
        ax.set_title("Speech-Text Consistency WER")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "consistency_wer.png", dpi=200)
        plt.close(fig)

    # Figure-3-style Answer Rate vs Conditioned Accuracy for paralinguistic tasks.
    emo = [row.get("emotion_conditioned_accuracy", 0) for row in summary]
    emo_ar = [row.get("emotion_answer_rate", 0) for row in summary]
    inst = [row.get("instruction_conditioned_accuracy", 0) for row in summary]
    inst_ar = [row.get("instruction_answer_rate", 0) for row in summary]
    if names:
        x = range(len(names))
        width = 0.36
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar([i-width/2 for i in x], emo, width=width, label="Emotion CA")
        ax.bar([i+width/2 for i in x], inst, width=width, label="Instruction CA")
        ax.plot(list(x), emo_ar, marker="o", linewidth=2, label="Emotion Answer Rate")
        ax.plot(list(x), inst_ar, marker="o", linewidth=2, label="Instruction Answer Rate")
        ax.set_xticks(list(x), names, rotation=30, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Figure-3-Style Answer Rate and Conditioned Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "figure3_style_ar_ca.png", dpi=220)
        plt.close(fig)


def make_table4(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact paper-oriented results table."""
    columns = [
        ("Knowledge ↑", "knowledge_accuracy"),
        ("Emotion ↑", "emotion_accuracy"),
        ("Instruction ↑", "instruction_accuracy"),
        ("Consistency WER ↓", "consistency_wer"),
        ("Latency ms ↓", "mean_latency_ms"),
        ("Tokens ↓", "mean_total_tokens"),
    ]
    rows = []
    for item in summary:
        row = {"Model": item["model_name"], "Provider": item["provider"], "LLM": item["model"]}
        for label, key in columns:
            row[label] = item.get(key)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.input / "summary.json").read_text())
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "table4_weings.csv", make_table4(summary))
    make_figures(summary, args.output)
    print(f"Reports written to {args.output}")


if __name__ == "__main__":
    main()
