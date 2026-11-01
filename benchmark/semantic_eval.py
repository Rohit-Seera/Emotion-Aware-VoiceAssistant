from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


SYSTEM = """You are a strict evaluation judge.

Score the assistant response for four criteria from 1 to 5:
C = correctness
R = relevance
H = coherence
I = instruction following

Return ONLY one line in exactly this format:
C=5 R=4 H=5 I=4

Do not explain.
Do not add any other text.
"""


# More tolerant than the previous parser:
# - allows commas/newlines between scores
# - allows optional spaces
# - still requires all four scores
PAT = re.compile(
    r"C\s*=\s*([1-5])"
    r".{0,80}?"
    r"R\s*=\s*([1-5])"
    r".{0,80}?"
    r"H\s*=\s*([1-5])"
    r".{0,80}?"
    r"I\s*=\s*([1-5])",
    re.I | re.DOTALL,
)


def extract_scores(text: str):
    if not text or not text.strip():
        raise ValueError("Judge returned empty content")

    cleaned = text.strip()

    m = PAT.search(cleaned)

    if not m:
        raise ValueError(
            f"Could not parse judge output: {cleaned[:500]!r}"
        )

    c, r, h, i = map(int, m.groups())

    mean5 = (c + r + h + i) / 4
    pct = (c + r + h + i) / 20 * 100

    return c, r, h, i, mean5, pct, cleaned


def call_judge(
    client,
    model,
    prompt,
    output,
    max_tokens=96,
):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        reasoning_effort="low",
        messages=[
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    f"USER REQUEST:\n{prompt}\n\n"
                    f"ASSISTANT RESPONSE:\n{output}\n\n"
                    "Score it now."
                ),
            },
        ],
    )

    msg = resp.choices[0].message

    content = getattr(msg, "content", None)

    if not isinstance(content, str):
        content = ""

    # Some SDK/model combinations may expose output elsewhere.
    if not content.strip():
        details = getattr(msg, "content_details", None)

        if isinstance(details, str):
            content = details

    usage = getattr(resp, "usage", None)

    total_tokens = getattr(
        usage,
        "total_tokens",
        0,
    ) or 0

    return content, total_tokens


def judge_with_retry(
    client,
    model,
    prompt,
    output,
    attempts=3,
    retry_delay=2.0,
):
    """
    Retry if the judge returns:
    - empty output
    - truncated output such as 'C='
    - malformed score format
    """

    last_error = None
    last_raw = ""

    for attempt in range(1, attempts + 1):
        try:
            raw, tokens = call_judge(
                client=client,
                model=model,
                prompt=prompt,
                output=output,
                max_tokens=96,
            )

            last_raw = raw

            scores = extract_scores(raw)

            return {
                "success": True,
                "scores": scores,
                "tokens": tokens,
                "raw": raw,
                "attempts": attempt,
            }

        except Exception as e:
            last_error = str(e)

            print(
                f"    judge attempt {attempt}/{attempts} failed: "
                f"{last_error}"
            )

            if attempt < attempts:
                time.sleep(retry_delay)

    return {
        "success": False,
        "scores": None,
        "tokens": 0,
        "raw": last_raw,
        "attempts": attempts,
        "error": last_error,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--model",
        default=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
    )

    ap.add_argument(
        "--delay",
        type=float,
        default=3.0,
    )

    ap.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    args = ap.parse_args()

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise SystemExit(
            "GROQ_API_KEY missing"
        )

    inp = Path(args.input)
    outdir = Path(args.output)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = json.loads(
        inp.read_text(
            encoding="utf-8"
        )
    )

    results_path = (
        outdir / "semantic_results.json"
    )

    summary_path = (
        outdir / "semantic_summary.json"
    )

    existing = {}

    if results_path.exists():
        try:
            old = json.loads(
                results_path.read_text(
                    encoding="utf-8"
                )
            )

            existing = {
                x["case_id"]: x
                for x in old
                if x.get("case_id")
            }

        except Exception:
            existing = {}

    client = Groq(
        api_key=key
    )

    for row in rows:
        cid = row["case_id"]

        # Successful cases stay cached.
        # Failed cases are automatically retried.
        if (
            existing.get(cid, {}).get(
                "success"
            )
            is True
        ):
            print(
                f"{cid:<20} cached"
            )
            continue

        print(
            f"\n{cid:<20} judging..."
        )

        t0 = time.perf_counter()

        result = judge_with_retry(
            client=client,
            model=args.model,
            prompt=row.get(
                "prompt",
                "",
            ),
            output=row.get(
                "output",
                "",
            ),
            attempts=max(
                1,
                args.retries,
            ),
        )

        elapsed = (
            time.perf_counter()
            - t0
        ) * 1000

        if result["success"]:
            (
                c,
                r,
                h,
                i,
                mean5,
                pct,
                raw,
            ) = result["scores"]

            item = {
                "case_id": cid,
                "model_name": row.get(
                    "model_name"
                ),
                "provider": row.get(
                    "provider"
                ),
                "model": row.get(
                    "model"
                ),
                "success": True,
                "correctness": c,
                "relevance": r,
                "coherence": h,
                "instruction_following": i,
                "mean_score_5": mean5,
                "semantic_score_pct": pct,
                "judge_latency_ms": elapsed,
                "judge_tokens": result[
                    "tokens"
                ],
                "judge_attempts": result[
                    "attempts"
                ],
                "raw_judge_output": raw,
                "error": None,
            }

            print(
                f"{cid:<20} "
                f"score={pct:6.2f}% "
                f"attempts={result['attempts']}"
            )

        else:
            item = {
                "case_id": cid,
                "model_name": row.get(
                    "model_name"
                ),
                "provider": row.get(
                    "provider"
                ),
                "model": row.get(
                    "model"
                ),
                "success": False,
                "correctness": None,
                "relevance": None,
                "coherence": None,
                "instruction_following": None,
                "mean_score_5": None,
                "semantic_score_pct": None,
                "judge_latency_ms": elapsed,
                "judge_tokens": 0,
                "judge_attempts": result[
                    "attempts"
                ],
                "raw_judge_output": result[
                    "raw"
                ],
                "error": result[
                    "error"
                ],
            }

            print(
                f"{cid:<20} "
                f"ERROR {result['error']}"
            )

        existing[cid] = item

        # Save after every case.
        results_path.write_text(
            json.dumps(
                list(
                    existing.values()
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        time.sleep(
            max(
                args.delay,
                0,
            )
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    good = [
        x
        for x in existing.values()
        if x.get("success")
    ]

    failed = [
        x
        for x in existing.values()
        if not x.get("success")
    ]

    summary = {
        "model_name": (
            good[0]["model_name"]
            if good
            else None
        ),
        "provider": (
            good[0]["provider"]
            if good
            else None
        ),
        "model": (
            good[0]["model"]
            if good
            else None
        ),
        "cases": len(good),
        "failed_cases": len(failed),
        "mean_semantic_score_5": (
            sum(
                x["mean_score_5"]
                for x in good
            )
            / len(good)
            if good
            else None
        ),
        "mean_semantic_score_pct": (
            sum(
                x["semantic_score_pct"]
                for x in good
            )
            / len(good)
            if good
            else None
        ),
        "rubric": {
            "correctness": "1-5",
            "relevance": "1-5",
            "coherence": "1-5",
            "instruction_following": "1-5",
            "normalized_percent": (
                "(sum/20)*100"
            ),
        },
        "judge_model": args.model,
        "failed_case_ids": [
            x["case_id"]
            for x in failed
        ],
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n" + json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

