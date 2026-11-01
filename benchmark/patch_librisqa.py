from pathlib import Path
import re

# ============================================================
# TARGET FILE
# ============================================================

TARGET = Path(
    r"D:\voice companion\github\testing\benchmark\sova_librisqa_500.py"
)

text = TARGET.read_text(encoding="utf-8")


# ============================================================
# 1. IMPORT + BATCH SIZE
# ============================================================

if "import re" not in text:
    text = text.replace(
        "import os\n",
        "import os\nimport re\n",
        1,
    )

if "LLM_BATCH_SIZE = 5" not in text:
    text = text.replace(
        "MAX_RETRIES = 5\n",
        "MAX_RETRIES = 5\nLLM_BATCH_SIZE = 5\n",
        1,
    )


# ============================================================
# 2. REPLACE normalize_label()
# ============================================================

start = text.index("def normalize_label(")
end = text.index(
    "\ndef extract_audio_id(",
    start,
)

new_normalize = r'''
def normalize_label(text: str) -> str | None:
    text = (text or "").strip().upper()

    if text in {"A", "B", "C", "D"}:
        return text

    patterns = [
        r"^\s*(?:ANSWER|OPTION|CHOICE)?\s*[:\-]?\s*([ABCD])\s*$",
        r"(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return None


def parse_batch_answers(
    text: str,
    batch_size: int,
) -> dict[int, str]:

    text = (text or "").upper()

    answers: dict[int, str] = {}

    patterns = [
        r"(?:ITEM\s*)?(\d+)\s*[:.)\-]\s*([ABCD])\b",
        r"(?:ANSWER|OPTION|CHOICE)\s*(\d+)\s*[:.)\-]\s*([ABCD])\b",
    ]

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
        ):

            idx = int(
                match.group(1)
            )

            if 1 <= idx <= batch_size:

                answers[idx] = match.group(2)

    # Fallback:
    # one answer per line:
    #
    # 1:C
    # 2:A
    # C
    # D

    if len(answers) < batch_size:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for idx, line in enumerate(
            lines,
            start=1,
        ):

            if idx > batch_size:
                break

            if idx in answers:
                continue

            label = normalize_label(
                line
            )

            if label:
                answers[idx] = label

    return answers

'''

text = (
    text[:start]
    + new_normalize
    + text[end:]
)


# ============================================================
# 3. REPLACE SINGLE QUESTION LLM WITH BATCH LLM
# ============================================================

start = text.index(
    "def answer_question("
)

end = text.index(
    "\n\n# ============================================================\n# MAIN",
    start,
)

new_llm = r'''
def answer_questions_batch(
    client: Groq,
    items: list[dict[str, Any]],
) -> tuple[dict[int, str], float]:

    blocks = []

    for idx, item in enumerate(
        items,
        start=1,
    ):

        blocks.append(
            f"""ITEM {idx}

Transcript:
{item["transcript"]}

Question and options:
{item["question_and_options"]}
"""
        )

    prompt = """
You are answering a spoken question-answering benchmark.

For every item, choose exactly ONE answer.

Return ONLY one line for each item in this format:

1:A
2:C
3:B

Do not explain.
Do not repeat the questions.
Do not write anything except the numbered answers.

""" + "\n".join(blocks)

    started = time.perf_counter()

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=max(
                    8,
                    len(items) * 4,
                ),
            )

            output = (
                response.choices[0].message.content
                or ""
            ).strip()

            predictions = parse_batch_answers(
                output,
                len(items),
            )

            latency = (
                time.perf_counter()
                - started
            ) * 1000.0

            return (
                predictions,
                latency,
            )

        except Exception as exc:

            if (
                not is_rate_limit(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            wait_s = min(
                60.0,
                10.0 * (2 ** attempt),
            )

            print(
                f"LLM RATE LIMIT -> "
                f"waiting {wait_s:.0f}s"
            )

            time.sleep(
                wait_s
            )

    raise RuntimeError(
        "Batch LLM request failed"
    )


def process_batch(
    client: Groq,
    pending: list[dict[str, Any]],
    results: list[dict[str, Any]],
    completed: set[str],
) -> None:

    predictions, batch_latency = (
        answer_questions_batch(
            client,
            pending,
        )
    )

    per_item_latency = (
        batch_latency
        / len(pending)
    )

    for idx, item in enumerate(
        pending,
        start=1,
    ):

        predicted = predictions.get(
            idx
        )

        expected = item[
            "expected"
        ]

        correct = (
            predicted == expected
        )

        success = predicted in {
            "A",
            "B",
            "C",
            "D",
        }

        result = {
            "case_id": item[
                "case_id"
            ],
            "task": "spoken_qa",
            "transcript": item[
                "transcript"
            ],
            "question_and_option": item[
                "question_and_options"
            ],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "success": success,
            "stt_latency_ms": item[
                "stt_latency_ms"
            ],
            "llm_batch_latency_ms":
                batch_latency,
            "llm_amortized_latency_ms":
                per_item_latency,
            "speech_path": item[
                "speech_path"
            ],
        }

        results.append(
            result
        )

        completed.add(
            item["case_id"]
        )

        print(
            f"{item['position']:03d}/{N} "
            f"{item['case_id']:<20} "
            f"pred={predicted} "
            f"exp={expected} "
            f"correct={correct} "
            f"STT={item['stt_latency_ms']:.0f}ms "
            f"LLM/share={per_item_latency:.0f}ms"
        )


def save_summary(
    results: list[dict[str, Any]]
) -> None:

    successful = [
        r
        for r in results
        if r.get("success")
        and r.get("predicted")
        in {"A", "B", "C", "D"}
    ]

    correct = [
        r
        for r in successful
        if r.get("correct")
    ]

    stt_latencies = [
        r["stt_latency_ms"]
        for r in successful
        if r.get(
            "stt_latency_ms"
        ) is not None
    ]

    llm_latencies = [
        r["llm_amortized_latency_ms"]
        for r in successful
        if r.get(
            "llm_amortized_latency_ms"
        ) is not None
    ]

    summary = {
        "dataset": "ZihanZhao/LibriSQA",
        "subset": "PartII-test",
        "target_samples": N,
        "successful_cases": len(
            successful
        ),
        "failed_cases": (
            N - len(successful)
        ),
        "accuracy": (
            len(correct)
            / len(successful)
            if successful
            else None
        ),
        "mean_stt_latency_ms": (
            sum(stt_latencies)
            / len(stt_latencies)
            if stt_latencies
            else None
        ),
        "mean_llm_amortized_latency_ms": (
            sum(llm_latencies)
            / len(llm_latencies)
            if llm_latencies
            else None
        ),
        "stt_model": STT_MODEL,
        "llm_model": LLM_MODEL,
        "llm_batch_size": LLM_BATCH_SIZE,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

'''

text = (
    text[:start]
    + new_llm
    + text[end:]
)


# ============================================================
# 4. REPLACE MAIN()
# ============================================================

start = text.index(
    "def main():"
)

end = text.index(
    '\n\nif __name__ == "__main__":',
    start,
)

new_main = r'''
def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load only valid completed results
    # --------------------------------------------------------

    existing: list[
        dict[str, Any]
    ] = []

    if RESULTS_PATH.exists():

        try:

            existing = json.loads(
                RESULTS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            print(
                f"RESUME: loaded "
                f"{len(existing)} saved results"
            )

        except Exception as exc:

            print(
                f"WARNING: couldn't load "
                f"previous results: {exc}"
            )

    completed = {
        str(row["case_id"])
        for row in existing
        if row.get("success")
        and row.get("predicted")
        in {"A", "B", "C", "D"}
    }

    # Ignore old malformed/failed rows
    results = [
        row
        for row in existing
        if (
            row.get("success")
            and row.get("predicted")
            in {"A", "B", "C", "D"}
        )
    ]

    # --------------------------------------------------------
    # Load LibriSQA
    # --------------------------------------------------------

    records = load_librisqa()

    targets = {
        extract_audio_id(
            row["speech_path"]
        )
        for row in records
    }

    # --------------------------------------------------------
    # Collect audio
    # --------------------------------------------------------

    audio_map = collect_target_audio(
        targets
    )

    print(
        f"Matched audio: "
        f"{len(audio_map)}/{len(targets)}"
    )

    if not audio_map:

        raise RuntimeError(
            "No LibriSpeech audio matched "
            "LibriSQA speech_path IDs."
        )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    client = load_groq()

    # --------------------------------------------------------
    # Batch pipeline
    # --------------------------------------------------------

    pending: list[
        dict[str, Any]
    ] = []

    for position, row in enumerate(
        records,
        start=1,
    ):

        case_id = extract_audio_id(
            row["speech_path"]
        )

        if case_id in completed:
            continue

        if case_id not in audio_map:

            print(
                f"SKIP {case_id}: "
                f"audio not found"
            )

            continue

        try:

            # --------------------------------------------
            # STT
            # --------------------------------------------

            transcript, stt_latency = (
                transcribe(
                    client,
                    audio_map[
                        case_id
                    ],
                    f"{case_id}.wav",
                )
            )

            # --------------------------------------------
            # Add to batch
            # --------------------------------------------

            pending.append(
                {
                    "position": position,
                    "case_id": case_id,
                    "transcript": transcript,
                    "question_and_options":
                        row[
                            "question_and_option"
                        ],
                    "expected": str(
                        row["answer"]
                    ).strip().upper(),
                    "stt_latency_ms":
                        stt_latency,
                    "speech_path":
                        row[
                            "speech_path"
                        ],
                }
            )

            # --------------------------------------------
            # Run LLM every 5 items
            # --------------------------------------------

            if len(pending) >= LLM_BATCH_SIZE:

                process_batch(
                    client,
                    pending,
                    results,
                    completed,
                )

                RESULTS_PATH.write_text(
                    json.dumps(
                        results,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                save_summary(
                    results
                )

                pending.clear()

        except Exception as exc:

            print(
                f"STT FAILED {case_id}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        time.sleep(
            INTER_CASE_DELAY
        )

    # --------------------------------------------------------
    # Leftover batch
    # --------------------------------------------------------

    if pending:

        process_batch(
            client,
            pending,
            results,
            completed,
        )

        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        save_summary(
            results
        )

    print()

    print(
        "=" * 50
    )

    print(
        "LIBRISQA 500 COMPLETE"
    )

    print(
        "=" * 50
    )

    print(
        f"Results: {RESULTS_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

'''

text = (
    text[:start]
    + new_main
    + text[end:]
)


# ============================================================
# SAVE
# ============================================================

TARGET.write_text(
    text,
    encoding="utf-8",
)

print(
    "PATCHED SUCCESSFULLY:"
)

print(
    TARGET
)

print(
    "LLM calls reduced from ~500 to ~100."
)

print(
    "Batch A/B/C/D parser added."
)

print(
    "Old malformed results will not be treated as completed."
)