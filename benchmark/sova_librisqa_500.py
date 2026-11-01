from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset, Audio
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

N = 500

LLM_MODEL = "openai/gpt-oss-120b"
STT_MODEL = "whisper-large-v3-turbo"

INTER_CASE_DELAY = 1.0
MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_sova_librisqa_500"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

LIBRISQA_URL = (
    "https://huggingface.co/datasets/ZihanZhao/LibriSQA/"
    "resolve/main/LibriSQA-PartII/LibriSQA-PartII-test.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_groq() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT.parent / ".env")
            api_key = os.getenv("GROQ_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Your Terminal 2 works, so copy the same .env-loading setup."
        )

    return Groq(api_key=api_key)


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def normalize_label(text: str) -> str | None:
    text = text.strip().upper()

    for label in ("A", "B", "C", "D"):
        if text == label:
            return label

    # Look for a standalone A/B/C/D in the answer.
    import re

    matches = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", text)
    if matches:
        return matches[0]

    return None


def extract_audio_id(speech_path: str) -> str:
    # LibriSpeech/test-clean/5639/40744/5639-40744-0000.wav
    return Path(speech_path).stem


def get_audio_bytes(audio: Any) -> bytes:
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)

        raise TypeError(
            "Streaming LibriSpeech item did not provide encoded bytes."
        )

    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)

    raise TypeError(
        f"Unsupported audio type: {type(audio)}"
    )


# ============================================================
# LOAD LIBRISQA METADATA
# ============================================================

def load_librisqa() -> list[dict[str, Any]]:
    print("Downloading LibriSQA Part II test metadata...")

    response = requests.get(
        LIBRISQA_URL,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    print(f"LibriSQA Part II total records: {len(data)}")

    return data[:N]


# ============================================================
# LOAD LIBRISPEECH AUDIO FOR TARGET IDS
# ============================================================

def collect_target_audio(
    target_ids: set[str],
) -> dict[str, bytes]:

    print(
        f"Streaming LibriSpeech test.clean "
        f"until {len(target_ids)} required audio files are found..."
    )

    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )

    try:
        dataset = dataset.cast_column(
            "audio",
            Audio(decode=False),
        )
    except Exception:
        pass

    audio_map: dict[str, bytes] = {}

    for item in dataset:

        item_id = str(item["id"])

        if item_id not in target_ids:
            continue

        audio_map[item_id] = get_audio_bytes(item["audio"])

        print(
            f"Audio matched "
            f"{len(audio_map):04d}/{len(target_ids)}: "
            f"{item_id}"
        )

        if len(audio_map) == len(target_ids):
            break

    return audio_map


# ============================================================
# STT
# ============================================================

def transcribe(
    client: Groq,
    audio_bytes: bytes,
    filename: str,
) -> tuple[str, float]:

    started = time.perf_counter()

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.audio.transcriptions.create(
                file=(
                    filename,
                    audio_bytes,
                    "audio/wav",
                ),
                model=STT_MODEL,
                response_format="json",
                language="en",
                temperature=0.0,
            )

            text = (
                getattr(response, "text", "")
                or ""
            ).strip()

            latency = (
                time.perf_counter()
                - started
            ) * 1000.0

            return text, latency

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
                f"STT RATE LIMIT -> "
                f"waiting {wait_s:.0f}s"
            )

            time.sleep(wait_s)

    raise RuntimeError("STT failed")


# ============================================================
# SPOKEN QA LLM
# ============================================================

def answer_question(
    client: Groq,
    transcript: str,
    question_and_options: str,
) -> tuple[str, float]:

    prompt = f"""
You are being evaluated on a spoken question answering benchmark.

The speech transcript is:

{transcript}

The question and four answer options are:

{question_and_options}

Choose exactly ONE correct option.

Return ONLY one letter:
A
B
C
or
D
""".strip()

    started = time.perf_counter()

    for attempt in range(MAX_RETRIES + 1):

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
                max_tokens=16,
            )

            output = (
                response.choices[0].message.content
                or ""
            ).strip()

            latency = (
                time.perf_counter()
                - started
            ) * 1000.0

            return output, latency

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

            time.sleep(wait_s)

    raise RuntimeError("LLM failed")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    existing: list[dict[str, Any]] = []

    if RESULTS_PATH.exists():

        try:
            existing = json.loads(
                RESULTS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            print(
                f"RESUME: loaded "
                f"{len(existing)} previous results"
            )

        except Exception:
            existing = []

    completed = {
        str(x["case_id"])
        for x in existing
        if x.get("success")
    }

    results = list(existing)

    # --------------------------------------------------------
    # LibriSQA metadata
    # --------------------------------------------------------

    records = load_librisqa()

    targets = {
        extract_audio_id(
            row["speech_path"]
        )
        for row in records
    }

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    audio_map = collect_target_audio(targets)

    print(
        f"Matched audio: "
        f"{len(audio_map)}/{len(targets)}"
    )

    if not audio_map:
        raise RuntimeError(
            "No LibriSpeech audio matched LibriSQA speech_path IDs."
        )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    client = load_groq()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for idx, row in enumerate(
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
                f"SKIP {case_id}: audio not found"
            )
            continue

        try:

            # -----------------------------
            # STT
            # -----------------------------

            transcript, stt_latency = transcribe(
                client,
                audio_map[case_id],
                f"{case_id}.wav",
            )

            # -----------------------------
            # QA
            # -----------------------------

            model_output, llm_latency = answer_question(
                client,
                transcript,
                row["question_and_option"],
            )

            predicted = normalize_label(
                model_output
            )

            expected = str(
                row["answer"]
            ).strip().upper()

            correct = (
                predicted == expected
            )

            result = {
                "case_id": case_id,
                "task": "spoken_qa",
                "transcript": transcript,
                "question_and_option": row[
                    "question_and_option"
                ],
                "expected": expected,
                "model_output": model_output,
                "predicted": predicted,
                "correct": correct,
                "stt_latency_ms": stt_latency,
                "llm_latency_ms": llm_latency,
                "success": True,
                "speech_path": row[
                    "speech_path"
                ],
            }

            results.append(result)

            completed.add(case_id)

            good = sum(
                1
                for r in results
                if r.get("success")
                and r.get("correct")
            )

            successful = sum(
                1
                for r in results
                if r.get("success")
            )

            accuracy = (
                good / successful
                if successful
                else 0.0
            )

            print(
                f"{idx:03d}/{N} "
                f"{case_id:<20} "
                f"pred={predicted} "
                f"exp={expected} "
                f"correct={correct} "
                f"running_acc={accuracy*100:.2f}% "
                f"STT={stt_latency:.0f}ms "
                f"LLM={llm_latency:.0f}ms"
            )

        except Exception as exc:

            print(
                f"FAILED {case_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "case_id": case_id,
                    "task": "spoken_qa",
                    "success": False,
                    "correct": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

        # Save after every case
        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        successful_rows = [
            r
            for r in results
            if r.get("success")
        ]

        correct_rows = [
            r
            for r in successful_rows
            if r.get("correct")
        ]

        stt_latencies = [
            r["stt_latency_ms"]
            for r in successful_rows
            if r.get("stt_latency_ms")
            is not None
        ]

        llm_latencies = [
            r["llm_latency_ms"]
            for r in successful_rows
            if r.get("llm_latency_ms")
            is not None
        ]

        summary = {
            "dataset": "ZihanZhao/LibriSQA",
            "subset": "PartII-test",
            "target_samples": N,
            "successful_cases": len(
                successful_rows
            ),
            "failed_cases": sum(
                not r.get("success", False)
                for r in results
            ),
            "accuracy": (
                len(correct_rows)
                / len(successful_rows)
                if successful_rows
                else None
            ),
            "mean_stt_latency_ms": (
                sum(stt_latencies)
                / len(stt_latencies)
                if stt_latencies
                else None
            ),
            "mean_llm_latency_ms": (
                sum(llm_latencies)
                / len(llm_latencies)
                if llm_latencies
                else None
            ),
            "stt_model": STT_MODEL,
            "llm_model": LLM_MODEL,
        }

        SUMMARY_PATH.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        time.sleep(
            INTER_CASE_DELAY
        )

    print()
    print("=" * 50)
    print("LIBRISQA 500 COMPLETE")
    print("=" * 50)
    print(
        f"Results: {RESULTS_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()