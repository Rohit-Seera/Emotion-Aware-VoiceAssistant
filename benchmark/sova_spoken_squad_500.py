from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset, Audio
from groq import Groq


N = 500

STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-120b"

DELAY = 1.0
MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_sova_spoken_squad_500"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def load_groq() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        from dotenv import load_dotenv
        load_dotenv(ROOT.parent / ".env")
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found.")

    return Groq(api_key=api_key)


def get_audio_bytes(audio: Any) -> bytes:
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)

    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)

    raise TypeError(
        f"Unsupported audio type: {type(audio)}"
    )


def transcribe(
    client: Groq,
    audio_bytes: bytes,
    filename: str,
) -> tuple[str, float]:

    start = time.perf_counter()

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
                time.perf_counter() - start
            ) * 1000

            return text, latency

        except Exception as exc:

            if (
                not is_rate_limit(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            wait = min(
                60,
                10 * (2 ** attempt),
            )

            print(
                f"STT RATE LIMIT -> waiting {wait}s"
            )

            time.sleep(wait)

    raise RuntimeError("STT failed")


def answer_question(
    client: Groq,
    transcript: str,
    question: str,
) -> tuple[str, float]:

    prompt = f"""
You are answering a spoken question-answering benchmark.

Speech transcript:
{transcript}

Question:
{question}

Answer the question using only information supported by the transcript.

Return ONLY the answer.
Do not explain your reasoning.
""".strip()

    start = time.perf_counter()

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
                max_tokens=64,
            )

            output = (
                response.choices[0]
                .message.content
                or ""
            ).strip()

            latency = (
                time.perf_counter() - start
            ) * 1000

            return output, latency

        except Exception as exc:

            if (
                not is_rate_limit(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            wait = min(
                60,
                10 * (2 ** attempt),
            )

            print(
                f"LLM RATE LIMIT -> waiting {wait}s"
            )

            time.sleep(wait)

    raise RuntimeError("LLM failed")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_match(
    prediction: str,
    reference: str,
) -> bool:

    p = normalize(prediction)
    r = normalize(reference)

    if not p or not r:
        return False

    if p == r:
        return True

    # Accept a concise answer contained in a longer reference,
    # or the reference contained in the model answer.
    if p in r or r in p:
        return True

    return False


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = []

    if RESULTS_PATH.exists():
        try:
            existing = json.loads(
                RESULTS_PATH.read_text(
                    encoding="utf-8"
                )
            )
            print(
                f"RESUME: loaded {len(existing)} results"
            )
        except Exception:
            existing = []

    done = {
        str(x["id"])
        for x in existing
        if x.get("success")
    }

    results = [
        x for x in existing
        if x.get("success")
    ]

    print(
        "Loading Spoken SQuAD in streaming mode..."
    )

    dataset = load_dataset(
        "AudioLLMs/spoken_squad_test",
        split="test",
        streaming=True,
    )

    try:
        dataset = dataset.cast_column(
            "context",
            Audio(decode=False),
        )
    except Exception:
        pass

    client = load_groq()

    for idx, item in enumerate(
        dataset,
        start=1,
    ):

        if idx > N:
            break

        case_id = str(idx)

        if case_id in done:
            continue

        try:

            audio_bytes = get_audio_bytes(
                item["context"]
            )

            transcript, stt_latency = transcribe(
                client,
                audio_bytes,
                f"spoken_squad_{idx}.wav",
            )

            prediction, llm_latency = answer_question(
                client,
                transcript,
                str(item["instruction"]),
            )

            reference = str(
                item["answer"]
            ).strip()

            correct = answer_match(
                prediction,
                reference,
            )

            result = {
                "id": case_id,
                "task": "spoken_squad",
                "transcript": transcript,
                "question": str(
                    item["instruction"]
                ),
                "reference": reference,
                "prediction": prediction,
                "correct": correct,
                "success": True,
                "stt_latency_ms": stt_latency,
                "llm_latency_ms": llm_latency,
            }

            results.append(result)
            done.add(case_id)

            good = sum(
                r["correct"]
                for r in results
            )

            accuracy = (
                good / len(results)
                if results
                else 0.0
            )

            print(
                f"{idx:03d}/{N} "
                f"correct={correct} "
                f"running_acc={accuracy*100:.2f}% "
                f"STT={stt_latency:.0f}ms "
                f"LLM={llm_latency:.0f}ms"
            )

        except Exception as exc:

            print(
                f"FAILED {idx}: "
                f"{type(exc).__name__}: {exc}"
            )

        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        successful = [
            r for r in results
            if r.get("success")
        ]

        correct = [
            r for r in successful
            if r.get("correct")
        ]

        summary = {
            "dataset": "AudioLLMs/spoken_squad_test",
            "split": "test",
            "target_samples": N,
            "successful_cases": len(successful),
            "failed_cases": N - len(successful),
            "accuracy": (
                len(correct) / len(successful)
                if successful
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

        time.sleep(DELAY)

    print()
    print("=" * 50)
    print("SPOKEN SQUAD 500 COMPLETE")
    print("=" * 50)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()