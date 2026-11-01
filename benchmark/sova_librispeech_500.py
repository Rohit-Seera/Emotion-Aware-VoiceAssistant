from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from datasets import load_dataset, Audio
from groq import Groq

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")
# ============================================================
# CONFIG
# ============================================================

N = 500
MODEL = "whisper-large-v3-turbo"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_sova_librispeech_500"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

INTER_CASE_DELAY = 1.0
MAX_RETRIES = 5


# ============================================================
# WER
# ============================================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Standard word error rate:
        WER = (S + D + I) / N
    """

    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n = len(ref_words)

    # DP matrix
    dp = [[0] * (len(hyp_words) + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, len(hyp_words) + 1):

            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1

                dp[i][j] = min(
                    substitution,
                    deletion,
                    insertion,
                )

    return dp[n][len(hyp_words)] / n


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client() -> Groq:

    api_key = os.getenv("GROQ_API_KEY")

    # Try project settings as fallback
    if not api_key:
        try:
            from app.core.settings import settings

            groq_settings = getattr(settings.ai, "groq", None)

            if groq_settings is not None:
                api_key = getattr(groq_settings, "api_key", None)

        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. "
            "Set it in your environment or project .env."
        )

    return Groq(api_key=api_key)


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def get_audio_bytes(audio: Any) -> bytes:

    if isinstance(audio, dict):

        raw_bytes = audio.get("bytes")

        if isinstance(raw_bytes, (bytes, bytearray)):
            return bytes(raw_bytes)

        # Fallback if HF returns decoded array
        array = audio.get("array")
        sampling_rate = int(
            audio.get("sampling_rate") or 16000
        )

        if array is not None:

            import numpy as np
            import soundfile as sf

            buffer = io.BytesIO()

            sf.write(
                buffer,
                np.asarray(array),
                sampling_rate,
                format="WAV",
            )

            return buffer.getvalue()

    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)

    raise TypeError(
        f"Unsupported audio format: {type(audio)}"
    )


# ============================================================
# RATE LIMIT
# ============================================================

def is_rate_limit_error(exc: Exception) -> bool:

    text = str(exc).lower()

    return (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_audio(
    client: Groq,
    audio_bytes: bytes,
    filename: str,
):

    started = time.perf_counter()

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.audio.transcriptions.create(
                file=(
                    filename,
                    audio_bytes,
                    "audio/flac",
                ),
                model=MODEL,
                response_format="json",
                language="en",
                temperature=0.0,
            )

            transcript = (
                getattr(response, "text", "")
                or ""
            ).strip()

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            return transcript, latency_ms

        except Exception as exc:

            if (
                not is_rate_limit_error(exc)
                or attempt >= MAX_RETRIES
            ):
                raise

            wait_seconds = min(
                60.0,
                10.0 * (2 ** attempt),
            )

            print(
                f"RATE LIMIT -> waiting "
                f"{wait_seconds:.0f}s "
                f"(retry {attempt + 1}/{MAX_RETRIES})"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected transcription failure")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    existing = []

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
                "WARNING: couldn't load previous results:",
                exc,
            )

            existing = []

    completed_ids = {
        str(row["id"])
        for row in existing
        if row.get("success")
    }

    results = list(existing)

    # --------------------------------------------------------
    # HF dataset
    # --------------------------------------------------------

    print("Loading LibriSpeech test.clean in streaming mode...")

    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    )

    # Avoid decoding all audio locally.
    try:
        dataset = dataset.cast_column(
            "audio",
            Audio(decode=False),
        )
    except Exception as exc:

        print(
            "WARNING: Audio(decode=False) unavailable:",
            exc,
        )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    client = get_groq_client()

    # --------------------------------------------------------
    # Process exactly first N samples
    # --------------------------------------------------------

    seen = 0

    for item in dataset:

        if seen >= N:
            break

        seen += 1

        case_id = str(item["id"])

        if case_id in completed_ids:
            continue

        reference = str(
            item["text"]
        ).strip()

        try:

            audio_bytes = get_audio_bytes(
                item["audio"]
            )

            hypothesis, latency_ms = transcribe_audio(
                client=client,
                audio_bytes=audio_bytes,
                filename=f"{case_id}.flac",
            )

            sample_wer = calculate_wer(
                reference,
                hypothesis,
            )

            row = {
                "id": case_id,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": sample_wer,
                "latency_ms": latency_ms,
                "success": True,
                "speaker_id": item.get("speaker_id"),
                "chapter_id": item.get("chapter_id"),
            }

            results.append(row)

            completed_ids.add(case_id)

            successful = [
                r["wer"]
                for r in results
                if r.get("success")
                and r.get("wer") is not None
            ]

            current_mean_wer = (
                sum(successful) / len(successful)
                if successful
                else 0.0
            )

            print(
                f"{len(successful):03d}/{N} "
                f"{case_id:<18} "
                f"WER={sample_wer:.4f} "
                f"MeanWER={current_mean_wer:.4f} "
                f"Latency={latency_ms:.1f}ms"
            )

        except Exception as exc:

            print(
                f"FAILED {case_id}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "id": case_id,
                    "reference": reference,
                    "hypothesis": "",
                    "wer": None,
                    "latency_ms": None,
                    "success": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "speaker_id": item.get(
                        "speaker_id"
                    ),
                    "chapter_id": item.get(
                        "chapter_id"
                    ),
                }
            )

        # ----------------------------------------------------
        # Save after EVERY sample
        # ----------------------------------------------------

        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        successful_rows = [
            r
            for r in results
            if r.get("success")
        ]

        wer_values = [
            r["wer"]
            for r in successful_rows
            if r.get("wer") is not None
        ]

        latency_values = [
            r["latency_ms"]
            for r in successful_rows
            if r.get("latency_ms") is not None
        ]

        summary = {

            "dataset": "openslr/librispeech_asr",

            "config": "clean",

            "split": "test",

            "target_samples": N,

            "successful_cases": len(
                successful_rows
            ),

            "failed_cases": sum(
                not r.get("success", False)
                for r in results
            ),

            "mean_wer": (
                sum(wer_values)
                / len(wer_values)
                if wer_values
                else None
            ),

            "mean_latency_ms": (
                sum(latency_values)
                / len(latency_values)
                if latency_values
                else None
            ),

            "stt_model": MODEL,
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

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n===================================")
    print("LIBRISPEECH 500 COMPLETE")
    print("===================================")

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()