"""Standalone SOVA-style speech-generation evaluator.

Avoids importing app.services.speech so it cannot trigger the project's
ElevenLabs <-> speech-service circular import.

Pipeline:
text response -> ElevenLabs TTS -> Groq Whisper -> WER
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from elevenlabs import AsyncElevenLabs
from groq import AsyncGroq


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def wer_percent(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    n = len(ref)
    if n == 0:
        return 0.0 if not hyp else 100.0

    # Levenshtein edit distance with operation costs 1.
    prev = list(range(len(hyp) + 1))
    for i, rword in enumerate(ref, 1):
        cur = [i]
        for j, hword in enumerate(hyp, 1):
            sub = prev[j - 1] + (rword != hword)
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            cur.append(min(sub, delete, insert))
        prev = cur
    return (prev[-1] / n) * 100.0


@dataclass
class AudioResult:
    model_name: str
    provider: str
    model: str
    case_id: str
    reference_text: str
    generated_transcript: str = ""
    consistency_wer: float | None = None
    tts_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    audio_path: str | None = None
    success: bool = False
    error: str | None = None


async def synthesize(client: AsyncElevenLabs, text: str, voice_id: str, model_id: str) -> bytes:
    stream = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    chunks: list[bytes] = []
    async for chunk in stream:
        if isinstance(chunk, bytes):
            chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("ElevenLabs returned empty audio.")
    return data


async def transcribe(client: AsyncGroq, audio: bytes, language: str = "en") -> str:
    response = await client.audio.transcriptions.create(
        file=("speech_eval.mp3", audio, "audio/mpeg"),
        model=env("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        response_format="verbose_json",
        language=language,
    )
    text = getattr(response, "text", "") or ""
    return text.strip()


async def run(results_path: Path, output_dir: Path, case_limit: int | None, delay_s: float) -> None:
    load_dotenv_if_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    source: list[dict[str, Any]] = json.loads(results_path.read_text(encoding="utf-8"))
    candidates = [
        x for x in source
        if x.get("success") and str(x.get("case_id", "")).startswith("generation_") and str(x.get("output", "")).strip()
    ]
    candidates.sort(key=lambda x: x["case_id"])
    if case_limit is not None:
        candidates = candidates[:case_limit]

    progress_path = output_dir / "audio_results.json"
    existing: list[AudioResult] = []
    if progress_path.exists():
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
        existing = [AudioResult(**x) for x in raw]
    done = {x.case_id for x in existing if x.success}
    results = list(existing)

    eleven_key = env("ELEVENLABS_API_KEY")
    voice_id = env("ELEVENLABS_VOICE_ID")
    eleven_model = env("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    groq_key = env("GROQ_API_KEY")
    if not eleven_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured in the environment.")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not configured in the environment.")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not configured in the environment.")

    tts = AsyncElevenLabs(api_key=eleven_key)
    stt = AsyncGroq(api_key=groq_key)

    try:
        for item in candidates:
            case_id = str(item["case_id"])
            if case_id in done:
                continue
            ref = str(item["output"]).strip()
            result = AudioResult(
                model_name=str(item.get("model_name", "unknown")),
                provider=str(item.get("provider", "unknown")),
                model=str(item.get("model", "unknown")),
                case_id=case_id,
                reference_text=ref,
            )
            try:
                t0 = time.perf_counter()
                audio = await synthesize(tts, ref, voice_id, eleven_model)
                result.tts_latency_ms = (time.perf_counter() - t0) * 1000.0
                audio_path = audio_dir / f"{case_id}.mp3"
                audio_path.write_bytes(audio)
                result.audio_path = str(audio_path)

                t1 = time.perf_counter()
                hyp = await transcribe(stt, audio)
                result.stt_latency_ms = (time.perf_counter() - t1) * 1000.0
                result.generated_transcript = hyp
                result.consistency_wer = wer_percent(ref, hyp)
                result.success = True
                print(
                    f"{case_id:<18} WER={result.consistency_wer:6.2f}% "
                    f"TTS={result.tts_latency_ms:7.1f} ms STT={result.stt_latency_ms:7.1f} ms"
                )
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                print(f"{case_id:<18} FAILED {result.error}")

            results = [x for x in results if x.case_id != case_id]
            results.append(result)
            results.sort(key=lambda x: x.case_id)
            progress_path.write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
            await asyncio.sleep(max(0.0, delay_s))
    finally:
        close = getattr(tts, "close", None)
        if close is not None:
            maybe = close()
            if hasattr(maybe, "__await__"):
                await maybe
        close = getattr(stt, "close", None)
        if close is not None:
            maybe = close()
            if hasattr(maybe, "__await__"):
                await maybe

    successful = [x for x in results if x.success and x.consistency_wer is not None]
    summary: dict[str, Any] = {
        "cases_requested": len(candidates),
        "successful_audio_cases": len(successful),
        "failed_audio_cases": sum(not x.success for x in results),
        "models": {},
    }
    groups: dict[tuple[str, str, str], list[AudioResult]] = {}
    for x in successful:
        groups.setdefault((x.model_name, x.provider, x.model), []).append(x)
    for key, xs in groups.items():
        summary["models"]["|".join(key)] = {
            "model_name": key[0],
            "provider": key[1],
            "model": key[2],
            "cases": len(xs),
            "mean_consistency_wer": sum(x.consistency_wer for x in xs if x.consistency_wer is not None) / len(xs),
            "mean_tts_latency_ms": sum(x.tts_latency_ms for x in xs) / len(xs),
            "mean_stt_latency_ms": sum(x.stt_latency_ms for x in xs) / len(xs),
            "mean_total_audio_latency_ms": sum(x.tts_latency_ms + x.stt_latency_ms for x in xs) / len(xs),
        }
    (output_dir / "audio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.case_limit, args.delay))


if __name__ == "__main__":
    main()
