"""Adapters for the SOVA-Bench baselines and the Emotion Aware Voice Assistant audio pipeline.

The module contains native connectors where the upstream model exposes a stable
batch/API interface. Models with streaming-specific official runtimes use a
JSONL bridge contract so their own inference code can remain unmodified.
"""
from __future__ import annotations

import base64
import email
import json
import os
import re
import shlex
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from email import policy
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = Path(__file__).with_name("sova_model_registry.json")


@dataclass(slots=True)
class CaseOutput:
    case_id: str
    model_id: str
    model_name: str
    task: str
    success: bool
    prediction: str = ""
    model_text_output: str = ""
    transcript: str = ""
    generated_transcript: str = ""
    response_audio_path: str = ""
    input_transcript: str = ""
    user_emotion: str = ""
    emotion_confidence: float | None = None
    emotion_source: str = ""
    latency_ms: float = 0.0
    error: str = ""
    metric_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self, source: dict[str, Any]) -> dict[str, Any]:
        rec = dict(source)
        rec.update(asdict(self))
        # Keep scorer-compatible canonical aliases.
        rec["model"] = self.model_id
        rec["model_name"] = self.model_name
        rec["case_id"] = self.case_id
        rec["output"] = self.prediction
        rec["response_text"] = self.prediction
        rec["model_text_output"] = self.model_text_output
        rec["generated_audio_path"] = self.response_audio_path
        rec["recognized_text"] = self.transcript or self.input_transcript
        rec["success"] = bool(self.success)
        rec["answered"] = bool(self.prediction or self.transcript or self.generated_transcript)
        rec["error"] = self.error or None
        if not rec.get("predicted_label") and self.prediction:
            labels = rec.get("labels")
            if isinstance(labels, list):
                text = self.prediction.strip().casefold()
                for label in labels:
                    candidate = str(label).strip()
                    if text == candidate.casefold() or re.search(r"(?<!\w)" + re.escape(candidate.casefold()) + r"(?!\w)", text):
                        rec["predicted_label"] = candidate
                        break
        if self.user_emotion and not rec.get("weings_predicted_emotion"):
            rec["weings_predicted_emotion"] = self.user_emotion
        if self.emotion_confidence is not None:
            rec["weings_emotion_confidence"] = self.emotion_confidence
        if self.emotion_source:
            rec["weings_emotion_source"] = self.emotion_source
        return rec


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _case_id(item: dict[str, Any], index: int) -> str:
    return str(item.get("case_id") or item.get("id") or item.get("question_id") or f"case_{index:06d}")


def _audio_path(item: dict[str, Any]) -> Path | None:
    raw = item.get("audio_path") or item.get("audio_file") or item.get("speech_path") or item.get("speech")
    if not raw or not isinstance(raw, str):
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    return candidate if candidate.is_file() else None


def _prompt(item: dict[str, Any]) -> str:
    """Read an explicit user-provided instruction; never use this for benchmark gold text."""
    for key in ("prompt", "question", "instruction", "query", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "Please respond naturally to the user's spoken message."


def _task_instruction(item: dict[str, Any]) -> str:
    """Build a task instruction without copying any clean/reference input text.

    A manifest may supply `text_modality_input` when the evaluation protocol
    intentionally gives text plus audio (e.g. LLaMA-Omni's supported mode).
    Other dataset fields like question/reference_text/answers are not model input.
    """
    side_text = item.get("text_modality_input")
    if isinstance(side_text, str) and side_text.strip():
        return side_text.strip()
    task = canonical_task(item.get("task", "unknown"))
    instructions = {
        "speech_recognition": "Transcribe the speech in the audio verbatim. Return only the transcript.",
        "knowledge_qa": "Answer the question spoken in the audio accurately and concisely.",
        "spoken_qa_multiple_choice": "Answer the multiple-choice question spoken in the audio. State the selected option clearly.",
        "spoken_qa_open_ended": "Answer the open-ended question spoken in the audio, using only the spoken content as the question.",
        "emotion_recognition": "Identify the emotion conveyed by the speaker's voice. Return the requested emotion label only.",
        "age_recognition": "Predict the speaker's age category from the voice. Return the requested category label only.",
        "gender_recognition": "Predict the speaker's gender label from the voice. Return the requested label only.",
        "generation_consistency": "Respond naturally to the spoken instruction. Keep the response semantically faithful to the instruction.",
        "generation_semantic": "Respond naturally and helpfully to the spoken instruction.",
        "generation_acoustic": "Respond naturally and helpfully to the spoken instruction.",
        "generation_response": "Respond naturally and helpfully to the spoken instruction.",
    }
    return instructions.get(task, "Respond to the user's spoken instruction naturally and accurately.")


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return clean[:120] or "case"


def _write_response_audio(data: bytes, mime: str, out_dir: Path, case_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lower = (mime or "").lower()
    suffix = ".mp3" if "mpeg" in lower or "mp3" in lower else ".wav" if "wav" in lower else ".bin"
    path = out_dir / f"{_safe_name(case_id)}{suffix}"
    path.write_bytes(data)
    if not data:
        raise RuntimeError("Model returned an empty audio response.")
    return path


_LOCAL_PAPER_WHISPER = None

def _paper_whisper_transcribe(path: Path) -> str:
    """Transcribe response audio locally with the paper-named Whisper-large-v3 checkpoint.

    This avoids invoking a hosted ASR API. It requires faster-whisper and downloads
    the open Whisper-large-v3 weights on first use. Provider/runtime parity with the
    paper authors is not asserted.
    """
    global _LOCAL_PAPER_WHISPER
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Local output transcription needs faster-whisper; install with: python -m pip install faster-whisper") from exc
    if _LOCAL_PAPER_WHISPER is None:
        model_name = _env("SOVA_OUTPUT_ASR_MODEL", "large-v3")
        device = _env("SOVA_OUTPUT_ASR_DEVICE", "cpu")
        compute_type = _env("SOVA_OUTPUT_ASR_COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
        _LOCAL_PAPER_WHISPER = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = _LOCAL_PAPER_WHISPER.transcribe(str(path), language=_env("SOVA_OUTPUT_ASR_LANGUAGE", "en"), vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _parse_multipart_response(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Parse multipart/mixed streaming chunks emitted by Mini-Omni2."""
    if "boundary=" not in content_type.lower():
        return [(content_type, body)]
    msg_bytes = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1") + body
    message = email.message_from_bytes(msg_bytes, policy=policy.default)
    parts: list[tuple[str, bytes]] = []
    if message.is_multipart():
        for part in message.iter_parts():
            payload = part.get_payload(decode=True) or b""
            parts.append((part.get_content_type(), payload))
    return parts


def _join_wav_chunks(chunks: list[bytes], out_path: Path) -> Path | None:
    """Join valid independent WAV chunks by PCM frames; ignore non-WAV chunks."""
    parsed: list[tuple[wave._wave_params, bytes]] = []
    for chunk in chunks:
        try:
            import io
            with wave.open(io.BytesIO(chunk), "rb") as reader:
                params = reader.getparams()
                frames = reader.readframes(reader.getnframes())
            parsed.append((params, frames))
        except (wave.Error, EOFError):
            continue
    if not parsed:
        return None
    first = parsed[0][0]
    compatible = [(params, frames) for params, frames in parsed if (params.nchannels, params.sampwidth, params.framerate, params.comptype) == (first.nchannels, first.sampwidth, first.framerate, first.comptype)]
    if not compatible:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as writer:
        writer.setnchannels(first.nchannels)
        writer.setsampwidth(first.sampwidth)
        writer.setframerate(first.framerate)
        writer.setcomptype(first.comptype, first.compname)
        for _, frames in compatible:
            writer.writeframes(frames)
    return out_path


class BaseAdapter:
    def __init__(self, model: dict[str, Any], out_dir: Path, *, timeout_s: float = 300.0) -> None:
        self.model = model
        self.model_id = model["id"]
        self.model_name = model["name"]
        self.out_dir = out_dir / self.model_id
        self.timeout_s = timeout_s

    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        raise NotImplementedError

    def failed(self, item: dict[str, Any], index: int, message: str) -> CaseOutput:
        return CaseOutput(
            case_id=_case_id(item, index), model_id=self.model_id,
            model_name=self.model_name, task=str(item.get("task", "unknown")),
            success=False, error=message, metric_method="not_run_or_failed_model_inference",
        )

    def require_audio(self, item: dict[str, Any], index: int) -> Path:
        path = _audio_path(item)
        if path is None:
            raise FileNotFoundError(
                f"Case {_case_id(item,index)} has no existing audio_path/audio_file/speech file. "
                "This SOVA model consumes speech; text-only records cannot be silently treated as audio inputs."
            )
        return path


class MiniOmniHTTPAdapter(BaseAdapter):
    """Native connector for official Mini-Omni's POST /chat endpoint."""
    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        base_url = _env("SOVA_MINI_OMNI_URL", self.model.get("default_base_url", "http://127.0.0.1:60808")).rstrip("/")
        output: list[CaseOutput] = []
        for idx, item in enumerate(items, 1):
            start = time.perf_counter()
            try:
                audio_path = self.require_audio(item, idx)
                data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
                response = httpx.post(
                    f"{base_url}/chat",
                    json={"audio": data, "stream_stride": int(_env("SOVA_STREAM_STRIDE", "4")), "max_tokens": int(_env("SOVA_MAX_NEW_TOKENS", "2048"))},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                case_id = _case_id(item, idx)
                audio_out = _write_response_audio(response.content, response.headers.get("content-type", "audio/wav"), self.out_dir, case_id)
                try:
                    transcript = _paper_whisper_transcribe(audio_out)
                except Exception:
                    transcript = ""
                output.append(CaseOutput(
                    case_id=case_id, model_id=self.model_id, model_name=self.model_name,
                    task=str(item.get("task", "unknown")), success=True,
                    prediction=transcript, generated_transcript=transcript,
                    response_audio_path=str(audio_out), latency_ms=(time.perf_counter()-start)*1000,
                    metric_method="official_mini_omni_http_audio_output; transcript_via_local_Whisper_large_v3" if transcript else "official_mini_omni_http_audio_output; ASR_not_configured_or_failed",
                    metadata={"endpoint": f"{base_url}/chat", "response_content_type": response.headers.get("content-type", "")},
                ))
            except Exception as exc:
                output.append(self.failed(item, idx, f"{type(exc).__name__}: {exc}"))
        return output


class MiniOmni2HTTPAdapter(BaseAdapter):
    """Native connector for official Mini-Omni2 multipart /chat endpoint."""
    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        base_url = _env("SOVA_MINI_OMNI2_URL", self.model.get("default_base_url", "http://127.0.0.1:60808")).rstrip("/")
        output: list[CaseOutput] = []
        for idx, item in enumerate(items, 1):
            start = time.perf_counter()
            try:
                audio_path = self.require_audio(item, idx)
                data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
                response = httpx.post(
                    f"{base_url}/chat",
                    json={"audio": data, "stream_stride": int(_env("SOVA_STREAM_STRIDE", "4")), "max_tokens": int(_env("SOVA_MAX_NEW_TOKENS", "2048"))},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                case_id = _case_id(item, idx)
                parts = _parse_multipart_response(response.headers.get("content-type", ""), response.content)
                text_chunks: list[str] = []
                audio_chunks: list[bytes] = []
                for mime, payload in parts:
                    if "text/plain" in mime:
                        text_chunks.append(payload.decode("utf-8", errors="replace"))
                    elif "audio/wav" in mime or "audio/x-wav" in mime:
                        audio_chunks.append(payload)
                text = "".join(text_chunks).strip()
                audio_out: Path | None = None
                if audio_chunks:
                    target = self.out_dir / f"{_safe_name(case_id)}.wav"
                    audio_out = _join_wav_chunks(audio_chunks, target)
                    if audio_out is None:
                        # Preserve the first valid-looking chunk for diagnostics.
                        audio_out = _write_response_audio(audio_chunks[-1], "audio/wav", self.out_dir, case_id)
                generated_transcript = ""
                if audio_out:
                    try:
                        generated_transcript = _paper_whisper_transcribe(audio_out)
                    except Exception:
                        generated_transcript = ""
                if not text and not generated_transcript and not audio_out:
                    raise RuntimeError("Mini-Omni2 response contained neither parseable text nor WAV audio parts.")
                # Keep native model text and ASR text from generated audio separate;
                # copying one into the other would artificially produce 0% consistency WER.
                output.append(CaseOutput(
                    case_id=case_id, model_id=self.model_id, model_name=self.model_name,
                    task=str(item.get("task", "unknown")), success=bool(text or generated_transcript or audio_out),
                    prediction=text or generated_transcript, model_text_output=text,
                    generated_transcript=generated_transcript, response_audio_path=str(audio_out or ""),
                    latency_ms=(time.perf_counter()-start)*1000,
                    metric_method="official_mini_omni2_native_text_plus_audio_and_local_Whisper_large_v3_for_consistency" if text and audio_out and generated_transcript else "official_mini_omni2_http_partial_outputs_or_ASR_unavailable",
                    metadata={"endpoint": f"{base_url}/chat", "parts": [mime for mime, _ in parts], "native_text_available": bool(text), "generated_audio_asr_available": bool(generated_transcript)},
                ))
            except Exception as exc:
                output.append(self.failed(item, idx, f"{type(exc).__name__}: {exc}"))
        return output


class WeingsHTTPAdapter(BaseAdapter):
    """Run the user's actual Weings voice API: STT + hybrid emotion + policy + TTS."""
    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        base_url = _env("SOVA_WEINGS_URL", self.model.get("default_base_url", "http://127.0.0.1:8000")).rstrip("/")
        token = _env("SOVA_WEINGS_API_TOKEN")
        timeout = self.timeout_s
        output: list[CaseOutput] = []
        try:
            with httpx.Client(base_url=base_url, timeout=timeout) as client:
                if not token:
                    token_response = client.get("/api/v1/demo/token")
                    token_response.raise_for_status()
                    token = str(token_response.json()["access_token"])
                headers = {"Authorization": f"Bearer {token}"}
                session_resp = client.post("/api/v1/voice/sessions", headers=headers)
                session_resp.raise_for_status()
                session_id = session_resp.json()["session_id"]
                for idx, item in enumerate(items, 1):
                    start = time.perf_counter()
                    try:
                        audio_path = self.require_audio(item, idx)
                        content_type = "audio/wav" if audio_path.suffix.lower() == ".wav" else "audio/flac" if audio_path.suffix.lower() == ".flac" else "audio/mpeg" if audio_path.suffix.lower() in {".mp3", ".mpeg"} else "application/octet-stream"
                        response = client.post(
                            f"/api/v1/voice/sessions/{session_id}/audio-stream-turn",
                            headers={**headers, "Content-Type": content_type},
                            content=audio_path.read_bytes(),
                        )
                        response.raise_for_status()
                        cid = _case_id(item, idx)
                        mime = response.headers.get("content-type", "audio/mpeg")
                        # Some development configurations deliberately return text/emotion headers
                        # with browser-TTS fallback and no server-generated audio. Preserve those
                        # linguistic/paralinguistic results, but do not invent an acoustic score.
                        audio_out = _write_response_audio(response.content, mime, self.out_dir, cid) if response.content else None
                        input_transcript = unquote(response.headers.get("x-voice-transcript", ""))
                        prediction = unquote(response.headers.get("x-voice-response", ""))
                        emotion = unquote(response.headers.get("x-voice-user-emotion", ""))
                        source = unquote(response.headers.get("x-voice-emotion-source", ""))
                        try:
                            confidence = float(response.headers.get("x-voice-emotion-confidence", ""))
                        except ValueError:
                            confidence = None
                        generated_transcript = ""
                        if audio_out:
                            try:
                                generated_transcript = _paper_whisper_transcribe(audio_out)
                            except Exception:
                                generated_transcript = ""
                        output.append(CaseOutput(
                            case_id=cid, model_id=self.model_id, model_name=self.model_name,
                            task=str(item.get("task", "unknown")), success=bool(prediction or input_transcript or response.content),
                            prediction=prediction, model_text_output=prediction, transcript=input_transcript, input_transcript=input_transcript,
                            generated_transcript=generated_transcript, response_audio_path=str(audio_out or ""),
                            user_emotion=emotion, emotion_confidence=confidence, emotion_source=source,
                            latency_ms=(time.perf_counter()-start)*1000,
                            metric_method="Weings authenticated audio-stream-turn API; reports backend emotion/source/confidence headers; acoustic metric unavailable when TTS fallback returns no audio",
                            metadata={"request_id": response.headers.get("x-voice-request-id"), "voice_style": unquote(response.headers.get("x-voice-voice-style", "")), "server_audio_available": bool(audio_out), "tts_fallback": response.headers.get("x-voice-tts-fallback", "")},
                        ))
                    except Exception as exc:
                        output.append(self.failed(item, idx, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:
            return [self.failed(x, i, f"Could not initialize Weings API session: {type(exc).__name__}: {exc}") for i, x in enumerate(items, 1)]
        return output


class LlamaOmniCLIAdapter(BaseAdapter):
    """Batch runner using the official LLaMA-Omni inference script and vocoder."""
    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        repo = Path(_env("SOVA_LLAMA_OMNI_REPO"))
        model_path = _env("SOVA_LLAMA_OMNI_MODEL", "Llama-3.1-8B-Omni")
        python = _env("SOVA_LLAMA_OMNI_PYTHON", sys.executable)
        if not repo.is_dir():
            return [self.failed(x, i, "Set SOVA_LLAMA_OMNI_REPO to the cloned official LLaMA-Omni repo.") for i, x in enumerate(items, 1)]
        infer = repo / "omni_speech" / "infer" / "infer.py"
        if not infer.is_file():
            return [self.failed(x, i, f"Official infer script not found: {infer}") for i, x in enumerate(items, 1)]
        run_dir = self.out_dir / "llama_omni_batch"
        run_dir.mkdir(parents=True, exist_ok=True)
        questions_path = run_dir / "questions.json"
        answers_path = run_dir / "answers.jsonl"
        questions: list[dict[str, Any]] = []
        valid: list[tuple[int, dict[str, Any]]] = []
        output: list[CaseOutput | None] = [None] * len(items)
        for idx, item in enumerate(items, 1):
            try:
                speech = self.require_audio(item, idx)
                questions.append({"id": _case_id(item, idx), "speech": str(speech), "conversations": [{"from": "human", "value": _task_instruction(item)}]})
                valid.append((idx, item))
            except Exception as exc:
                output[idx-1] = self.failed(item, idx, str(exc))
        if not questions:
            return [x if x is not None else self.failed(items[i], i+1, "Missing audio.") for i, x in enumerate(output)]
        questions_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [
            python, str(infer), "--model-path", model_path,
            "--question-file", str(questions_path), "--answer-file", str(answers_path),
            "--num-chunks", "1", "--chunk-idx", "0", "--temperature", "0",
            "--conv-mode", _env("SOVA_LLAMA_OMNI_CONV_MODE", "llama_3"),
            "--input_type", "mel", "--mel_size", "128", "--s2s",
            "--max_new_tokens", _env("SOVA_MAX_NEW_TOKENS", "256"),
        ]
        try:
            completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=float(_env("SOVA_LLAMA_OMNI_TIMEOUT", "86400")), check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"LLaMA-Omni infer.py exited {completed.returncode}: {(completed.stderr or completed.stdout)[-4000:]}")
            rows: list[dict[str, Any]] = []
            if answers_path.exists():
                for line in answers_path.read_text(encoding="utf-8").splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            by_id = {str(r.get("question_id")): r for r in rows}
            wave_files: list[Path] = []
            vocoder = _env("SOVA_LLAMA_OMNI_VOCODER", str(repo / "vocoder" / "g_00500000"))
            vocoder_cfg = _env("SOVA_LLAMA_OMNI_VOCODER_CFG", str(repo / "vocoder" / "config.json"))
            converter = repo / "omni_speech" / "infer" / "convert_jsonl_to_txt.py"
            waveform = repo / "fairseq" / "examples" / "speech_to_speech" / "generate_waveform_from_code.py"
            unit_path = run_dir / "answers.unit"
            wave_dir = run_dir / "audio"
            if any(str(r.get("prediction_units", "")).strip() for r in rows) and converter.is_file() and waveform.is_file():
                conv = subprocess.run([python, str(converter), str(answers_path), str(unit_path)], cwd=repo, capture_output=True, text=True, timeout=300, check=False)
                if conv.returncode == 0:
                    wave_dir.mkdir(exist_ok=True)
                    voc = subprocess.run([python, str(waveform), "--in-code-file", str(unit_path), "--vocoder", vocoder, "--vocoder-cfg", vocoder_cfg, "--results-path", str(wave_dir), "--dur-prediction"], cwd=repo, capture_output=True, text=True, timeout=float(_env("SOVA_LLAMA_OMNI_TIMEOUT", "86400")), check=False)
                    if voc.returncode == 0:
                        wave_files = sorted(list(wave_dir.glob("*.wav")) + list(wave_dir.glob("*.flac")))
            file_index = 0
            for idx, item in valid:
                cid = _case_id(item, idx)
                row = by_id.get(cid)
                if row is None and file_index < len(rows):
                    row = rows[file_index]
                file_index += 1
                if row is None:
                    output[idx-1] = self.failed(item, idx, f"No answer row for {cid}; inspect {answers_path}")
                    continue
                audio_out = ""
                if file_index-1 < len(wave_files):
                    dest = self.out_dir / f"{_safe_name(cid)}.wav"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(wave_files[file_index-1].read_bytes())
                    audio_out = str(dest)
                output[idx-1] = CaseOutput(
                    case_id=cid, model_id=self.model_id, model_name=self.model_name,
                    task=str(item.get("task", "unknown")), success=bool(str(row.get("prediction", "")).strip() or audio_out),
                    prediction=str(row.get("prediction", "")).strip(), model_text_output=str(row.get("prediction", "")).strip(), response_audio_path=audio_out,
                    generated_transcript="",
                    metric_method="official_LLaMA-Omni_infer.py_s2s; audio_postprocessed_with_official_HiFi-GAN_vocoder" if audio_out else "official_LLaMA-Omni_infer.py_s2s; audio_vocoder_not_available_or_failed",
                    metadata={"prediction_units": row.get("prediction_units", ""), "answer_file": str(answers_path), "stdout_tail": completed.stdout[-500:]},
                )
        except Exception as exc:
            for idx, item in valid:
                output[idx-1] = self.failed(item, idx, f"{type(exc).__name__}: {exc}")
        return [x if x is not None else self.failed(items[i], i+1, "No result returned.") for i, x in enumerate(output)]


class JSONLBridgeAdapter(BaseAdapter):
    """Batch bridge for model-specific official runtimes that lack a common API.

    The command is run once per input batch, not once per sample, so a bridge can
    load its model once and preserve streaming/stateful inference semantics.
    """
    def run_batch(self, items: list[dict[str, Any]]) -> list[CaseOutput]:
        env_key = str(self.model.get("bridge_env", ""))
        command_text = _env(env_key)
        if not command_text:
            msg = f"Model is registered but its official runtime bridge is not configured. Set {env_key} to a JSONL bridge command. See benchmark/adapters/README.md."
            return [self.failed(item, idx, msg) for idx, item in enumerate(items, 1)]
        requests_rows: list[dict[str, Any]] = []
        for idx, item in enumerate(items, 1):
            record = dict(item)
            record.update({"case_id": _case_id(item, idx), "model_id": self.model_id, "model_name": self.model_name,
                           "response_audio_dir": str(self.out_dir)})
            audio_path = _audio_path(item)
            if audio_path:
                record["audio_path"] = str(audio_path)
            requests_rows.append(record)
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests_rows)
        try:
            # POSIX parsing strips wrapper quotes on Windows as well; quoted
            # drive paths with spaces remain a single argument and backslashes survive.
            command = shlex.split(command_text, posix=True)
            proc = subprocess.run(command, input=data, text=True, capture_output=True, timeout=self.timeout_s * max(1, len(items)), check=False, cwd=ROOT)
            if proc.returncode != 0:
                raise RuntimeError(f"Bridge exited {proc.returncode}: {(proc.stderr or proc.stdout)[-4000:]}")
            parsed: dict[str, dict[str, Any]] = {}
            noisy_stdout: list[str] = []
            for line in proc.stdout.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Upstream model libraries sometimes print startup/progress logs
                    # to stdout. Keep valid JSONL records and retain diagnostics rather
                    # than converting the entire batch into a parse failure.
                    noisy_stdout.append(line)
                    continue
                if not isinstance(row, dict):
                    noisy_stdout.append(line)
                    continue
                row_id = row.get("case_id") or row.get("id")
                if row_id is None:
                    noisy_stdout.append(line)
                    continue
                parsed[str(row_id)] = row
            results: list[CaseOutput] = []
            for idx, item in enumerate(items, 1):
                cid = _case_id(item, idx)
                row = parsed.get(cid)
                if not row:
                    diagnostic = (proc.stderr + "\n" + "\n".join(noisy_stdout))[-1200:]
                    results.append(self.failed(item, idx, f"Bridge returned no JSONL result for case {cid}. Diagnostics: {diagnostic}"))
                    continue
                audio_raw = row.get("response_audio_path") or row.get("generated_audio_path") or row.get("audio_output_path") or ""
                if audio_raw and not Path(audio_raw).is_absolute():
                    audio_raw = str((ROOT / audio_raw).resolve())
                results.append(CaseOutput(
                    case_id=cid, model_id=self.model_id, model_name=self.model_name,
                    task=str(row.get("task", item.get("task", "unknown"))), success=bool(row.get("success", True)),
                    prediction=str(row.get("prediction") or row.get("text") or row.get("transcript") or ""),
                    model_text_output=str(row.get("model_text_output") or row.get("prediction") or row.get("text") or ""),
                    transcript=str(row.get("transcript") or ""), generated_transcript=str(row.get("generated_transcript") or ""),
                    response_audio_path=str(audio_raw), user_emotion=str(row.get("user_emotion") or row.get("emotion") or ""),
                    emotion_confidence=row.get("emotion_confidence"), emotion_source=str(row.get("emotion_source") or ""),
                    latency_ms=float(row.get("latency_ms", 0) or 0), error=str(row.get("error") or ""),
                    metric_method=str(row.get("metric_method") or "official_model_bridge_JSONL"), metadata=row.get("metadata") or {},
                ))
            return results
        except Exception as exc:
            return [self.failed(item, idx, f"{type(exc).__name__}: {exc}") for idx, item in enumerate(items, 1)]


TASK_ALIASES = {
    "knowledge": "knowledge_qa", "triviaqa": "knowledge_qa", "general_knowledge": "knowledge_qa",
    "asr": "speech_recognition", "librispeech": "speech_recognition", "speech_recognition_wer": "speech_recognition",
    "librisqa": "spoken_qa_multiple_choice", "spoken_qa_mc": "spoken_qa_multiple_choice",
    "spoken_squad": "spoken_qa_open_ended", "spoken_squad_qa": "spoken_qa_open_ended",
    "emotion": "emotion_recognition", "age": "age_recognition", "gender": "gender_recognition",
    "consistency": "generation_consistency", "generation_wer": "generation_consistency",
    "semantic": "generation_semantic", "generation": "generation_semantic", "generation_response": "generation_semantic", "gptscore": "generation_semantic",
    "acoustic": "generation_acoustic", "utmosv2": "generation_acoustic",
}


def canonical_task(value: Any) -> str:
    raw = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return TASK_ALIASES.get(raw, raw)


def make_adapter(model: dict[str, Any], out_dir: Path, timeout_s: float = 300.0) -> BaseAdapter:
    kind = model.get("adapter")
    adapters: dict[str, type[BaseAdapter]] = {
        "mini_omni_http": MiniOmniHTTPAdapter,
        "mini_omni_native_bridge": JSONLBridgeAdapter,
        "mini_omni2_http": MiniOmni2HTTPAdapter,
        "weings_http": WeingsHTTPAdapter,
        "llama_omni_cli": LlamaOmniCLIAdapter,
        "jsonl_bridge": JSONLBridgeAdapter,
    }
    if kind not in adapters:
        raise ValueError(f"Unknown SOVA adapter '{kind}' for model '{model.get('id')}'.")
    return adapters[kind](model, out_dir, timeout_s=timeout_s)


def run_model(model_id: str, items: list[dict[str, Any]], output_dir: Path, timeout_s: float = 300.0) -> list[dict[str, Any]]:
    registry = load_registry()
    by_id = {model["id"]: model for model in registry["models"]}
    if model_id not in by_id:
        raise KeyError(f"Unknown model '{model_id}'. Known ids: {', '.join(by_id)}")
    model = by_id[model_id]
    paper_support = model.get("paper_support", {})
    eligible_items: list[dict[str, Any]] = []
    eligible_positions: list[int] = []
    records: list[dict[str, Any] | None] = [None] * len(items)
    for idx, item in enumerate(items):
        task = canonical_task(item.get("task", "unknown"))
        split = str(item.get("dataset_split") or item.get("split") or item.get("subset") or "").lower().replace("_", "-")
        split_support = model.get("paper_support_splits", {}).get(task)
        unsupported = task in paper_support and paper_support[task] is False
        # If Table 4 only reports a subset of splits, never run unsupported splits
        # by accident simply because the manifest omitted its split name.
        unsupported_split = bool(split_support and (not split or not any(valid_split in split for valid_split in split_support)))
        # Table 4 dashes (including GLM-4-Voice's absent dev-clean/dev-other WER)
        # indicate not evaluated/reported. Never turn a paper dash into 0%.
        if unsupported or unsupported_split:
            why = (f"SOVA-Bench Table 4 reports no value for {task} on split '{split}'." if unsupported_split
                   else "SOVA-Bench Table 4 reports no value for this model-task combination.")
            record = dict(item)
            record.update({
                "model": model_id, "model_name": model["name"],
                "task": "not_evaluated_by_paper", "requested_task": task,
                "case_id": _case_id(item, idx + 1), "success": False, "answered": False,
                "measurement_status": "not_evaluated_by_paper",
                "metric_method": "excluded_per_SOVA_Bench_Table4_dash_not_zero_score",
                "error": None,
                "metadata": {"requested_task": task, "requested_split": split, "reason": why},
            })
            records[idx] = record
        else:
            eligible_items.append(item)
            eligible_positions.append(idx)
    if eligible_items:
        adapter = make_adapter(model, output_dir, timeout_s=timeout_s)
        outputs = adapter.run_batch(eligible_items)
        for position, item, result in zip(eligible_positions, eligible_items, outputs, strict=True):
            records[position] = result.to_record(item)
    return [record if record is not None else {"model": model_id, "model_name": model["name"], "task": "unknown", "success": False, "error": "No result was produced."} for record in records]
