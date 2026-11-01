"""JSONL batch bridge for official Mini-Omni task-specific inference.

Speech-recognition records use the upstream A1_T1 ASR path. Other audio-in
records use the upstream A1_A2 speech-to-speech path so the model's native text
and generated audio can be preserved for generation evaluation.

Run inside Mini-Omni's own Python environment. STDOUT is reserved for JSONL;
upstream logs are redirected to STDERR.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)[:120] or "case"


def _is_asr_task(item: dict[str, Any]) -> bool:
    task = str(item.get("task", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return task in {"speech_recognition", "asr", "librispeech", "speech_recognition_wer"}


def main() -> int:
    repo = Path(os.environ.get("SOVA_MINI_OMNI_REPO", "")).resolve()
    checkpoint = Path(os.environ.get("SOVA_MINI_OMNI_CHECKPOINT", str(repo / "checkpoint"))).resolve()
    device = os.environ.get("SOVA_MINI_OMNI_DEVICE", "cuda:0")

    if not repo.is_dir():
        print(json.dumps({"case_id": "__bridge__", "success": False,
                          "error": "Set SOVA_MINI_OMNI_REPO to the official gpt-omni/mini-omni checkout."}), flush=True)
        return 2
    required_checkpoint_files = ["lit_model.pth", "model_config.yaml", "tokenizer_config.json"]
    missing = [name for name in required_checkpoint_files if not (checkpoint / name).is_file()]
    if missing:
        print(json.dumps({"case_id": "__bridge__", "success": False,
                          "error": f"Mini-Omni checkpoint is incomplete at {checkpoint}; missing {missing}. Download the official gpt-omni/mini-omni checkpoint."}), flush=True)
        return 2

    sys.path.insert(0, str(repo))
    try:
        # Import the official A1_T1 ASR and A1_A2 speech-to-speech paths.
        # Keep stdout JSONL-only for the parent benchmark runner.
        with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
            from inference import (
                OmniInference,
                load_audio,
                get_input_ids_whisper,
                A1_T1,
                A1_A2,
                _pad_a,
                _asr,
            )
            inference = OmniInference(str(checkpoint), device)
    except Exception as exc:
        print(json.dumps({"case_id": "__bridge__", "success": False,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-3000:]}), flush=True)
        return 2

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        item: dict[str, Any] = {}
        try:
            item = json.loads(raw_line)
            cid = str(item.get("case_id") or item.get("id") or "case")
            raw_audio = item.get("audio_path") or item.get("speech") or ""
            audio_path = Path(str(raw_audio)).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio input for {cid}: {audio_path}")

            out_root = Path(str(item.get("response_audio_dir") or
                                 os.environ.get("SOVA_MINI_OMNI_OUTPUT_DIR", Path.cwd() / "sova_mini_omni_output"))).resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            case_root = out_root / (".mini_omni_work_" + _safe(cid))
            if case_root.exists():
                shutil.rmtree(case_root)

            with contextlib.redirect_stdout(sys.stderr), contextlib.redirect_stderr(sys.stderr):
                mel, length = load_audio(str(audio_path))
                if _is_asr_task(item):
                    # Match upstream inference.py's official ASR task: A1_T1.
                    # Its ASR prompt uses _pad_a and _asr, not the A1_A2 tokens.
                    audio_feature, input_ids = get_input_ids_whisper(
                        mel, length, inference.whispermodel, device,
                        special_token_a=_pad_a,
                        special_token_t=_asr,
                    )
                    text = A1_T1(
                        inference.fabric, audio_feature, input_ids, length,
                        inference.model, inference.text_tokenizer, 0,
                    )
                    is_asr = True
                    final_audio = None
                else:
                    audio_feature, input_ids = get_input_ids_whisper(
                        mel, length, inference.whispermodel, device,
                    )
                    text = A1_A2(
                        inference.fabric, audio_feature, input_ids, length,
                        inference.model, inference.text_tokenizer, 0,
                        inference.snacmodel, out_dir=str(case_root),
                    )
                    is_asr = False
                    generated = case_root / "A1-A2" / "00.wav"
                    final_audio = out_root / (_safe(cid) + ".wav")
                    if generated.is_file():
                        shutil.copy2(generated, final_audio)
                    else:
                        final_audio = None

            text = str(text or "").strip()
            success = bool(text) if is_asr else bool(text or (final_audio and final_audio.is_file()))
            row = {
                "case_id": cid,
                "task": item.get("task", "unknown"),
                "success": success,
                "prediction": text,
                "model_text_output": text,
                "transcript": text if is_asr else "",
                "response_audio_path": str(final_audio) if final_audio and final_audio.is_file() else "",
                "metric_method": "official_Mini-Omni_A1_T1_ASR" if is_asr else "official_Mini-Omni_A1_A2_speech_to_speech",
                "metadata": {
                    "official_repo": "https://github.com/gpt-omni/mini-omni",
                    "checkpoint": str(checkpoint),
                    "device": device,
                    "inference_path": "A1_T1" if is_asr else "A1_A2",
                },
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({
                "case_id": item.get("case_id", item.get("id", "unknown")),
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2500:],
            }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
