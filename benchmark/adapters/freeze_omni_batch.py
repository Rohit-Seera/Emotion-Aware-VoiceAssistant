"""JSONL batch wrapper around the official Freeze-Omni bin/inference.py.

Run this inside the Freeze-Omni conda environment. It loads the model once for
an entire JSONL batch; one stdin JSON object and one stdout JSON object per line.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> int:
    repo = Path(os.environ.get("SOVA_FREEZE_OMNI_REPO", "")).resolve()
    model_path = os.environ.get("SOVA_FREEZE_OMNI_MODEL_PATH", str(repo / "checkpoints"))
    llm_path = os.environ.get("SOVA_FREEZE_OMNI_LLM_PATH", str(repo / "Qwen2-7B-Instruct"))
    if not repo.is_dir():
        print(json.dumps({"case_id": "__bridge__", "success": False, "error": "Set SOVA_FREEZE_OMNI_REPO to the official VITA-MLLM/Freeze-Omni checkout."}), flush=True)
        return 2
    source = repo / "bin" / "inference.py"
    if not source.is_file():
        print(json.dumps({"case_id": "__bridge__", "success": False, "error": f"Official inference file missing: {source}"}), flush=True)
        return 2
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("freeze_official_inference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official inference module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The official implementation accepts an argparse-like config namespace.
    configs = SimpleNamespace(
        model_path=model_path, llm_path=llm_path, input_wav="", output_wav="",
        top_k=int(os.environ.get("SOVA_FREEZE_TOP_K", "20")),
        top_p=float(os.environ.get("SOVA_FREEZE_TOP_P", "0.8")),
        temperature=float(os.environ.get("SOVA_FREEZE_TEMPERATURE", "0.8")),
    )
    pipeline = module.inferencePipeline(configs)
    tts = module.llm2TTS(configs.model_path)
    audio_processor = module.audioEncoderProcessor()
    temp_out = Path(os.environ.get("SOVA_FREEZE_OUTPUT_DIR", str(repo / "sova_bridge_output"))).resolve()
    temp_out.mkdir(parents=True, exist_ok=True)
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        item: dict[str, Any] = {}
        try:
            item = json.loads(raw_line)
            cid = str(item.get("case_id") or item.get("id") or "case")
            audio_path = Path(str(item.get("audio_path") or item.get("speech") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"No input audio file for {cid}: {audio_path}")
            case_out_dir = Path(str(item.get("response_audio_dir") or temp_out)).resolve()
            case_out_dir.mkdir(parents=True, exist_ok=True)
            out_audio = case_out_dir / ("".join(c if c.isalnum() or c in "-_." else "_" for c in cid) + ".wav")
            configs.input_wav = str(audio_path)
            configs.output_wav = str(out_audio)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                module.inference(pipeline, audio_processor, tts, configs)
            lines = [line.strip() for line in captured.getvalue().splitlines() if line.strip()]
            # Official inference prints its accumulated natural-language response.
            prediction = lines[-1] if lines else ""
            print(json.dumps({
                "case_id": cid, "success": out_audio.is_file() and out_audio.stat().st_size > 44,
                "prediction": prediction, "model_text_output": prediction,
                "response_audio_path": str(out_audio) if out_audio.is_file() else "",
                "metric_method": "official_Freeze-Omni_bin/inference.py; batch wrapper keeps model loaded",
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"case_id": item.get("case_id", item.get("id", "unknown")) if "item" in locals() else "unknown", "success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
