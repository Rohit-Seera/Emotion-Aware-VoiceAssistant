"""JSONL batch wrapper using the official Kyutai Moshi PyTorch API.

Run inside the official Moshi Python environment. The checkpoint/model is loaded
once and each audio case is processed in a fresh streaming context. The paper's
2.5-second silence prefix for Moshi is applied by default.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio


def main() -> int:
    from moshi.models import LMGen, loaders

    device = os.environ.get("SOVA_MOSHI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    hf_repo = os.environ.get("SOVA_MOSHI_HF_REPO", "kyutai/moshiko-pytorch-bf16")
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16 if device.startswith("cuda") else torch.float32
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)
    text_tokenizer = info.get_text_tokenizer()
    moshi = info.get_moshi(device=device, dtype=dtype)
    generator = LMGen(moshi, temp=float(os.environ.get("SOVA_MOSHI_TEMP", "0.8")), temp_text=float(os.environ.get("SOVA_MOSHI_TEXT_TEMP", "0.7")))
    out_dir = Path(os.environ.get("SOVA_MOSHI_OUTPUT_DIR", str(Path.cwd() / "sova_moshi_output"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    prefix_seconds = float(os.environ.get("SOVA_MOSHI_SILENCE_PREFIX_SECONDS", "2.5"))

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        item: dict[str, Any] = {}
        try:
            item = json.loads(raw_line)
            cid = str(item.get("case_id") or item.get("id") or "case")
            audio_path = Path(str(item.get("audio_path") or item.get("speech") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio path for {cid}: {audio_path}")
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio).view(1, 1, -1)
            if sr != int(mimi.sample_rate):
                waveform = torchaudio.functional.resample(waveform.squeeze(0), orig_freq=sr, new_freq=int(mimi.sample_rate)).unsqueeze(0)
            if prefix_seconds > 0:
                prefix = torch.zeros((1, 1, int(prefix_seconds * int(mimi.sample_rate))), dtype=waveform.dtype)
                waveform = torch.cat([prefix, waveform], dim=-1)
            length = waveform.shape[-1]
            pad = (-length) % frame_size
            if pad:
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            waveform = waveform.to(device)
            audio_chunks: list[torch.Tensor] = []
            text_chunks: list[str] = []
            with torch.no_grad(), mimi.streaming(1), generator.streaming(1):
                for offset in range(0, waveform.shape[-1], frame_size):
                    frame = waveform[:, :, offset:offset + frame_size]
                    codes = mimi.encode(frame)
                    tokens_out = generator.step(codes[:, :, :1])
                    if tokens_out is None:
                        continue
                    text_token = int(tokens_out[:, 0, 0][0].item())
                    if text_token not in (0, 3):
                        piece = text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                        text_chunks.append(piece)
                    audio_tokens = tokens_out[:, 1:]
                    if audio_tokens.numel():
                        decoded = mimi.decode(audio_tokens)
                        audio_chunks.append(decoded.detach().float().cpu())
            prediction = "".join(text_chunks).strip()
            case_out_dir = Path(str(item.get("response_audio_dir") or out_dir)).resolve()
            case_out_dir.mkdir(parents=True, exist_ok=True)
            out_audio = case_out_dir / ("".join(c if c.isalnum() or c in "-_." else "_" for c in cid) + ".wav")
            if out_audio.exists():
                out_audio.unlink()
            if audio_chunks:
                out_wave = torch.cat(audio_chunks, dim=-1)[0, 0].numpy().astype(np.float32)
                sf.write(str(out_audio), out_wave, int(mimi.sample_rate))
            success = bool(prediction or out_audio.is_file())
            print(json.dumps({
                "case_id": cid, "success": success, "prediction": prediction,
                "model_text_output": prediction, "response_audio_path": str(out_audio) if out_audio.is_file() else "",
                "metric_method": "official_Kyutai_Moshi_PyTorch_LMGen+Mimi; 2.5s silence prefix per supplied SOVA paper protocol",
                "metadata": {"hf_repo": hf_repo, "sample_rate": int(mimi.sample_rate), "silence_prefix_seconds": prefix_seconds},
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"case_id": item.get("case_id", item.get("id", "unknown")), "success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
