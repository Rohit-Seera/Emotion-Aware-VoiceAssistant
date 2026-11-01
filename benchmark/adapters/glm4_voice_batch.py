"""JSONL adapter for the official GLM-4-Voice tokenizer/server/decoder stack.

Run in the GLM-4-Voice environment after starting its official model_server.py
on port 10000. This wrapper turns input WAVs into speech tokens, calls the
official /generate_stream endpoint, then decodes the generated audio token IDs.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
import torch


def main() -> int:
    repo = Path(os.environ.get("SOVA_GLM4_VOICE_REPO", "")).resolve()
    if not repo.is_dir():
        print(json.dumps({"case_id": "__bridge__", "success": False, "error": "Set SOVA_GLM4_VOICE_REPO to the official zai-org/GLM-4-Voice checkout."}), flush=True)
        return 2
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "cosyvoice"))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
    import torchaudio
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from speech_tokenizer.utils import extract_speech_token
    from flow_inference import AudioDecoder

    model_path = os.environ.get("SOVA_GLM4_VOICE_MODEL_PATH", "THUDM/glm-4-voice-9b")
    tokenizer_path = os.environ.get("SOVA_GLM4_VOICE_TOKENIZER_PATH", "THUDM/glm-4-voice-tokenizer")
    decoder_path = Path(os.environ.get("SOVA_GLM4_VOICE_DECODER_PATH", str(repo / "glm-4-voice-decoder"))).resolve()
    server_url = os.environ.get("SOVA_GLM4_VOICE_SERVER_URL", "http://127.0.0.1:10000").rstrip("/")
    device = os.environ.get("SOVA_GLM4_VOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens = int(os.environ.get("SOVA_MAX_NEW_TOKENS", "512"))
    glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    audio_decoder = AudioDecoder(
        config_path=str(decoder_path / "config.yaml"),
        flow_ckpt_path=str(decoder_path / "flow.pt"),
        hift_ckpt_path=str(decoder_path / "hift.pt"),
        device=device,
    )
    whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    out_dir = Path(os.environ.get("SOVA_GLM4_VOICE_OUTPUT_DIR", str(Path.cwd() / "sova_glm4_output"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_zero = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
    user_end = glm_tokenizer.convert_tokens_to_ids("<|user|>")

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        item: dict[str, Any] = {}
        try:
            item = json.loads(raw_line)
            cid = str(item.get("case_id") or item.get("id") or "case")
            audio_path = Path(str(item.get("audio_path") or item.get("speech") or "")).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing input WAV for {cid}: {audio_path}")
            input_tokens = extract_speech_token(whisper_model, feature_extractor, [str(audio_path)])[0]
            if len(input_tokens) == 0:
                raise RuntimeError("GLM-4-Voice tokenizer produced no audio tokens")
            audio_token_text = "<|begin_of_audio|>" + "".join(f"<|audio_{int(x)}|>" for x in input_tokens) + "<|end_of_audio|>"
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in an interleaved manner, with 13 text token followed by 26 audio tokens."
            prompt = f"<|system|>\n{system_prompt}<|user|>\n{audio_token_text}<|assistant|>streaming_transcription\n"
            response = requests.post(
                f"{server_url}/generate_stream",
                data=json.dumps({"prompt": prompt, "temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new_tokens}),
                stream=True,
                timeout=(20, 900),
            )
            response.raise_for_status()
            response_text_parts: list[str] = []
            audio_ids: list[int] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                packet = json.loads(line)
                token_id = int(packet.get("token_id", -1))
                if token_id == user_end:
                    break
                token_str = str(glm_tokenizer.convert_ids_to_tokens(token_id)) if token_id >= 0 else ""
                match = re.fullmatch(r"<\|audio_(\d+)\|>", token_str)
                if match:
                    audio_ids.append(int(match.group(1)))
                elif token_id >= 0:
                    decoded = glm_tokenizer.decode([token_id], skip_special_tokens=True)
                    if decoded:
                        response_text_parts.append(decoded)
            prediction = "".join(response_text_parts).strip()
            case_out_dir = Path(str(item.get("response_audio_dir") or out_dir)).resolve()
            case_out_dir.mkdir(parents=True, exist_ok=True)
            out_audio_path = case_out_dir / ("".join(c if c.isalnum() or c in "-_." else "_" for c in cid) + ".wav")
            audio_ok = False
            if audio_ids:
                token_tensor = torch.tensor(audio_ids, device=device, dtype=torch.int64).unsqueeze(0)
                empty_feat = torch.zeros(1, 0, 80, device=device)
                empty_token = torch.zeros(1, 0, dtype=torch.int64, device=device)
                speech, _ = audio_decoder.token2wav(
                    token_tensor, uuid=str(uuid.uuid4()), prompt_token=empty_token,
                    prompt_feat=empty_feat, finalize=True,
                )
                sf.write(str(out_audio_path), speech.detach().cpu().numpy()[0].reshape(-1), 22050)
                audio_ok = out_audio_path.is_file() and out_audio_path.stat().st_size > 44
            print(json.dumps({
                "case_id": cid, "success": bool(prediction or audio_ok), "prediction": prediction,
                "model_text_output": prediction, "response_audio_path": str(out_audio_path) if audio_ok else "",
                "metric_method": "official_GLM-4-Voice_speech_tokenizer+model_server_generate_stream+official_AudioDecoder",
                "metadata": {"audio_token_count": len(audio_ids), "server_url": server_url},
            }, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"case_id": item.get("case_id", item.get("id", "unknown")), "success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
