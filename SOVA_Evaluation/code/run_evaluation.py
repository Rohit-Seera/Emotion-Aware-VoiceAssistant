"""Evaluate the Emotion Aware Voice Assistant on the saved SOVA task samples."""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

RUN = Path(__file__).resolve().parents[1]
INPUT = RUN / "inputs"
OUT = RUN / "outputs"
LOG = RUN / "logs"
sys.path.insert(0, str(RUN / "source_backup"))
from sova_metrics import normalize_words, corpus_wer, percent  # noqa: E402

BACKEND = os.getenv("SOVA_ASSISTANT_URL", "http://127.0.0.1:8000").rstrip("/")
LLM_PAUSE = float(os.getenv("SOVA_LLM_PAUSE_SECONDS", "6.0"))
STT_PAUSE = float(os.getenv("SOVA_ASR_PAUSE_SECONDS", "6.0"))
MAX_RECENT_FAILURES = 8
MAX_ACCUMULATED_TOKENS = 185_000


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for n, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                        found[str(row["case_id"])] = row
                    except Exception as exc:
                        raise RuntimeError(f"Invalid result JSONL line {n}: {exc}") from exc
    return found


def token_f1(prediction: str, references: list[str]) -> float:
    pred = normalize_words(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for ref_text in references:
        ref = normalize_words(ref_text)
        if not ref:
            continue
        common: dict[str, int] = {}
        for word in ref:
            common[word] = common.get(word, 0) + 1
        overlap = 0
        for word in pred:
            if common.get(word, 0) > 0:
                overlap += 1
                common[word] -= 1
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_words(prediction), normalize_words(reference)
    if not pred or not ref:
        return 0.0
    common = lcs_length(pred, ref)
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def clean_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^(?:the\s+)?(?:short\s+answer\s+is|answer\s+is|answer)\s*[:\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"^(?:option\s+)?[A-D][).:\s]+", "", text, flags=re.I)
    return text.strip()


def parse_choice(text: str) -> str | None:
    m = re.search(r"\b(?:option\s*)?([A-D])\b", str(text or ""), flags=re.I)
    return m.group(1).upper() if m else None


def task_manifest(task: str) -> list[dict[str, Any]]:
    names = {
        "triviaqa": "triviaqa_eval_sample_55.json",
        "librisqa": "librisqa_eval_sample_55.json",
        "spoken_squad": "spoken_squad_eval_sample_55.json",
        "alpaca": "alpaca_eval_sample_55.json",
    }
    return read_json(INPUT / names[task])


def make_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "triviaqa":
        return (
            "Answer this factual knowledge question with the shortest correct answer. "
            "Do not explain.\nQuestion: " + row["prompt"]
        )
    if task == "librisqa":
        return (
            "Answer the multiple-choice question using only the spoken passage transcript. "
            "Return one letter A, B, C, or D only.\nPassage transcript: " + row["recognized_text"] +
            "\n\n" + row["question_and_option"]
        )
    if task == "spoken_squad":
        if not row.get("recognized_text"):
            raise ValueError("Spoken-SQuAD transcript missing; run audio transcription first")
        return (
            "Answer the question using only the spoken passage transcript. Give the shortest correct answer.\n"
            "Passage transcript: " + row["recognized_text"] +
            "\nQuestion: " + row["question"]
        )
    if task == "alpaca":
        prompt = row["instruction"].strip()
        if row.get("input", "").strip():
            prompt += "\n\nInput:\n" + row["input"].strip()
        return prompt
    raise KeyError(task)


def score(task: str, row: dict[str, Any]) -> dict[str, Any]:
    prediction = str(row.get("prediction", "")).strip()
    if task == "triviaqa":
        refs = [str(x) for x in (row.get("expected") or row.get("references") or []) if str(x).strip()]
        cleaned = clean_answer(prediction)
        norm_pred = normalize_words(cleaned)
        exact = any(norm_pred == normalize_words(ref) for ref in refs)
        return {"answered": bool(prediction), "correct_exact_match": exact, "token_f1": token_f1(cleaned, refs)}
    if task == "librisqa":
        chosen = parse_choice(prediction)
        ref = str(row["answer"]).strip().upper()
        return {"answered": chosen is not None, "predicted_choice": chosen, "correct": chosen == ref if chosen else False}
    if task == "spoken_squad":
        refs = [str(x) for x in row.get("answer_aliases", [row["reference_answer"]]) if str(x).strip()]
        cleaned = clean_answer(prediction)
        return {
            "answered": bool(prediction),
            "exact_match": any(normalize_words(cleaned) == normalize_words(ref) for ref in refs),
            "token_f1": token_f1(cleaned, refs),
        }
    if task == "alpaca":
        ref = str(row["reference_output"])
        return {
            "answered": bool(prediction),
            "rouge_l_f1": rouge_l_f1(prediction, ref),
            "token_f1": token_f1(prediction, [ref]),
        }
    raise KeyError(task)


def summarize(manifests: dict[str, list[dict[str, Any]]], results: dict[str, dict[str, Any]], token_total: int) -> dict[str, Any]:
    task_summary: dict[str, Any] = {}
    for task, cases in manifests.items():
        rows = [results[str(row["case_id"])] for row in cases if str(row["case_id"]) in results]
        success = [row for row in rows if row.get("success")]
        answered = [row for row in success if row.get("answered")]
        item: dict[str, Any] = {
            "target_cases": len(cases), "attempted_cases": len(rows), "successful_cases": len(success),
            "failed_cases": len(rows) - len(success), "answer_rate_percent": percent(len(answered), len(cases)),
        }
        if task == "triviaqa":
            item["exact_match_accuracy_percent"] = percent(sum(bool(x.get("correct_exact_match")) for x in success), len(cases))
            item["mean_token_f1_percent"] = 100 * statistics.fmean([x["token_f1"] for x in success]) if success else None
            item["metric_note"] = "Exact-match and token-F1 proxies; SOVA paper used GPT-evaluated accuracy."
        elif task == "librisqa":
            correct = sum(bool(x.get("correct")) for x in success)
            item["accuracy_percent"] = percent(correct, len(cases))
            item["conditioned_accuracy_percent"] = percent(correct, sum(bool(x.get("answered")) for x in success))
            item["metric_note"] = "Four-choice accuracy; evaluated cascade-style from prior ASR transcripts."
        elif task == "spoken_squad":
            item["exact_match_accuracy_percent"] = percent(sum(bool(x.get("exact_match")) for x in success), len(cases))
            item["mean_token_f1_percent"] = 100 * statistics.fmean([x["token_f1"] for x in success]) if success else None
            item["metric_note"] = "Exact-match and token-F1 proxies; SOVA paper used GPT-evaluated accuracy."
        elif task == "alpaca":
            item["mean_rouge_l_f1_percent"] = 100 * statistics.fmean([x["rouge_l_f1"] for x in success]) if success else None
            item["mean_token_f1_percent"] = 100 * statistics.fmean([x["token_f1"] for x in success]) if success else None
            item["metric_note"] = "Text-only lexical proxies; not paper GPTScore, consistency WER, or UTMOSv2."
        item["completed"] = len(success) == len(cases)
        task_summary[task] = item
    return {
        "benchmark": "SOVA-Bench compatible task-family follow-up samples",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Emotion Aware Voice Assistant configured backend (Groq GPT-OSS 120B; Groq Whisper large-v3-turbo for Spoken SQuAD audio)",
        "sampling_seed": 20261031,
        "sampling": "fixed-seed random sample per accessible dataset; sample sizes vary by free-tier constraints",
        "llm_tokens_reported_by_backend": token_total,
        "task_results": task_summary,
        "unmeasured_sova_cells": {
            "emotion_iemocap": "Not run: IEMOCAP access files are not available locally; paper requires its 5-way label mapping.",
            "age_common_voice": "Not run: the assistant has no age-prediction output; Common Voice source needs authorized download.",
            "gender_common_voice": "Not run: the assistant has no gender-prediction output; Common Voice source needs authorized download.",
            "alpaca_speech_consistency_and_acoustic": "Not run in this text pass: TTS/UTMOSv2 would require speech generation and additional runtime/credits.",
            "paper_baseline_models": "Published paper values are references only; baseline checkpoints were not run locally.",
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    manifests = {task: task_manifest(task) for task in ("triviaqa", "librisqa", "alpaca", "spoken_squad")}
    if not (INPUT / "spoken_squad_eval_sample_55.json").exists():
        raise SystemExit("Create the 55-case Spoken SQuAD evaluation manifest first.")
    case_path = OUT / "task_case_results.jsonl"
    results = load_outputs(case_path)
    token_total = sum(int((row.get("token_usage") or {}).get("total_tokens", 0)) for row in results.values())
    print(f"Configured LLM={len(manifests['triviaqa'])+len(manifests['librisqa'])+len(manifests['alpaca'])} text samples; Spoken SQuAD audio={len(manifests['spoken_squad'])}; already successful={sum(bool(r.get('success')) for r in results.values())}; token total={token_total}; pause={LLM_PAUSE}s", flush=True)
    recent_failures = 0
    consecutive_provider_failures = 0
    limit_stop = False
    with httpx.Client(timeout=httpx.Timeout(150, connect=15)) as client, case_path.open("a", encoding="utf-8", buffering=1) as output:
        health = client.get(f"{BACKEND}/api/v1/health/ready")
        health.raise_for_status()

        def fresh_token() -> str:
            response = client.get(f"{BACKEND}/api/v1/demo/token")
            response.raise_for_status()
            return response.json()["access_token"]

        token = fresh_token()
        headers = {"Authorization": f"Bearer {token}"}
        order = ("triviaqa", "librisqa", "spoken_squad", "alpaca")
        for task in order:
            if limit_stop:
                break
            cases = manifests[task]
            print(f"Starting {task}: {len(cases)} cases", flush=True)
            for index, case in enumerate(cases, 1):
                case_id = str(case["case_id"])
                if results.get(case_id, {}).get("success"):
                    continue
                current: dict[str, Any] = {
                    "case_id": case_id, "task": task, "dataset": {"triviaqa": "TriviaQA", "librisqa": "LibriSQA Part II", "spoken_squad": "Spoken SQuAD", "alpaca": "Alpaca"}[task],
                    "success": False,
                }
                started = time.perf_counter()
                turn = None
                stt = None
                try:
                    recognized = case.get("recognized_text")
                    if task == "spoken_squad":
                        audio_path = Path(case["audio_path"])
                        if not audio_path.is_absolute():
                            audio_path = INPUT / audio_path
                        audio = audio_path.read_bytes()
                        stt_headers = {**headers, "Content-Type": "audio/wav"}
                        stt = client.post(f"{BACKEND}/api/v1/voice/transcriptions?language=en", headers=stt_headers, content=audio)
                        if stt.status_code == 401:
                            token = fresh_token()
                            headers = {"Authorization": f"Bearer {token}"}
                            stt = client.post(f"{BACKEND}/api/v1/voice/transcriptions?language=en", headers={**headers, "Content-Type": "audio/wav"}, content=audio)
                        current["stt_http_status"] = stt.status_code
                        if stt.status_code == 429:
                            current["error"] = "HTTP 429 STT rate limit; checkpoint and stop Spoken SQuAD sample"
                            limit_stop = True
                        else:
                            stt.raise_for_status()
                            stt_payload = stt.json()
                            recognized = str(stt_payload.get("transcript", "")).strip()
                            current.update(stt_transcript=recognized, stt_language=stt_payload.get("language"), audio_duration_seconds=stt_payload.get("duration_seconds"), audio_path=str(audio_path))
                    if not limit_stop:
                        prompt_case = dict(case)
                        if task == "spoken_squad":
                            prompt_case["recognized_text"] = recognized
                        session = client.post(f"{BACKEND}/api/v1/voice/sessions", headers=headers)
                        if session.status_code == 401:
                            token = fresh_token()
                            headers = {"Authorization": f"Bearer {token}"}
                            session = client.post(f"{BACKEND}/api/v1/voice/sessions", headers=headers)
                        session.raise_for_status()
                        session_id = session.json()["session_id"]
                        turn = client.post(
                            f"{BACKEND}/api/v1/voice/sessions/{session_id}/turns",
                            headers=headers,
                            json={"transcript": make_prompt(task, prompt_case), "language": "en"},
                        )
                        if turn.status_code == 401:
                            token = fresh_token()
                            headers = {"Authorization": f"Bearer {token}"}
                            turn = client.post(f"{BACKEND}/api/v1/voice/sessions/{session_id}/turns", headers=headers, json={"transcript": make_prompt(task, prompt_case), "language": "en"})
                        current["llm_http_status"] = turn.status_code
                        if turn.status_code >= 400:
                            current["llm_response_detail"] = turn.text[:1000]
                        turn.raise_for_status()
                        payload = turn.json()
                        current.update(
                            success=True, prediction=str(payload.get("text", "")).strip(),
                            user_emotion=payload.get("user_emotion"), emotion_source=payload.get("emotion_source"),
                            processing_ms=payload.get("processing_ms"), llm_provider=payload.get("llm_provider"),
                            llm_model=payload.get("llm_model"), token_usage=payload.get("token_usage"),
                        )
                        current.update(score(task, {**case, **current}))
                        consecutive_provider_failures = 0
                        recent_failures = 0
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    response = getattr(exc, "response", None)
                    if response is not None and not current.get("llm_response_detail"):
                        current["llm_response_detail"] = str(getattr(response, "text", ""))[:1000]
                    current["error"] = f"{type(exc).__name__}: {str(exc)[:350]}"
                    if status is not None and status >= 500:
                        consecutive_provider_failures += 1
                    elif status is None:
                        recent_failures += 1
                    else:
                        consecutive_provider_failures = 0
                    if consecutive_provider_failures >= MAX_RECENT_FAILURES:
                        limit_stop = True
                current["latency_seconds"] = round(time.perf_counter() - started, 4)
                current["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
                output.write(json.dumps(current, ensure_ascii=False) + "\n")
                output.flush()
                results[case_id] = current
                if current.get("token_usage"):
                    token_total += int(current["token_usage"].get("total_tokens", 0))
                if index % 25 == 0 or index == len(cases):
                    summary = summarize(manifests, results, token_total)
                    (OUT / "task_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                    task_stats = summary["task_results"][task]
                    print(f"{task} {index}/{len(cases)} successful={task_stats['successful_cases']} errors={task_stats['failed_cases']} total_tokens={token_total}", flush=True)
                if token_total >= MAX_ACCUMULATED_TOKENS:
                    print(f"Stopping at safety token budget {MAX_ACCUMULATED_TOKENS}; rerun resumes later.", flush=True)
                    limit_stop = True
                if limit_stop:
                    break
                time.sleep(STT_PAUSE if task == "spoken_squad" else LLM_PAUSE)
    summary = summarize(manifests, results, token_total)
    (OUT / "task_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if all(x["completed"] for x in summary["task_results"].values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; checkpointed results remain resumable.", file=sys.stderr)
        raise SystemExit(130)
