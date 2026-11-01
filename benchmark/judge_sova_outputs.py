"""Use the user’s existing GPT-OSS 120B via Groq to judge SOVA outputs without the paid OpenAI API.

The five-page SOVA preprint names GPTEval/GPTScore but does not specify all
judge prompts. This script versions its own prompt and records that it is a
compatible operationalization, not an exact author-prompt reproduction.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

TASK_ALIASES = {
    "knowledge": "knowledge_qa", "triviaqa": "knowledge_qa", "trivia_qa": "knowledge_qa",
    "librisqa": "spoken_qa_multiple_choice", "spoken_qa": "spoken_qa_multiple_choice",
    "spoken_squad": "spoken_qa_open_ended", "open_ended_qa": "spoken_qa_open_ended",
    "generation": "generation_semantic", "open_ended": "generation_semantic", "generation_response": "generation_semantic",
    "emotion": "emotion_recognition", "age": "age_recognition", "gender": "gender_recognition",
}
PROMPT_VERSION = "weings-sova-judge-v1"


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end+1]
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("Judge response is not a JSON object")
    return obj


def call_judge(client: httpx.Client, model: str, api_key: str, prompt: str) -> dict[str, Any]:
    response = client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a strict, evidence-based benchmark judge. Return only the requested JSON object."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return parse_json_object(text)


def get_prompt(row: dict[str, Any], task: str) -> str | None:
    question = str(row.get("prompt") or row.get("question") or row.get("instruction") or row.get("query") or "")
    if task in {"emotion_recognition", "age_recognition", "gender_recognition"}:
        # For Weings, the emotion prediction is an explicit acoustic-SER output in
        # the API headers; do not mistake the assistant's comforting reply for its
        # predicted label.
        if task == "emotion_recognition":
            answer = row.get("weings_predicted_emotion") or row.get("predicted_label") or row.get("prediction_label") or row.get("prediction") or row.get("output")
        else:
            answer = row.get("predicted_label") or row.get("prediction_label") or row.get("prediction") or row.get("output")
        expected = row.get("expected_label") or row.get("gold_label") or row.get("label") or row.get("speaker_emotion" if task == "emotion_recognition" else "speaker_age_label" if task == "age_recognition" else "speaker_gender")
        labels = row.get("labels") or row.get("choices") or row.get("options")
        if isinstance(labels, dict):
            labels = list(labels.values())
        if not isinstance(labels, list) or not labels:
            labels = []
        if answer is None or expected is None:
            return None
        return (
            "Evaluate this categorical speech-understanding prediction against the ground-truth label. "
            "Count equivalent label wording as correct only when it denotes the same category. "
            "Do not infer the gold label from the answer. Return JSON {\"correct\": true|false, \"reason\": \"brief\"}.\n"
            f"Task: {task}\nAllowed label set: {json.dumps(labels, ensure_ascii=False)}\n"
            f"Ground-truth label: {expected}\nPredicted label/output: {answer}"
        )
    answer = row.get("prediction") or row.get("model_text_output") or row.get("output") or row.get("response_text") or row.get("transcript")
    if not isinstance(answer, str) or not answer.strip():
        return None
    if task in {"knowledge_qa", "spoken_qa_open_ended", "spoken_qa_multiple_choice"}:
        refs = row.get("answers") or row.get("expected") or row.get("reference_answers") or row.get("reference_answer") or row.get("answer")
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not refs:
            return None
        return (
            "Judge whether the assistant response answers the question correctly, allowing valid paraphrases and equivalent facts. "
            "Do not reward an answer that contradicts the references. Return JSON {\"correct\": true|false, \"reason\": \"brief\"}.\n"
            f"Question/instruction: {question}\nReference answer(s): {json.dumps(refs, ensure_ascii=False)}\nAssistant response: {answer}"
        )
    if task == "generation_semantic":
        return (
            "Rate the semantic quality of the generated voice-assistant response on a 1-5 integer scale. "
            "5 = fully relevant, correct, coherent, and follows the request; 4 = good with minor issues; "
            "3 = partially useful or noticeably incomplete; 2 = mostly poor/off-target; 1 = unusable, incoherent, or not responsive. "
            "Judge semantic/text quality only, not the audio naturalness. Return JSON {\"gptscore\": 1..5, \"reason\": \"brief\"}.\n"
            f"User request: {question}\nAssistant response: {answer}"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Default: update input in place after creating a one-time backup.")
    parser.add_argument("--model", default=None, help="Judge model; defaults to SOVA_JUDGE_MODEL or openai/gpt-oss-120b.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    key = env("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is required for the existing Groq GPT-OSS 120B judge; this script does not call OpenAI.")
    judge_model = args.model or env("SOVA_JUDGE_MODEL", "openai/gpt-oss-120b")
    source = args.input.resolve()
    output = (args.output or source).resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Input must be a JSON list of result rows.")
    if output == source:
        backup = source.with_name(source.stem + ".pre_gpt_judge" + source.suffix)
        if not backup.exists():
            shutil.copy2(source, backup)
        print(f"Backup: {backup}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempted = 0
    errors = 0
    with httpx.Client() as client:
        for index, row in enumerate(rows):
            raw_task = str(row.get("task", "")).lower()
            task = TASK_ALIASES.get(raw_task, raw_task)
            if task not in {"knowledge_qa", "spoken_qa_open_ended", "spoken_qa_multiple_choice", "emotion_recognition", "age_recognition", "gender_recognition", "generation_semantic"}:
                continue
            if row.get("success", True) is False:
                continue
            if task == "generation_semantic" and (row.get("gptscore") is not None or row.get("paper_gptscore") is not None):
                continue
            if task != "generation_semantic" and row.get("judge_correct") is not None:
                continue
            prompt = get_prompt(row, task)
            if not prompt:
                row["gpt_judge_error"] = "Missing prediction/response or reference answer(s); no judge score assigned."
                continue
            if args.limit is not None and attempted >= args.limit:
                break
            attempted += 1
            try:
                judgement = call_judge(client, judge_model, key, prompt)
                row["judge_model"] = judge_model
                row["judge_provider"] = "Groq API (user-configured access)"
                row["judge_prompt_version"] = PROMPT_VERSION
                if task == "generation_semantic":
                    value = int(judgement.get("gptscore", 0))
                    if value not in range(1, 6):
                        raise ValueError(f"Judge returned out-of-range GPTScore: {value}")
                    row["gptscore"] = float(value)
                    row["metric_method"] = f"{PROMPT_VERSION}_GPTScore_1_to_5_GPT_OSS_120B_operationalization_not_authors_hidden_prompt"
                else:
                    value = judgement.get("correct")
                    if not isinstance(value, bool):
                        raise ValueError("Judge response did not contain a boolean 'correct'.")
                    row["judge_correct"] = value
                    row["correct"] = value
                    row["metric_method"] = f"{PROMPT_VERSION}_GPT-judged_accuracy_not_authors_hidden_prompt"
                row.pop("gpt_judge_error", None)
            except Exception as exc:
                row["gpt_judge_error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
            # Persist after each judged case so long jobs can be resumed.
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{attempted}] {row.get('case_id') or row.get('id')} {'ok' if 'gpt_judge_error' not in row else 'ERROR'}", flush=True)
            if args.delay > 0:
                time.sleep(args.delay)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {output}; judged={attempted}, errors={errors}. Prompt version: {PROMPT_VERSION}.")
    print("Note: prompts are explicit and reproducible but are not claimed to be byte-for-byte identical to the paper authors' prompt.")


if __name__ == "__main__":
    main()
