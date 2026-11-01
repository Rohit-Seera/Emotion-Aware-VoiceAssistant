"""Run the Emotion Aware Voice Assistant SOVA-inspired benchmark against configured LLM providers."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.core.enums import Workflow
from app.core.settings import settings
from app.engines.character.engine import CharacterEngine
from app.engines.emotion.engine import EmotionEngine
from app.engines.personality.engine import PersonalityEngine
from app.engines.relationship.engine import RelationshipEngine
from app.providers.manager.provider_manager import ProviderManager
from app.runtime.companion_workflow import CompanionWorkflow
from app.runtime.executor import RuntimeExecutor
from app.runtime.runtime import Runtime

from benchmark.metrics import (
    answer_rate,
    conditioned_accuracy,
    exact_or_alias_match,
    mean,
    overall_accuracy,
    wer,
)

ROOT = Path(__file__).resolve().parent


def extract_label(output: str, labels: list[str]) -> str | None:
    """Extract a benchmark label robustly from a forced single-label response."""
    text = output.strip().lower()
    # Prefer an exact whole-output match first.
    for label in labels:
        if text == label.strip().lower():
            return label
    # Otherwise inspect individual lines/tokens, preferring the earliest explicit label.
    for line in text.splitlines():
        candidate = re.sub(r"[^a-z0-9_ -]", " ", line).strip()
        if not candidate:
            continue
        for label in labels:
            lab = label.strip().lower()
            if re.search(r"(?<!\w)" + re.escape(lab) + r"(?!\w)", candidate):
                return label
    return None


@dataclass(slots=True)
class CaseResult:
    model_name: str
    provider: str
    model: str
    case_id: str
    task: str
    prompt: str
    output: str = ""
    success: bool = False
    answered: bool = False
    correct: bool = False
    predicted_label: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    consistency_wer: float | None = None
    semantic_score: float | None = None
    error: str | None = None


class BenchmarkRunner:
    def __init__(self, *, config: dict[str, Any]) -> None:
        self.config = config
        self.provider_manager = ProviderManager()
        workflow = CompanionWorkflow(
            provider_manager=self.provider_manager,
            character_engine=CharacterEngine(),
            emotion_engine=EmotionEngine(),
            personality_engine=PersonalityEngine(),
            relationship_engine=RelationshipEngine(),
        )
        self.runtime = Runtime(
            RuntimeExecutor(
                workflow,
                timeout=float(config.get("timeout_seconds", 75)),
            )
        )

    async def close(self) -> None:
        await self.provider_manager.close()

    async def run_case(
        self,
        item: dict[str, Any],
        model_entry: dict[str, str],
    ) -> CaseResult:
        provider = model_entry["provider"]
        model = model_entry.get("model") or getattr(
            getattr(settings.ai, provider, None), "model", ""
        )
        name = model_entry.get("name", f"{provider}:{model}")
        started = time.perf_counter()
        try:
            task_name = str(item.get("task", "")).lower()
            model_input = str(item["prompt"])
            if task_name == "emotion":
                labels = [str(x) for x in item.get("labels", [])]
                label_text = ", ".join(labels)
                model_input = (
                    "Emotion classification benchmark.\n"
                    f"User message: {model_input}\n\n"
                    f"Choose exactly ONE label from: {label_text}.\n"
                    "Return ONLY the single label, with no explanation, punctuation, or extra words."
                )
            elif task_name in {"age", "gender"}:
                labels = [str(x) for x in item.get("labels", [])]
                label_text = ", ".join(labels)
                model_input = (
                    f"{task_name.title()} classification benchmark.\n"
                    f"User message: {model_input}\n\n"
                    f"Choose exactly ONE label from: {label_text}.\n"
                    "Return ONLY the single label, with no explanation, punctuation, or extra words."
                )

            if task_name in {"emotion", "age", "gender"}:
                # Classification tasks are evaluated at the model level, without
                # the companion's conversational system prompt changing the output.
                response = await self.provider_manager.generate(
                    [{"role": "user", "content": model_input}],
                    provider=provider,
                    model=model,
                    temperature=self.config.get("temperature", 0.2),
                    max_tokens=self.config.get("max_tokens", 512),
                    fallback=False,
                )
                output = response.content or ""
                tokens = response.usage or type("U", (), {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
                })()
                success = True
                result_success = True
            else:
                result = await self.runtime.execute(
                    workflow=Workflow.VOICE,
                    input=model_input,
                    metadata={
                        "provider": provider,
                        "model": model,
                        "temperature": self.config.get("temperature", 0.2),
                        "max_tokens": self.config.get("max_tokens", 512),
                    },
                )
                output = result.output or ""
                tokens = result.token_usage
                success = result.success
                result_success = result.success
            answered = bool(output.strip())
            correct = False
            predicted_label = None
            task_name = str(item.get("task", "")).lower()
            answers = item.get("answers")

            # TriviaQA converter may store reference answers under "expected".
            if not answers:
                answers = item.get("expected", [])

            if isinstance(answers, str):
                answers = [answers]

            if isinstance(answers, list) and answers and task_name in {"knowledge", "instruction"}:
                correct = exact_or_alias_match(
                    output,
                    [str(a) for a in answers],
                )
            labels = item.get("labels")
            expected_label = item.get("expected_label")
            aliases = item.get("aliases", [])
            if isinstance(labels, list) and isinstance(expected_label, str):
                allowed = [str(label) for label in labels]
                predicted_label = extract_label(output, allowed)
                expected_aliases = [expected_label] + [str(a) for a in aliases]
                correct = bool(
                    predicted_label is not None
                    and predicted_label.lower() in {a.lower() for a in expected_aliases}
                )
            consistency = None
            reference = item.get("reference_text")
            generated_transcript = item.get("generated_transcript")
            if isinstance(reference, str) and isinstance(generated_transcript, str):
                consistency = wer(reference, generated_transcript)
            return CaseResult(
                model_name=name,
                provider=provider,
                model=model,
                case_id=str(item.get("id") or item.get("case_id") or item.get("caseId") or ""),
                task=str(item.get("task", "open_ended")),
                prompt=str(item["prompt"]),
                output=output,
                success=result_success,
                answered=answered,
                correct=correct if str(item.get("task", "")).lower() in {"knowledge", "emotion", "age", "gender", "instruction"} else False,
                predicted_label=predicted_label,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                prompt_tokens=tokens.prompt_tokens,
                completion_tokens=tokens.completion_tokens,
                total_tokens=tokens.total_tokens,
                consistency_wer=consistency,
            )
        except Exception as exc:  # benchmark runner should preserve failures
            return CaseResult(
                model_name=name,
                provider=provider,
                model=model,
                case_id=str(item.get("id") or item.get("case_id") or item.get("caseId") or ""),
                task=str(item.get("task", "open_ended")),
                prompt=str(item["prompt"]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def summarize(results: list[CaseResult]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[CaseResult]] = {}
        for result in results:
            grouped.setdefault(
                (result.model_name, result.provider, result.model), []
            ).append(result)

        summaries: list[dict[str, Any]] = []
        for key, items in grouped.items():
            row: dict[str, Any] = {
                "model_name": key[0],
                "provider": key[1],
                "model": key[2],
                "mean_latency_ms": mean(r.latency_ms for r in items),
                "mean_prompt_tokens": mean(r.prompt_tokens for r in items),
                "mean_completion_tokens": mean(r.completion_tokens for r in items),
                "mean_total_tokens": mean(r.total_tokens for r in items),
                "cases": len(items),
                "failed_cases": sum(not r.success for r in items),
            }

            for task in ("knowledge", "emotion", "age", "gender", "instruction"):
                subset = [r for r in items if r.task == task]
                answered = sum(r.answered for r in subset)
                correct = sum(r.correct for r in subset)
                total = len(subset)
                prefix = task
                row[f"{prefix}_accuracy"] = overall_accuracy(correct, total)
                row[f"{prefix}_answer_rate"] = answer_rate(answered, total)
                row[f"{prefix}_conditioned_accuracy"] = conditioned_accuracy(correct, answered)

            consistency = [
                r.consistency_wer for r in items if r.consistency_wer is not None
            ]
            row["consistency_wer"] = mean(consistency) if consistency else None
            summaries.append(row)
        return summaries



async def async_main(
    dataset_path: Path,
    models_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Optional smoke-test limit. The slice happens before execution so
    # --limit N runs only the first N dataset cases.
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        dataset = dataset[:limit]
        print(f"LIMIT: running first {len(dataset)} dataset cases")

    model_entries = json.loads(models_path.read_text(encoding="utf-8"))["models"]

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    runner = BenchmarkRunner(config=config)

    results_path = output_dir / "results.json"
    existing: list[CaseResult] = []
    if results_path.exists():
        try:
            raw = json.loads(results_path.read_text(encoding="utf-8"))
            existing = [CaseResult(**row) for row in raw]
            print(f"RESUME: loaded {len(existing)} saved case results")
        except Exception as exc:
            print(f"WARNING: could not load existing results.json: {exc}")

    # Successful cases are immutable for a reproducible run. Failed/rate-limited
    # cases are retried so temporary provider limits do not become benchmark data.
    successful_keys = {
        (r.provider, r.model, r.case_id) for r in existing if r.success
    }
    results: list[CaseResult] = [r for r in existing if r.success]

    inter_case_delay = float(config.get("inter_case_delay_seconds", 1.0))
    max_retries = int(config.get("rate_limit_max_retries", 5))

    def is_rate_limit_error(result: CaseResult) -> bool:
        text = (result.error or "").lower()
        return "429" in text or "too many requests" in text or "rate limit" in text

    def save_results() -> None:
        results_path.write_text(
            json.dumps([asdict(r) for r in results], indent=2),
            encoding="utf-8",
        )
        summaries = runner.summarize(results)
        (output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2),
            encoding="utf-8",
        )

    try:
        available = runner.provider_manager.providers
        for model_entry in model_entries:
            provider = model_entry["provider"]
            if provider not in available:
                print(f"SKIP {provider}: provider is not configured/available")
                continue
            model_entry = dict(model_entry)
            model_entry.setdefault(
                "model",
                getattr(getattr(settings.ai, provider), "model", ""),
            )

            for idx, item in enumerate(dataset, start=1):
                model = model_entry.get("model", "")
                case_id = str(item.get("id") or item.get("case_id") or item.get("caseId"))
                if not case_id:
                    raise KeyError("Dataset item is missing both 'id' and 'case_id'")
                key = (provider, model, case_id)
                if key in successful_keys:
                    continue

                result: CaseResult | None = None
                for attempt in range(max_retries + 1):
                    result = await runner.run_case(item, model_entry)
                    if result.success or not is_rate_limit_error(result) or attempt >= max_retries:
                        break
                    wait_s = min(60.0, 10.0 * (2 ** attempt))
                    print(
                        f"RATE LIMIT {result.model_name} {result.case_id}: "
                        f"waiting {wait_s:.0f}s before retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(wait_s)

                assert result is not None
                print(
                    f"{result.model_name:<32} {result.case_id:<16} "
                    f"{result.latency_ms:8.1f} ms ok={result.success}"
                )

                if result.success:
                    results.append(result)
                    successful_keys.add(key)
                    save_results()
                else:
                    # Keep failures out of the resumable success set. A later run
                    # will retry them instead of treating provider throttling as data.
                    print(f"FAILED {result.case_id}: {result.error}")
                    if not is_rate_limit_error(result):
                        results.append(result)
                        save_results()

                await asyncio.sleep(inter_case_delay)
    finally:
        await runner.close()

    save_results()
    print(f"Wrote {len(results)} successful/retained case results to {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "starter.json",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "models.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N dataset cases",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.dataset, args.models, args.output, args.limit))


if __name__ == "__main__":
    main()
