"""Metric implementations for a SOVA-Bench-compatible evaluation.

The paper gives metric names but does not publish all judge prompts or
implementation details. These functions implement standard, explicitly
specified metric definitions; this module does not claim to reproduce the
authors' undisclosed evaluation prompts.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WERStats:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer_fraction(self) -> float | None:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else None
        return self.errors / self.reference_words

    @property
    def wer_percent(self) -> float | None:
        fraction = self.wer_fraction
        return None if fraction is None else fraction * 100.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            errors=self.errors,
            wer_fraction=self.wer_fraction,
            wer_percent=self.wer_percent,
        )
        return result


def normalize_words(text: str) -> list[str]:
    """Normalize text consistently for WER: NFKC, casefold, punctuation off."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    # Preserve apostrophes inside contractions but treat all other punctuation
    # as word boundaries. This is documented because text normalization affects WER.
    text = text.replace("’", "'")
    text = re.sub(r"[^\w']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return text.split()


def wer_stats(reference: str, hypothesis: str) -> WERStats:
    """Return S/D/I counts from a minimum-edit-distance alignment.

    Ties are resolved substitution, deletion, insertion, then match, in that
    order. WER itself depends only on the total edit count, not this tie-break.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] != hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        else:  # Defensive fallback for empty-reference / empty-hypothesis edges.
            if i > 0:
                deletions += 1
                i -= 1
            elif j > 0:
                insertions += 1
                j -= 1

    return WERStats(substitutions, deletions, insertions, n)


def wer(reference: str, hypothesis: str, *, percent: bool = True) -> float | None:
    """Standard WER = (S + D + I) / N, returned as percent by default."""
    stats = wer_stats(reference, hypothesis)
    return stats.wer_percent if percent else stats.wer_fraction


def corpus_wer(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Compute micro/corpus WER by summing edit counts before dividing."""
    totals = WERStats(0, 0, 0, 0)
    macro: list[float] = []
    n_valid = 0
    for reference, hypothesis in pairs:
        item = wer_stats(reference, hypothesis)
        totals = WERStats(
            totals.substitutions + item.substitutions,
            totals.deletions + item.deletions,
            totals.insertions + item.insertions,
            totals.reference_words + item.reference_words,
        )
        if item.wer_percent is not None:
            macro.append(item.wer_percent)
            n_valid += 1
    result = totals.to_dict()
    result.update(
        utterances=n_valid,
        macro_wer_percent=(sum(macro) / len(macro)) if macro else None,
        aggregation="corpus/micro WER = sum(S+D+I) / sum(reference words); macro WER is also reported separately",
    )
    return result


def percent(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def accuracy(correct: int, total: int) -> float | None:
    """Classification/QA accuracy as a percentage of all target cases."""
    return percent(correct, total)


def answer_rate(answered: int, total: int) -> float | None:
    """AR = answered cases / all target cases * 100."""
    return percent(answered, total)


def conditioned_accuracy(correct: int, answered: int) -> float | None:
    """CA = correct answers / answered cases * 100."""
    return percent(correct, answered)


def exact_label_match(prediction: str | None, reference: str | None) -> bool | None:
    """Case-insensitive whole-label match; None if either value is absent."""
    if prediction is None or reference is None:
        return None
    p = unicodedata.normalize("NFKC", str(prediction)).strip().casefold()
    r = unicodedata.normalize("NFKC", str(reference)).strip().casefold()
    if not p or not r:
        return None
    return p == r


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    clean = [float(x) for x in values if x is not None]
    return sum(clean) / len(clean) if clean else None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
