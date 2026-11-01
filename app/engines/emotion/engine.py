"""Emotion engine."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.runtime.state import RuntimeState


@dataclass(slots=True, frozen=True)
class EmotionResult:
    """Represent a detected emotional state."""

    emotion: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


class KeywordEmotionDetector:
    """Development-only text-based emotion detector."""

    _KEYWORDS: dict[str, tuple[str, ...]] = {
        "sad": (
            "sad", "upset", "cry", "crying",
            "hurt", "lonely", "depressed", "feeling down", "feel low",
            "mood off", "low mood", "udaas", "udas", "dukhi", "dukh",
            "mann kharab", "dil toot", "akela", "akeli", "tanha",
            "उदास", "दुखी", "अकेला", "अकेली", "मन खराब", "दिल टूट",
        ),
        "anxious": (
            "anxious", "anxiety", "worried", "worry",
            "nervous", "scared", "fear", "stress",
            "stressed", "afraid", "ghabrahat", "ghabra raha",
            "ghabra rahi", "tension ho rahi", "dar lag raha",
            "चिंतित", "घबराहट", "तनाव", "डर लग",
        ),
        "angry": (
            "angry", "mad", "furious", "annoyed",
            "hate", "irritated", "annoying", "gussa", "naraz",
            "नाराज़", "गुस्सा",
        ),
        "frustrated": (
            "frustrated", "frustration", "fed up",
            "stuck", "this bug is driving me crazy",
        ),
        "happy": (
            "happy", "great", "awesome", "excited",
            "love", "amazing", "good", "glad", "khush", "khushi",
            "maza aa", "bahut accha", "bahut acha", "बढ़िया", "खुश",
        ),
    }

    _WORD_BOUNDARY = re.compile(r"[^\w]+", re.UNICODE)
    _NEGATIONS = frozenset({"not", "never", "no", "n't", "nahi", "nahin", "mat", "\u0928\u0939\u0940", "\u0928\u0939\u0940\u0902"})
    _EXTRA_KEYWORDS: dict[str, tuple[str, ...]] = {
        "sad": ("\u0909\u0926\u093e\u0938", "\u0926\u0941\u0916\u0940", "\u0926\u0941\u0903\u0916\u0940", "\u092e\u0928 \u0916\u0930\u093e\u092c", "\u0926\u093f\u0932 \u091f\u0942\u091f"),
        "anxious": ("\u091a\u093f\u0902\u0924\u093f\u0924", "\u0918\u092c\u0930\u093e\u0939\u091f", "\u0924\u0928\u093e\u0935", "\u0921\u0930 \u0932\u0917"),
        "angry": ("\u0928\u093e\u0930\u093e\u091c", "\u0917\u0941\u0938\u094d\u0938\u093e"),
        "happy": ("excellent", "wonderful", "perfect", "very good", "very well", "badiya", "\u092c\u0922\u093c\u093f\u092f\u093e", "\u0916\u0941\u0936"),
    }

    @classmethod
    def _matches_keyword(cls, text: str, keyword: str) -> bool:
        tokens = cls._WORD_BOUNDARY.sub(" ", text.casefold()).split()
        keyword_tokens = cls._WORD_BOUNDARY.sub(" ", keyword.casefold()).split()
        for index in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[index : index + len(keyword_tokens)] != keyword_tokens:
                continue
            preceding = tokens[max(0, index - 3) : index]
            if not any(token in cls._NEGATIONS for token in preceding):
                return True
        return False

    async def detect_emotion(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Detect emotion from transcript keywords."""

        text = unicodedata.normalize("NFKC", state.input or "").casefold()

        keywords_by_emotion = dict(self._KEYWORDS)
        for emotion, extra_keywords in self._EXTRA_KEYWORDS.items():
            keywords_by_emotion[emotion] += extra_keywords

        for emotion, keywords in keywords_by_emotion.items():
            matches = [keyword for keyword in keywords if self._matches_keyword(text, keyword)]
            if matches:
                return {
                    "emotion": emotion,
                    "confidence": 0.90,
                    "source": "demo_text_proxy",
                    "matched_keyword": matches[0],
                }

        return {
            "emotion": "neutral",
            "confidence": 0.50,
            "source": "demo_text_proxy",
        }


class EmotionEngine:
    """Detect and normalize the user's current emotion."""

    def __init__(
        self,
        detector: Any | None = None,
    ) -> None:
        self._detector = detector

    @property
    def detector(self) -> Any | None:
        """Return the configured emotion detector."""

        return self._detector

    async def detect_emotion(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Detect emotion using the standard detector path."""

        if self._detector is None:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
            }

        result = await self._detector.detect_emotion(state)

        if not isinstance(result, dict):
            raise TypeError(
                "Emotion detector must return a dictionary."
            )

        return result

    async def detect_emotion_with_context(
        self,
        state: RuntimeState,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Forward runtime metadata to context-aware detectors.

        This allows HybridEmotionDetector to access audio_bytes
        and combine acoustic emotion with lexical emotion cues.
        """

        if self._detector is None:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "source": "no_detector",
            }

        context_method = getattr(
            self._detector,
            "detect_emotion_with_context",
            None,
        )

        if callable(context_method):
            result = await context_method(
                state,
                metadata,
            )
        else:
            # Preserve compatibility with text-only detectors.
            result = await self._detector.detect_emotion(
                state,
            )

        if not isinstance(result, dict):
            raise TypeError(
                "Emotion detector must return a dictionary."
            )

        return result

    @staticmethod
    def normalize(
        result: EmotionResult,
    ) -> dict[str, Any]:
        """Convert an EmotionResult into runtime context."""

        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            **result.metadata,
        }
