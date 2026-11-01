"""Hybrid acoustic + lexical emotion detection for Emotion Aware Voice Assistant."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.runtime.state import RuntimeState
from app.engines.emotion.acoustic import Wav2Vec2AcousticEmotionDetector
from app.engines.emotion.engine import KeywordEmotionDetector

logger = logging.getLogger(__name__)


class HybridEmotionDetector:
    """Fuse acoustic speech emotion with transcript emotion cues.

    The acoustic model is the primary path for audio turns. The lexical signal
    remains useful for intent-heavy cases (for example words such as
    "frustrated" or "worried") and provides a graceful fallback when the local
    model is unavailable.
    """

    def __init__(
        self,
        *,
        acoustic: Wav2Vec2AcousticEmotionDetector,
        lexical: KeywordEmotionDetector | None = None,
        acoustic_timeout: float = 5.0,
    ) -> None:
        if acoustic_timeout <= 0:
            raise ValueError("acoustic_timeout must be positive.")
        self._acoustic = acoustic
        self._lexical = lexical or KeywordEmotionDetector()
        self._acoustic_timeout = acoustic_timeout

    @property
    def acoustic(self) -> Wav2Vec2AcousticEmotionDetector:
        """Return the acoustic detector."""

        return self._acoustic

    async def detect_emotion(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Text-only compatibility path."""

        lexical = await self._lexical.detect_emotion(state)
        return {
            **lexical,
            "source": "text_proxy_fallback",
        }

    async def detect_emotion_with_context(
        self,
        state: RuntimeState,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect emotion using any audio attached to the current turn."""

        lexical = await self._lexical.detect_emotion(state)
        audio = metadata.get("audio_bytes")

        acoustic: dict[str, Any] | None = None
        if isinstance(audio, bytes):
            try:
                acoustic = await asyncio.wait_for(
                    self._acoustic.detect(audio),
                    timeout=self._acoustic_timeout,
                )
            except TimeoutError:
                logger.warning(
                    "Acoustic emotion detection exceeded %.1f seconds; "
                    "continuing with text fallback.",
                    self._acoustic_timeout,
                )

        if acoustic is None:
            return {
                **lexical,
                "source": "text_proxy_fallback",
                "acoustic_available": False,
                "text": lexical,
            }

        acoustic_emotion = str(
            acoustic.get(
                "emotion",
                "neutral",
            )
        )
        acoustic_confidence = float(
            acoustic.get(
                "confidence",
                0.0,
            )
        )

        lexical_emotion = str(
            lexical.get(
                "emotion",
                "neutral",
            )
        )
        lexical_confidence = float(
            lexical.get(
                "confidence",
                0.0,
            )
        )

        # Strong lexical cues are especially useful for classes that the public
        # four-class SUPERB checkpoint does not expose directly, such as
        # "frustrated" and "anxious".
        if (
            lexical_emotion not in {"neutral", ""}
            and lexical_confidence >= 0.85
            and acoustic_emotion == "neutral"
        ):
            final_emotion = lexical_emotion
            final_confidence = lexical_confidence
            source = "hybrid_acoustic_text"
        elif (
            lexical_emotion not in {"neutral", ""}
            and lexical_confidence >= 0.85
            and acoustic_emotion != lexical_emotion
            and acoustic_emotion != "neutral"
        ):
            final_emotion = lexical_emotion
            final_confidence = lexical_confidence
            source = "text_override_acoustic"
        elif acoustic_emotion == lexical_emotion:
            final_emotion = acoustic_emotion
            final_confidence = min(
                0.99,
                max(
                    acoustic_confidence,
                    lexical_confidence,
                )
                + 0.05,
            )
            source = "hybrid_acoustic_text"
        elif acoustic_confidence >= self._acoustic.min_confidence:
            final_emotion = acoustic_emotion
            final_confidence = acoustic_confidence
            source = "acoustic_wav2vec2"
        else:
            final_emotion = lexical_emotion
            final_confidence = lexical_confidence
            source = "hybrid_text_fallback"

        return {
            "emotion": final_emotion,
            "confidence": round(final_confidence, 4),
            "source": source,
            "acoustic_available": True,
            "model": acoustic.get("model"),
            "top_emotions": acoustic.get("top_emotions", []),
            "acoustic": acoustic,
            "text": lexical,
        }
