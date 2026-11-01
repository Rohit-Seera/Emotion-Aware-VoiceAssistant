"""Deterministic character directives for the frontend animation layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpressionDirective:
    """A constrained visual response, never arbitrary model-provided data."""

    emotion: str
    expression: str
    gesture: str
    gaze: str
    lip_sync: str = "speech"
    voice_style: str = "warm"


class ExpressionPlanner:
    """Map conversation context to a small safe animation vocabulary."""

    _EMPATHETIC_WORDS = ("sorry", "here for you", "that sounds hard")
    _EXCITED_WORDS = ("congratulations", "amazing", "excited", "great news")

    def plan(
        self,
        *,
        user_emotion: str,
        response_text: str,
    ) -> ExpressionDirective:
        """Produce frontend-ready facial and body animation cues."""

        text = response_text.casefold()

        if user_emotion in {"sad", "anxious", "upset", "fearful", "frustrated", "angry"}:
            return ExpressionDirective(
                emotion="empathetic",
                expression="soft_concern",
                gesture="reassuring_nod",
                gaze="gentle_eye_contact",
                voice_style="empathetic",
            )

        if any(word in text for word in self._EMPATHETIC_WORDS):
            return ExpressionDirective(
                emotion="empathetic",
                expression="soft_concern",
                gesture="reassuring_nod",
                gaze="gentle_eye_contact",
                voice_style="empathetic",
            )

        if any(word in text for word in self._EXCITED_WORDS):
            return ExpressionDirective(
                emotion="excited",
                expression="bright_smile",
                gesture="small_wave",
                gaze="direct_eye_contact",
                voice_style="excited",
            )

        return ExpressionDirective(
            emotion="warm",
            expression="gentle_smile",
            gesture="idle_breath",
            gaze="direct_eye_contact",
            voice_style="warm",
        )
