"""Detect the user's current emotional state."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class EmotionProviderProtocol(Protocol):
    """Contract for detecting user emotion."""

    async def detect_emotion(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Detect emotion from the current runtime state."""
        ...


class DetectEmotionNode:
    """Detect and attach the current user emotion."""

    def __init__(
        self,
        provider: EmotionProviderProtocol | None = None,
    ) -> None:
        self._provider = provider

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Detect emotion and store it in runtime metadata."""

        provider = self._provider

        # Resolve provider from runtime services if not injected.
        if provider is None:
            if not context.has_service("emotion_provider"):
                raise RuntimeError(
                    "Emotion provider has not been configured."
                )

            provider = context.get_service(
                "emotion_provider",
            )

        # Check whether the method genuinely exists on the object
        # or its class. This avoids dynamically generated MagicMock
        # attributes being mistaken for real methods.
        declared_method = inspect.getattr_static(
            provider,
            "detect_emotion_with_context",
            None,
        )

        if declared_method is not None:
            context_aware = getattr(
                provider,
                "detect_emotion_with_context",
            )

            if not callable(context_aware):
                raise TypeError(
                    "detect_emotion_with_context must be callable."
                )

            emotion = await context_aware(
                state,
                context.metadata,
            )
        else:
            # Backward compatibility with text-only emotion providers.
            emotion = await provider.detect_emotion(
                state,
            )

        if not isinstance(emotion, dict):
            raise TypeError(
                "Emotion provider must return a dictionary."
            )

        # Store the latest emotion, replacing stale context.
        context.metadata["emotion"] = emotion

        return state