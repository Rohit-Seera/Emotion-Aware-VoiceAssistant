"""Route runtime execution to an AI provider and model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.settings import settings
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Selected model configuration for an execution."""

    provider: str
    model: str
    temperature: float
    max_tokens: int


class ModelRouterNode:
    """Select the provider and model for the current execution."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def route(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> ModelRoute:
        """Resolve the model configuration."""

        provider = (
            self._provider
            or context.metadata.get("provider")
            or settings.ai.default_provider
        )

        model = (
            self._model
            or context.metadata.get("model")
            or ""
        )

        temperature = (
            self._temperature
            if self._temperature is not None
            else context.metadata.get(
                "temperature",
                0.7,
            )
        )

        max_tokens = (
            self._max_tokens
            if self._max_tokens is not None
            else context.metadata.get(
                "max_tokens",
                4096,
            )
        )

        return ModelRoute(
            provider=str(provider),
            model=str(model),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Resolve and store the model route."""

        route = self.route(
            state,
            context,
        )

        context.metadata["model_route"] = {
            "provider": route.provider,
            "model": route.model,
            "temperature": route.temperature,
            "max_tokens": route.max_tokens,
        }

        return state