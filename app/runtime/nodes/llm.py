"""Execute the selected language model."""

from __future__ import annotations

from typing import Any, Protocol

from app.providers.base.types import ProviderResponse
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class LLMProviderProtocol(Protocol):
    """Contract for the provider manager used by the LLM node."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a model response."""
        ...


class LLMNode:
    """Generate a response using the selected model."""

    def __init__(
        self,
        provider_manager: LLMProviderProtocol | None = None,
    ) -> None:
        self._provider_manager = provider_manager

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Generate the assistant response."""

        manager = self._provider_manager

        if manager is None:
            if not context.has_service("provider_manager"):
                raise RuntimeError(
                    "Provider manager has not been configured."
                )

            manager = context.get_service(
                "provider_manager",
            )

        prompt = context.metadata.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(
                "Runtime prompt has not been configured."
            )

        route = context.metadata.get(
            "model_route",
            {},
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = await manager.generate(
            messages,
            provider=route.get("provider"),
            model=route.get("model"),
            temperature=route.get("temperature"),
            max_tokens=route.get("max_tokens"),
        )

        state.output = response.content

        if response.usage is not None:
            state.token_usage.prompt_tokens = (
                response.usage.prompt_tokens
            )
            state.token_usage.completion_tokens = (
                response.usage.completion_tokens
            )
            state.token_usage.total_tokens = (
                response.usage.total_tokens
            )

        context.metadata["provider_response"] = response
        context.metadata["llm_provider"] = response.provider
        context.metadata["llm_model"] = response.model

        return state
