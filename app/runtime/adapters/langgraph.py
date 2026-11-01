"""LangGraph adapter for the runtime."""

from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.runtime.adapters.state_adapter import RuntimeStateAdapter
from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


class LangGraphAdapter:
    """Adapt a compiled LangGraph graph to the runtime."""

    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        self._graph = graph

    @property
    def graph(self) -> CompiledStateGraph:
        """Return the wrapped compiled graph."""

        return self._graph

    async def invoke(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Invoke the LangGraph graph and return a runtime result."""

        graph_state = RuntimeStateAdapter.to_dict(state)

        result = await self._graph.ainvoke(
            graph_state,
            config=self._build_config(context),
        )

        final_state = RuntimeStateAdapter.from_dict(
            dict(result),
        )

        return RuntimeResult(
            request_id=final_state.request_id,
            workflow=final_state.workflow,
            status=final_state.status,
            output=final_state.output,
            error=final_state.error,
            metadata=final_state.metadata,
            token_usage=final_state.token_usage,
        )

    async def stream(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ):
        """Stream updates from the LangGraph graph."""

        graph_state = RuntimeStateAdapter.to_dict(state)

        async for update in self._graph.astream(
            graph_state,
            config=self._build_config(context),
        ):
            yield update

    @staticmethod
    def _build_config(
        context: RuntimeContext,
    ) -> dict[str, Any]:
        """Build LangGraph execution configuration."""

        return {
            "configurable": {
                "thread_id": str(
                    context.request_id,
                ),
            },
        }