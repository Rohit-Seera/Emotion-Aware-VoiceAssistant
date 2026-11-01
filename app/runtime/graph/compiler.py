"""LangGraph compiler."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from app.runtime.graph.builder import RuntimeGraphBuilder


class RuntimeGraphCompiler:
    """Compile runtime graph definitions."""

    def compile(
        self,
        builder: RuntimeGraphBuilder,
    ) -> CompiledStateGraph:
        """Compile a runtime graph."""

        if not isinstance(
            builder,
            RuntimeGraphBuilder,
        ):
            raise TypeError(
                "builder must be a RuntimeGraphBuilder."
            )

        return builder.build().compile()