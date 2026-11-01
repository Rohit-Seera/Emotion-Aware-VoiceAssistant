"""Runtime component registry."""

from __future__ import annotations

from app.runtime.exceptions import RuntimeError
from app.runtime.interfaces import RuntimeNodeProtocol


class RuntimeRegistry:
    """Registry for runtime workflow nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, RuntimeNodeProtocol] = {}

    def register_node(
        self,
        name: str,
        node: RuntimeNodeProtocol,
    ) -> None:
        """Register a runtime node."""

        if not name.strip():
            raise ValueError(
                "Node name cannot be empty."
            )

        if name in self._nodes:
            raise RuntimeError(
                f"Runtime node '{name}' is already registered."
            )

        self._nodes[name] = node

    def get_node(
        self,
        name: str,
    ) -> RuntimeNodeProtocol:
        """Return a registered node."""

        if name not in self._nodes:
            raise RuntimeError(
                f"Runtime node '{name}' is not registered."
            )

        return self._nodes[name]

    def has_node(
        self,
        name: str,
    ) -> bool:
        """Check whether a node is registered."""

        return name in self._nodes

    def unregister_node(
        self,
        name: str,
    ) -> None:
        """Remove a registered node."""

        if name not in self._nodes:
            raise RuntimeError(
                f"Runtime node '{name}' is not registered."
            )

        del self._nodes[name]

    def clear(self) -> None:
        """Remove all registered nodes."""

        self._nodes.clear()

    @property
    def size(self) -> int:
        """Return the number of registered nodes."""

        return len(self._nodes)