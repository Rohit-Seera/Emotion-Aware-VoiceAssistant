"""Vector store infrastructure with lazy public exports."""

__all__ = ["CollectionManager", "collections", "qdrant_client"]


def __getattr__(name: str):
    if name == "qdrant_client":
        from .client import qdrant_client
        return qdrant_client
    if name in {"CollectionManager", "collections"}:
        from .collections import CollectionManager, collections
        return {"CollectionManager": CollectionManager, "collections": collections}[name]
    raise AttributeError(name)
