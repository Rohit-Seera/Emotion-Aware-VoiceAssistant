"""Database infrastructure public exports.

Exports are lazy to avoid importing asyncpg/creating a DB engine merely by
importing ``app.infrastructure.database``.
"""

__all__ = ["Base", "SessionLocal", "engine", "get_db"]


def __getattr__(name: str):
    if name == "Base":
        from .base import Base
        return Base
    if name == "SessionLocal":
        from .session import SessionLocal
        return SessionLocal
    if name == "engine":
        from .engine import engine
        return engine
    if name == "get_db":
        from .dependencies import get_db
        return get_db
    raise AttributeError(name)
