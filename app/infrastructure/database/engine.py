"""Database engine."""

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.settings import settings


def create_engine() -> AsyncEngine:
    """Create the database engine."""

    return create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=settings.database.pool_pre_ping,
        pool_recycle=settings.database.pool_recycle,
        future=True,
    )


engine = create_engine()