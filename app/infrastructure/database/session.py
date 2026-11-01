"""Database session management."""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.infrastructure.database.engine import engine


SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


async def create_session() -> AsyncSession:
    """Create a new database session."""

    return SessionLocal()