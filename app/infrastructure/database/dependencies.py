"""Database dependencies."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.session import SessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session."""

    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()