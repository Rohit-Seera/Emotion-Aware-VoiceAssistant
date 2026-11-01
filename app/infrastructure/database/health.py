"""Database health check without eager PostgreSQL dependency imports."""


async def check_database_health() -> bool:
    """Return DB health; false when PostgreSQL support is unavailable."""

    try:
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError
        from app.infrastructure.database.session import SessionLocal
    except ImportError:
        return False

    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
