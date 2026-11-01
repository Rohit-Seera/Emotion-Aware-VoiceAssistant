"""Unit of Work implementation."""

from __future__ import annotations

from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.session import create_session


class UnitOfWork:
    """Manage database transactions as a single unit."""

    def __init__(
        self,
        session: AsyncSession | None = None,
    ) -> None:
        self.session = session
        self._owns_session = session is None

    async def __aenter__(self) -> UnitOfWork:
        """Start the unit of work."""

        if self.session is None:
            self.session = await create_session()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Commit on success and rollback on failure."""

        if self.session is None:
            return

        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            if self._owns_session:
                await self.session.close()

    async def commit(self) -> None:
        """Commit the current transaction."""

        if self.session is None:
            raise RuntimeError(
                "UnitOfWork has not been started."
            )

        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the current transaction."""

        if self.session is None:
            raise RuntimeError(
                "UnitOfWork has not been started."
            )

        await self.session.rollback()

    async def close(self) -> None:
        """Close the owned database session."""

        if self.session is None:
            return

        if self._owns_session:
            await self.session.close()