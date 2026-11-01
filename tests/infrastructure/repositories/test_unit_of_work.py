"""Tests for the Unit of Work."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.repositories.unit_of_work import UnitOfWork


@pytest.fixture
def session() -> MagicMock:
    """Return a mocked async database session."""

    session = MagicMock()

    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    return session


@pytest.mark.asyncio
async def test_uow_stores_provided_session(
    session: MagicMock,
) -> None:
    """UnitOfWork should store a provided session."""

    uow = UnitOfWork(session)

    assert uow.session is session
    assert uow._owns_session is False


@pytest.mark.asyncio
async def test_uow_creates_session_on_enter() -> None:
    """UnitOfWork should create a session when none is provided."""

    session = MagicMock()

    # The UnitOfWork awaits these methods during __aexit__.
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    with patch(
        "app.infrastructure.repositories.unit_of_work.create_session",
        new=AsyncMock(return_value=session),
    ) as create_session:
        async with UnitOfWork() as uow:
            assert uow.session is session
            assert uow._owns_session is True

    create_session.assert_awaited_once()
    session.commit.assert_awaited_once()
    session.close.assert_awaited_once()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_enter_with_provided_session_does_not_create_session(
    session: MagicMock,
) -> None:
    """Provided sessions should not trigger session creation."""

    with patch(
        "app.infrastructure.repositories.unit_of_work.create_session",
        new=AsyncMock(),
    ) as create_session:
        async with UnitOfWork(session) as uow:
            assert uow.session is session

    create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_manager_commits_on_success(
    session: MagicMock,
) -> None:
    """Successful context manager execution should commit."""

    async with UnitOfWork(session):
        pass

    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()
    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_manager_rolls_back_on_exception(
    session: MagicMock,
) -> None:
    """Failed context manager execution should rollback."""

    with pytest.raises(ValueError):
        async with UnitOfWork(session):
            raise ValueError("test error")

    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_session_is_closed_after_success() -> None:
    """Owned sessions should be closed after successful execution."""

    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    with patch(
        "app.infrastructure.repositories.unit_of_work.create_session",
        new=AsyncMock(return_value=session),
    ):
        async with UnitOfWork():
            pass

    session.commit.assert_awaited_once()
    session.close.assert_awaited_once()
    session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_session_is_closed_after_exception() -> None:
    """Owned sessions should be closed after failed execution."""

    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    with patch(
        "app.infrastructure.repositories.unit_of_work.create_session",
        new=AsyncMock(return_value=session),
    ):
        with pytest.raises(RuntimeError):
            async with UnitOfWork():
                raise RuntimeError("test error")

    session.rollback.assert_awaited_once()
    session.close.assert_awaited_once()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit(
    session: MagicMock,
) -> None:
    """commit should commit the current transaction."""

    uow = UnitOfWork(session)

    await uow.commit()

    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_rollback(
    session: MagicMock,
) -> None:
    """rollback should rollback the current transaction."""

    uow = UnitOfWork(session)

    await uow.rollback()

    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_owned_session(
    session: MagicMock,
) -> None:
    """close should close an owned session."""

    uow = UnitOfWork()
    uow.session = session
    uow._owns_session = True

    await uow.close()

    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_does_not_close_external_session(
    session: MagicMock,
) -> None:
    """close should not close an externally provided session."""

    uow = UnitOfWork(session)

    await uow.close()

    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_requires_started_uow() -> None:
    """commit should fail when no session exists."""

    uow = UnitOfWork()

    with pytest.raises(
        RuntimeError,
        match="UnitOfWork has not been started",
    ):
        await uow.commit()


@pytest.mark.asyncio
async def test_rollback_requires_started_uow() -> None:
    """rollback should fail when no session exists."""

    uow = UnitOfWork()

    with pytest.raises(
        RuntimeError,
        match="UnitOfWork has not been started",
    ):
        await uow.rollback()