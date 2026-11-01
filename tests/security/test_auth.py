"""Tests for FastAPI authentication dependencies."""

from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.security.auth import (
    extract_bearer_token,
    get_current_claims,
    get_current_subject,
    get_current_token,
)


def make_credentials(
    token: str,
    scheme: str = "Bearer",
) -> HTTPAuthorizationCredentials:
    """Create HTTP authorization credentials for testing."""

    return HTTPAuthorizationCredentials(
        scheme=scheme,
        credentials=token,
    )


@pytest.mark.asyncio
async def test_extracts_bearer_token() -> None:
    """A valid Bearer credential should return its token."""

    credentials = make_credentials("test-token")

    assert extract_bearer_token(credentials) == "test-token"


def test_missing_credentials_are_rejected() -> None:
    """Missing authorization credentials must be rejected."""

    with pytest.raises(HTTPException) as error:
        extract_bearer_token(None)

    assert error.value.status_code == 401
    assert error.value.detail == "Authentication required."
    assert error.value.headers["WWW-Authenticate"] == "Bearer"


def test_non_bearer_scheme_is_rejected() -> None:
    """Only Bearer authentication should be accepted."""

    credentials = make_credentials(
        "test-token",
        scheme="Basic",
    )

    with pytest.raises(HTTPException) as error:
        extract_bearer_token(credentials)

    assert error.value.status_code == 401


def test_empty_token_is_rejected() -> None:
    """An empty Bearer credential must be rejected."""

    credentials = make_credentials("   ")

    with pytest.raises(HTTPException):
        extract_bearer_token(credentials)


@pytest.mark.asyncio
async def test_valid_access_token_returns_claims() -> None:
    """A valid access token should produce trusted claims."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_access_token(
        "user-123",
        extra_claims={
            "role": "user",
        },
    )

    credentials = make_credentials(token)

    claims = await get_current_claims(credentials)

    assert claims["sub"] == "user-123"
    assert claims["type"] == "access"
    assert claims["role"] == "user"


@pytest.mark.asyncio
async def test_valid_access_token_returns_subject() -> None:
    """A valid access token should return its subject."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_access_token("user-123")

    credentials = make_credentials(token)

    assert (
        await get_current_subject(credentials)
        == "user-123"
    )


@pytest.mark.asyncio
async def test_refresh_token_is_rejected() -> None:
    """Refresh tokens must never authenticate API requests."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_refresh_token("user-123")

    credentials = make_credentials(token)

    with pytest.raises(HTTPException) as error:
        await get_current_claims(credentials)

    assert error.value.status_code == 401
    assert error.value.detail == "Authentication required."


@pytest.mark.asyncio
async def test_tampered_token_is_rejected() -> None:
    """Tampered access tokens must be rejected."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_access_token("user-123")

    parts = token.split(".")
    parts[1] = parts[1][::-1]

    credentials = make_credentials(
        ".".join(parts),
    )

    with pytest.raises(HTTPException) as error:
        await get_current_claims(credentials)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_does_not_leak_internal_error() -> None:
    """JWT internals must never be exposed to clients."""

    credentials = make_credentials(
        "this-is-not-a-valid-jwt",
    )

    with pytest.raises(HTTPException) as error:
        await get_current_claims(credentials)

    assert error.value.status_code == 401
    assert error.value.detail == "Authentication required."


@pytest.mark.asyncio
async def test_wrong_secret_is_rejected() -> None:
    """A token signed with another secret must fail."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_access_token("user-123")

    with patch(
        "app.security.jwt.settings.security.jwt_secret_key",
        "a" * 64,
    ):
        credentials = make_credentials(token)

        with pytest.raises(HTTPException) as error:
            await get_current_claims(credentials)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_token_returns_raw_token() -> None:
    """The token dependency should return the raw credential."""

    credentials = make_credentials("raw-token")

    assert (
        await get_current_token(credentials)
        == "raw-token"
    )
    