"""Tests for JWT authentication."""

from datetime import datetime, timezone
from unittest.mock import patch

import jwt
import pytest

from app.security.jwt import (
    JWTConfigurationError,
    JWTService,
    JWTValidationError,
    jwt_service,
)


def test_create_access_token() -> None:
    """Access tokens should contain the required claims."""

    token = jwt_service.create_access_token(
        "user-123",
    )

    payload = jwt_service.decode_access_token(token)

    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"
    assert payload["iss"] == "emotion-aware-voice-assistant"
    assert payload["aud"] == "emotion-aware-voice-assistant-client"
    assert isinstance(payload["jti"], str)


def test_create_refresh_token() -> None:
    """Refresh tokens should have the refresh token type."""

    token = jwt_service.create_refresh_token(
        "user-123",
    )

    payload = jwt_service.decode_refresh_token(token)

    assert payload["sub"] == "user-123"
    assert payload["type"] == "refresh"


def test_access_and_refresh_tokens_are_different() -> None:
    """Access and refresh tokens must not be interchangeable."""

    access = jwt_service.create_access_token("user-123")
    refresh = jwt_service.create_refresh_token("user-123")

    assert access != refresh


def test_access_token_cannot_be_used_as_refresh_token() -> None:
    """An access token must never pass refresh validation."""

    token = jwt_service.create_access_token("user-123")

    with pytest.raises(JWTValidationError):
        jwt_service.decode_refresh_token(token)


def test_refresh_token_cannot_be_used_as_access_token() -> None:
    """A refresh token must never pass access validation."""

    token = jwt_service.create_refresh_token("user-123")

    with pytest.raises(JWTValidationError):
        jwt_service.decode_access_token(token)


def test_tampered_token_is_rejected() -> None:
    """Changing a signed token must invalidate it."""

    token = jwt_service.create_access_token("user-123")

    parts = token.split(".")
    parts[1] = parts[1][::-1]
    tampered = ".".join(parts)

    with pytest.raises(JWTValidationError):
        jwt_service.decode_access_token(tampered)


def test_wrong_secret_is_rejected() -> None:
    """Tokens signed with another secret must fail."""

    token = jwt_service.create_access_token("user-123")

    with patch(
        "app.security.jwt.settings.security.jwt_secret_key",
        "a-different-secret-that-is-at-least-32-chars",
    ):
        with pytest.raises(JWTValidationError):
            jwt_service.decode_access_token(token)


def test_expired_token_is_rejected() -> None:
    """Expired JWTs must never be accepted."""

    now = datetime.now(timezone.utc)

    token = jwt.encode(
        {
            "sub": "user-123",
            "type": "access",
            "iss": "emotion-aware-voice-assistant",
            "aud": "emotion-aware-voice-assistant-client",
            "iat": now,
            "nbf": now,
            "exp": now.replace(
                year=now.year - 1,
            ),
            "jti": "expired-token",
        },
        "test-secret-that-is-at-least-32-characters-long",
        algorithm="HS256",
    )

    with patch(
        "app.security.jwt.settings.security.jwt_secret_key",
        "test-secret-that-is-at-least-32-characters-long",
    ):
        with pytest.raises(JWTValidationError):
            jwt_service.decode_access_token(token)


def test_invalid_token_is_rejected() -> None:
    """Random strings must not be accepted as JWTs."""

    with pytest.raises(JWTValidationError):
        jwt_service.decode_access_token(
            "not-a-real-jwt",
        )


def test_empty_subject_is_rejected() -> None:
    """Tokens cannot be created without a subject."""

    with pytest.raises(ValueError):
        jwt_service.create_access_token("")


def test_unique_jti_is_generated() -> None:
    """Every issued token should receive a unique token ID."""

    first = jwt_service.decode_access_token(
        jwt_service.create_access_token("user-123"),
    )

    second = jwt_service.decode_access_token(
        jwt_service.create_access_token("user-123"),
    )

    assert first["jti"] != second["jti"]


def test_extra_claims_are_preserved() -> None:
    """Application claims should survive JWT encoding and decoding."""

    token = jwt_service.create_access_token(
        "user-123",
        extra_claims={
            "role": "user",
        },
    )

    payload = jwt_service.decode_access_token(token)

    assert payload["role"] == "user"


def test_algorithm_is_explicitly_allowlisted() -> None:
    """Only the configured safe JWT algorithm should be accepted."""

    assert jwt_service.algorithm == "HS256"


def test_unsafe_algorithm_configuration_is_rejected() -> None:
    """Unsupported algorithms must fail configuration validation."""

    with patch(
        "app.security.jwt.settings.security.jwt_algorithm",
        "none",
    ):
        with pytest.raises(JWTConfigurationError):
            JWTService()


def test_missing_secret_is_rejected() -> None:
    """JWT signing must never operate without a secret."""

    with patch(
        "app.security.jwt.settings.security.jwt_secret_key",
        "",
    ):
        with pytest.raises(JWTConfigurationError):
            JWTService()


def test_short_secret_is_rejected() -> None:
    """Weak JWT secrets must be rejected."""

    with patch(
        "app.security.jwt.settings.security.jwt_secret_key",
        "too-short",
    ):
        with pytest.raises(JWTConfigurationError):
            JWTService()
