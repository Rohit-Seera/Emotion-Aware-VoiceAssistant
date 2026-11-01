"""JWT creation and validation utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from jwt import InvalidTokenError

from app.core.enums import Environment
from app.core.settings import settings


class JWTError(Exception):
    """Base JWT security error."""


class JWTConfigurationError(JWTError):
    """Raised when JWT configuration is unsafe or invalid."""


class JWTValidationError(JWTError):
    """Raised when a JWT cannot be trusted."""


class JWTService:
    """Create and validate access and refresh JWTs."""

    # ------------------------------------------------------------------
    # Algorithm allowlist
    # ------------------------------------------------------------------
    #
    # Never accept an algorithm supplied by an untrusted token.
    # The application configuration determines the only algorithm
    # that can be used.
    #
    _ALLOWED_ALGORITHMS = frozenset(
        {
            "HS256",
        }
    )

    # ------------------------------------------------------------------
    # Token types
    # ------------------------------------------------------------------

    _ACCESS_TOKEN = "access"
    _REFRESH_TOKEN = "refresh"

    def __init__(self) -> None:
        """Initialize the JWT service after validating configuration."""

        self._validate_configuration()

    @property
    def algorithm(self) -> str:
        """Return the configured JWT algorithm."""

        return settings.security.jwt_algorithm

    def _validate_configuration(self) -> None:
        """Validate security-critical JWT configuration."""

        algorithm = settings.security.jwt_algorithm

        if algorithm not in self._ALLOWED_ALGORITHMS:
            raise JWTConfigurationError(
                "Configured JWT algorithm is not allowed."
            )

        secret = settings.security.jwt_secret_key

        if not secret:
            if (
                settings.app.environment
                == Environment.PRODUCTION
                and settings.security.require_secure_secret_in_production
            ):
                raise JWTConfigurationError(
                    "JWT secret key must be configured in production."
                )

            raise JWTConfigurationError(
                "JWT secret key has not been configured."
            )

        minimum_length = (
            settings.security.jwt_secret_min_length
        )

        if len(secret) < minimum_length:
            raise JWTConfigurationError(
                "JWT secret key does not meet "
                "the minimum security length."
            )

        if (
            settings.security.access_token_expire_minutes
            <= 0
        ):
            raise JWTConfigurationError(
                "Access token lifetime must be positive."
            )

        if (
            settings.security.refresh_token_expire_days
            <= 0
        ):
            raise JWTConfigurationError(
                "Refresh token lifetime must be positive."
            )

        if not settings.security.jwt_issuer.strip():
            raise JWTConfigurationError(
                "JWT issuer must not be empty."
            )

        if not settings.security.jwt_audience.strip():
            raise JWTConfigurationError(
                "JWT audience must not be empty."
            )

    def _create_token(
        self,
        *,
        subject: str,
        token_type: str,
        expires_delta: timedelta,
        extra_claims: dict[str, object] | None = None,
    ) -> str:
        """Create a signed JWT."""

        if not isinstance(subject, str):
            raise TypeError(
                "Token subject must be a string."
            )

        if not subject.strip():
            raise ValueError(
                "Token subject cannot be empty."
            )

        if token_type not in {
            self._ACCESS_TOKEN,
            self._REFRESH_TOKEN,
        }:
            raise ValueError(
                "Invalid token type."
            )

        now = datetime.now(timezone.utc)

        expires_at = now + expires_delta

        payload: dict[str, object] = {
            "sub": subject,
            "type": token_type,
            "iss": settings.security.jwt_issuer,
            "aud": settings.security.jwt_audience,
            "iat": now,
            "nbf": now,
            "exp": expires_at,
            "jti": str(uuid4()),
        }

        if extra_claims:
            payload.update(extra_claims)

        return jwt.encode(
            payload,
            settings.security.jwt_secret_key,
            algorithm=self.algorithm,
        )

    def create_access_token(
        self,
        subject: str,
        *,
        extra_claims: dict[str, object] | None = None,
    ) -> str:
        """Create a short-lived access token."""

        return self._create_token(
            subject=subject,
            token_type=self._ACCESS_TOKEN,
            expires_delta=timedelta(
                minutes=(
                    settings.security
                    .access_token_expire_minutes
                ),
            ),
            extra_claims=extra_claims,
        )

    def create_refresh_token(
        self,
        subject: str,
        *,
        extra_claims: dict[str, object] | None = None,
    ) -> str:
        """Create a longer-lived refresh token."""

        return self._create_token(
            subject=subject,
            token_type=self._REFRESH_TOKEN,
            expires_delta=timedelta(
                days=(
                    settings.security
                    .refresh_token_expire_days
                ),
            ),
            extra_claims=extra_claims,
        )

    def decode(
        self,
        token: str,
        *,
        expected_type: str | None = None,
    ) -> dict[str, object]:
        """Validate and decode a JWT."""

        if not isinstance(token, str) or not token:
            raise JWTValidationError(
                "Invalid authentication token."
            )

        try:
            payload = jwt.decode(
                token,
                settings.security.jwt_secret_key,
                algorithms=[self.algorithm],
                issuer=settings.security.jwt_issuer,
                audience=settings.security.jwt_audience,
                options={
                    "require": [
                        "sub",
                        "type",
                        "iss",
                        "aud",
                        "iat",
                        "nbf",
                        "exp",
                        "jti",
                    ],
                },
            )

        except InvalidTokenError as error:
            raise JWTValidationError(
                "Invalid authentication token."
            ) from error

        token_type = payload.get("type")

        if token_type not in {
            self._ACCESS_TOKEN,
            self._REFRESH_TOKEN,
        }:
            raise JWTValidationError(
                "Invalid authentication token."
            )

        if (
            expected_type is not None
            and token_type != expected_type
        ):
            raise JWTValidationError(
                "Invalid authentication token type."
            )

        subject = payload.get("sub")

        if not isinstance(subject, str):
            raise JWTValidationError(
                "Invalid authentication token."
            )

        if not subject.strip():
            raise JWTValidationError(
                "Invalid authentication token."
            )

        jti = payload.get("jti")

        if settings.security.require_jti:
            if not isinstance(jti, str) or not jti.strip():
                raise JWTValidationError(
                    "Invalid authentication token."
                )

        return payload

    def decode_access_token(
        self,
        token: str,
    ) -> dict[str, object]:
        """Validate and decode an access token."""

        return self.decode(
            token,
            expected_type=self._ACCESS_TOKEN,
        )

    def decode_refresh_token(
        self,
        token: str,
    ) -> dict[str, object]:
        """Validate and decode a refresh token."""

        return self.decode(
            token,
            expected_type=self._REFRESH_TOKEN,
        )


jwt_service = JWTService()