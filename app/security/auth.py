"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import Final

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.security.jwt import JWTValidationError, jwt_service


# Do not expose authentication failure details to clients.
_AUTHENTICATION_ERROR: Final[str] = "Authentication required."

# Bearer authentication extractor.
# auto_error=False lets us return one controlled error message
# instead of leaking framework-specific authentication details.
_bearer_scheme = HTTPBearer(
    auto_error=False,
)


def _authentication_error() -> HTTPException:
    """Return the standard authentication failure response."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_AUTHENTICATION_ERROR,
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )


def extract_bearer_token(
    credentials: HTTPAuthorizationCredentials | None,
) -> str:
    """Extract a Bearer token from HTTP authorization credentials."""

    if credentials is None:
        raise _authentication_error()

    # HTTPBearer already parses the scheme, but we explicitly
    # verify it before trusting the credential.
    if credentials.scheme.lower() != "bearer":
        raise _authentication_error()

    token = credentials.credentials

    if not isinstance(token, str) or not token.strip():
        raise _authentication_error()

    return token


async def get_current_token(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> str:
    """Return the raw authenticated access token.

    This dependency is intentionally small and reusable.
    JWT validation is performed by ``get_current_claims``.
    """

    return extract_bearer_token(credentials)


async def get_current_claims(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> dict[str, object]:
    """Validate an access token and return its trusted claims."""

    token = extract_bearer_token(credentials)

    try:
        return jwt_service.decode_access_token(token)
    except JWTValidationError as error:
        # Never expose the underlying JWT validation reason.
        raise _authentication_error() from error


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> str:
    """Return the authenticated subject from the access token."""

    claims = await get_current_claims(credentials)

    subject = claims.get("sub")

    if not isinstance(subject, str) or not subject.strip():
        # This should normally be impossible because JWTService
        # already validates the subject. Keep the boundary defensive.
        raise _authentication_error()

    return subject