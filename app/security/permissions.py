"""Authorization and permission utilities."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from fastapi import Depends, HTTPException, status

from app.security.auth import get_current_claims


class Role(StrEnum):
    """Supported application roles."""

    USER = "user"
    ADMIN = "admin"
    SERVICE = "service"


class Permission(StrEnum):
    """Supported application permissions."""

    CHAT = "chat"
    VOICE = "voice"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    RESEARCH = "research"
    AUTOMATION = "automation"
    TOOLS = "tools"
    ADMIN = "admin"


# Explicit allow-list.
#
# Do NOT derive permissions dynamically from role names.
# Every role must have an explicitly defined permission set.
ROLE_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    Role.USER.value: frozenset(
        {
            Permission.CHAT.value,
            Permission.VOICE.value,
            Permission.MEMORY_READ.value,
            Permission.MEMORY_WRITE.value,
        }
    ),
    Role.ADMIN.value: frozenset(
        {
            Permission.CHAT.value,
            Permission.VOICE.value,
            Permission.MEMORY_READ.value,
            Permission.MEMORY_WRITE.value,
            Permission.RESEARCH.value,
            Permission.AUTOMATION.value,
            Permission.TOOLS.value,
            Permission.ADMIN.value,
        }
    ),
    Role.SERVICE.value: frozenset(
        {
            Permission.RESEARCH.value,
            Permission.TOOLS.value,
        }
    ),
}


_AUTHENTICATION_ERROR: Final[str] = "Authentication required."
_AUTHORIZATION_ERROR: Final[str] = "Insufficient permissions."


def _authentication_error() -> HTTPException:
    """Return a generic authentication error."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_AUTHENTICATION_ERROR,
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )


def _authorization_error() -> HTTPException:
    """Return a generic authorization error."""

    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=_AUTHORIZATION_ERROR,
    )


def _extract_role(
    claims: dict[str, object],
) -> str:
    """Extract and validate the role claim."""

    role = claims.get("role")

    if not isinstance(role, str) or not role:
        raise _authorization_error()

    if role not in ROLE_PERMISSIONS:
        raise _authorization_error()

    return role


def get_role_from_claims(
    claims: dict[str, object],
) -> str:
    """Return the validated role from trusted JWT claims."""

    return _extract_role(claims)


def get_permissions_for_role(
    role: str,
) -> frozenset[str]:
    """Return the permissions assigned to a role."""

    permissions = ROLE_PERMISSIONS.get(role)

    if permissions is None:
        raise _authorization_error()

    return permissions


def has_permission(
    claims: dict[str, object],
    permission: str | Permission,
) -> bool:
    """Return whether the claims grant the requested permission."""

    role = _extract_role(claims)

    requested_permission = str(permission)

    return requested_permission in ROLE_PERMISSIONS[role]


def require_permission(
    permission: str | Permission,
):
    """Create a FastAPI dependency requiring a permission."""

    required_permission = str(permission)

    async def dependency(
        claims: dict[str, object] = Depends(
            get_current_claims,
        ),
    ) -> dict[str, object]:
        """Validate the required permission."""

        if not has_permission(
            claims,
            required_permission,
        ):
            raise _authorization_error()

        return claims

    return dependency


def require_role(
    role: str | Role,
):
    """Create a FastAPI dependency requiring a specific role."""

    required_role = str(role)

    if required_role not in ROLE_PERMISSIONS:
        raise ValueError(
            "Unknown authorization role."
        )

    async def dependency(
        claims: dict[str, object] = Depends(
            get_current_claims,
        ),
    ) -> dict[str, object]:
        """Validate the required role."""

        actual_role = _extract_role(claims)

        if actual_role != required_role:
            raise _authorization_error()

        return claims

    return dependency


def require_any_role(
    *roles: str | Role,
):
    """Create a dependency accepting any of the supplied roles."""

    allowed_roles = frozenset(
        str(role)
        for role in roles
    )

    if not allowed_roles:
        raise ValueError(
            "At least one authorization role is required."
        )

    unknown_roles = (
        allowed_roles - ROLE_PERMISSIONS.keys()
    )

    if unknown_roles:
        raise ValueError(
            "Unknown authorization role."
        )

    async def dependency(
        claims: dict[str, object] = Depends(
            get_current_claims,
        ),
    ) -> dict[str, object]:
        """Validate membership in the allowed roles."""

        actual_role = _extract_role(claims)

        if actual_role not in allowed_roles:
            raise _authorization_error()

        return claims

    return dependency