"""Tests for authorization and permission utilities."""

import pytest
from fastapi import HTTPException

from app.security.permissions import (
    Permission,
    Role,
    get_permissions_for_role,
    get_role_from_claims,
    has_permission,
    require_any_role,
    require_permission,
    require_role,
)


def user_claims() -> dict[str, object]:
    """Return standard user claims."""

    return {
        "sub": "user-123",
        "type": "access",
        "role": Role.USER.value,
    }


def admin_claims() -> dict[str, object]:
    """Return standard admin claims."""

    return {
        "sub": "admin-123",
        "type": "access",
        "role": Role.ADMIN.value,
    }


def service_claims() -> dict[str, object]:
    """Return standard service claims."""

    return {
        "sub": "service-123",
        "type": "access",
        "role": Role.SERVICE.value,
    }


def test_user_role_has_expected_permissions() -> None:
    """Users should receive only user permissions."""

    permissions = get_permissions_for_role(
        Role.USER.value,
    )

    assert Permission.CHAT.value in permissions
    assert Permission.VOICE.value in permissions
    assert Permission.MEMORY_READ.value in permissions
    assert Permission.MEMORY_WRITE.value in permissions

    assert Permission.ADMIN.value not in permissions
    assert Permission.AUTOMATION.value not in permissions


def test_admin_role_has_admin_permissions() -> None:
    """Admins should receive administrative permissions."""

    permissions = get_permissions_for_role(
        Role.ADMIN.value,
    )

    assert Permission.ADMIN.value in permissions
    assert Permission.TOOLS.value in permissions
    assert Permission.RESEARCH.value in permissions
    assert Permission.AUTOMATION.value in permissions


def test_service_role_is_restricted() -> None:
    """Service roles should not automatically receive user permissions."""

    permissions = get_permissions_for_role(
        Role.SERVICE.value,
    )

    assert Permission.TOOLS.value in permissions
    assert Permission.RESEARCH.value in permissions

    assert Permission.ADMIN.value not in permissions
    assert Permission.MEMORY_WRITE.value not in permissions


def test_extracts_valid_role() -> None:
    """A known role should be returned."""

    assert (
        get_role_from_claims(user_claims())
        == Role.USER.value
    )


def test_missing_role_is_rejected() -> None:
    """Claims without a role must not be authorized."""

    claims = {
        "sub": "user-123",
        "type": "access",
    }

    with pytest.raises(HTTPException) as error:
        get_role_from_claims(claims)

    assert error.value.status_code == 403
    assert error.value.detail == "Insufficient permissions."


def test_unknown_role_is_rejected() -> None:
    """Unknown roles must never receive implicit permissions."""

    claims = {
        "sub": "user-123",
        "type": "access",
        "role": "super-admin",
    }

    with pytest.raises(HTTPException) as error:
        get_role_from_claims(claims)

    assert error.value.status_code == 403


def test_user_has_chat_permission() -> None:
    """Normal users should be allowed to chat."""

    assert has_permission(
        user_claims(),
        Permission.CHAT,
    ) is True


def test_user_cannot_use_admin_permission() -> None:
    """Normal users must not receive admin permission."""

    assert has_permission(
        user_claims(),
        Permission.ADMIN,
    ) is False


def test_admin_has_admin_permission() -> None:
    """Admins should have administrative permission."""

    assert has_permission(
        admin_claims(),
        Permission.ADMIN,
    ) is True


def test_service_cannot_write_memory() -> None:
    """Service role must not receive memory write permission."""

    assert has_permission(
        service_claims(),
        Permission.MEMORY_WRITE,
    ) is False


@pytest.mark.asyncio
async def test_require_permission_allows_valid_claims() -> None:
    """Permission dependency should allow authorized claims."""

    dependency = require_permission(
        Permission.CHAT,
    )

    result = await dependency(
        claims=user_claims(),
    )

    assert result["sub"] == "user-123"


@pytest.mark.asyncio
async def test_require_permission_rejects_invalid_claims() -> None:
    """Permission dependency should reject unauthorized claims."""

    dependency = require_permission(
        Permission.ADMIN,
    )

    with pytest.raises(HTTPException) as error:
        await dependency(
            claims=user_claims(),
        )

    assert error.value.status_code == 403
    assert error.value.detail == "Insufficient permissions."


@pytest.mark.asyncio
async def test_require_role_allows_matching_role() -> None:
    """Role dependency should allow the matching role."""

    dependency = require_role(Role.ADMIN)

    result = await dependency(
        claims=admin_claims(),
    )

    assert result["sub"] == "admin-123"


@pytest.mark.asyncio
async def test_require_role_rejects_wrong_role() -> None:
    """Role dependency should reject a different role."""

    dependency = require_role(Role.ADMIN)

    with pytest.raises(HTTPException) as error:
        await dependency(
            claims=user_claims(),
        )

    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_require_any_role_accepts_allowed_role() -> None:
    """Any-role dependency should accept one of the allowed roles."""

    dependency = require_any_role(
        Role.USER,
        Role.ADMIN,
    )

    result = await dependency(
        claims=user_claims(),
    )

    assert result["sub"] == "user-123"


@pytest.mark.asyncio
async def test_require_any_role_rejects_disallowed_role() -> None:
    """Any-role dependency should reject roles outside the allow-list."""

    dependency = require_any_role(
        Role.ADMIN,
    )

    with pytest.raises(HTTPException) as error:
        await dependency(
            claims=user_claims(),
        )

    assert error.value.status_code == 403


def test_unknown_required_role_is_rejected() -> None:
    """Dependencies must reject unknown roles at construction time."""

    with pytest.raises(
        ValueError,
        match="Unknown authorization role",
    ):
        require_role("super-admin")


def test_empty_any_role_list_is_rejected() -> None:
    """At least one role must be supplied."""

    with pytest.raises(
        ValueError,
        match="At least one authorization role",
    ):
        require_any_role()