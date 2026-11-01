"""Tests for secure password hashing."""

from argon2 import PasswordHasher as Argon2PasswordHasher

from app.security.hashing import (
    PasswordHasherService,
    password_hasher,
)


def test_password_hash_is_not_plaintext() -> None:
    """Password hashes must never equal the original password."""

    password = "A-very-strong-password-123!"

    hashed = password_hasher.hash(password)

    assert hashed != password
    assert hashed.startswith("$argon2id$")


def test_password_hash_can_be_verified() -> None:
    """A valid password should verify successfully."""

    password = "A-very-strong-password-123!"

    hashed = password_hasher.hash(password)

    assert password_hasher.verify(
        password,
        hashed,
    ) is True


def test_wrong_password_fails_verification() -> None:
    """An incorrect password must fail verification."""

    password = "A-very-strong-password-123!"

    hashed = password_hasher.hash(password)

    assert password_hasher.verify(
        "Wrong-password-123!",
        hashed,
    ) is False


def test_each_hash_uses_a_unique_salt() -> None:
    """Two hashes of the same password should differ."""

    password = "A-very-strong-password-123!"

    first = password_hasher.hash(password)
    second = password_hasher.hash(password)

    assert first != second
    assert password_hasher.verify(password, first)
    assert password_hasher.verify(password, second)


def test_short_password_is_rejected() -> None:
    """Passwords below the configured minimum must be rejected."""

    try:
        password_hasher.hash("short")
    except ValueError as error:
        assert "minimum length" in str(error)
    else:
        raise AssertionError(
            "Expected short password to be rejected."
        )


def test_long_password_is_rejected() -> None:
    """Passwords above the configured maximum must be rejected."""

    password = "A" * 129

    try:
        password_hasher.hash(password)
    except ValueError as error:
        assert "maximum" in str(error)
    else:
        raise AssertionError(
            "Expected long password to be rejected."
        )


def test_malformed_hash_fails_safely() -> None:
    """Malformed hashes must not leak hashing implementation errors."""

    assert password_hasher.verify(
        "some-password",
        "not-a-valid-argon2-hash",
    ) is False


def test_non_string_password_fails_safely() -> None:
    """Non-string passwords must not be accepted during verification."""

    assert password_hasher.verify(
        123,  # type: ignore[arg-type]
        "not-a-valid-hash",
    ) is False


def test_hash_uses_argon2id() -> None:
    """The configured service should use Argon2id."""

    service = PasswordHasherService()

    password = "A-very-strong-password-123!"

    hashed = service.hash(password)

    assert hashed.startswith("$argon2id$")


def test_needs_rehash_returns_false_for_current_hash() -> None:
    """Current hashes should not require an immediate upgrade."""

    password = "A-very-strong-password-123!"

    hashed = password_hasher.hash(password)

    assert password_hasher.needs_rehash(hashed) is False


def test_needs_rehash_handles_invalid_hash() -> None:
    """Invalid hashes should be treated as needing replacement."""

    assert password_hasher.needs_rehash(
        "not-a-valid-hash",
    ) is True


def test_argon2_dependency_is_available() -> None:
    """The Argon2 implementation should be available."""

    assert Argon2PasswordHasher is not None