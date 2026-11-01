"""Secure password hashing utilities."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

from app.core.settings import settings


class PasswordHasherService:
    """Provide secure password hashing and verification."""

    def __init__(self) -> None:
        self._hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16,
        )

    def validate_password(self, password: str) -> None:
        """Validate password input against the application policy."""

        if not isinstance(password, str):
            raise TypeError("Password must be a string.")

        if len(password) < settings.security.password_min_length:
            raise ValueError(
                "Password does not meet the minimum length requirement."
            )

        if len(password) > settings.security.password_max_length:
            raise ValueError(
                "Password exceeds the maximum allowed length."
            )

    def hash(self, password: str) -> str:
        """Create an Argon2id password hash."""

        self.validate_password(password)

        return self._hasher.hash(password)

    def verify(
        self,
        password: str,
        password_hash: str,
    ) -> bool:
        """Verify a password against an Argon2id hash."""

        if not isinstance(password, str):
            return False

        if not isinstance(password_hash, str):
            return False

        try:
            return self._hasher.verify(
                password_hash,
                password,
            )

        except (
            VerifyMismatchError,
            VerificationError,
            InvalidHashError,
            HashingError,
        ):
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """Return whether a stored hash should be upgraded."""

        try:
            return self._hasher.check_needs_rehash(
                password_hash,
            )
        except (
            VerificationError,
            InvalidHashError,
            HashingError,
        ):
            return True


password_hasher = PasswordHasherService()