"""Exceptions raised by AI providers."""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for provider-related failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)

        self.provider = provider
        self.retryable = retryable


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is incorrectly configured."""


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )

        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is temporarily unavailable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            retryable=True,
        )


class ProviderRequestError(ProviderError):
    """Raised when a provider rejects or cannot process a request."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns an invalid or unusable response."""


class ProviderStreamingError(ProviderError):
    """Raised when provider streaming fails."""