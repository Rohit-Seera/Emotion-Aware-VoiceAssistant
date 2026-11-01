"""Application exceptions."""


class AssistantError(Exception):
    """Base exception for the application."""


class ConfigurationError(AssistantError):
    """Raised when the application configuration is invalid."""


class ValidationError(AssistantError):
    """Raised when validation fails."""


class AuthenticationError(AssistantError):
    """Raised when authentication fails."""


class AuthorizationError(AssistantError):
    """Raised when the user is not authorized."""


class ProviderError(AssistantError):
    """Raised when an AI provider fails."""


class RuntimeError(AssistantError):
    """Raised when the runtime encounters an error."""


class DatabaseError(AssistantError):
    """Raised when a database operation fails."""


class CacheError(AssistantError):
    """Raised when a cache operation fails."""


class VectorStoreError(AssistantError):
    """Raised when a vector database operation fails."""


class ServiceError(AssistantError):
    """Raised when an external service fails."""
