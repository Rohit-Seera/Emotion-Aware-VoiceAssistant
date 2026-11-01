"""Controlled voice service errors safe to map to HTTP responses."""


class VoiceServiceError(Exception):
    """Base error for the companion voice service."""


class VoiceSessionNotFoundError(VoiceServiceError):
    """Raised when a requested voice session is unknown."""


class VoiceSessionAccessError(VoiceServiceError):
    """Raised when a caller does not own a voice session."""


class VoiceTurnInProgressError(VoiceServiceError):
    """Raised when a session already has an active turn."""


class VoiceTurnCancelledError(VoiceServiceError):
    """Raised when an active voice turn was interrupted."""


class VoiceProviderUnavailableError(VoiceServiceError):
    """Raised when an STT or TTS provider has not been configured."""


class VoiceProcessingError(VoiceServiceError):
    """Raised when a provider or runtime fails.

    A short internal detail is retained for development diagnostics only;
    the HTTP layer still hides it in non-development environments.
    """

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "Voice request could not be completed.")
        self.detail = detail
