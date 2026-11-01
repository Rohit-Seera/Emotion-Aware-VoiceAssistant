"""Voice companion service layer."""

from app.services.speech.service import (
    VoiceCompanionService,
    build_voice_companion_service,
)

__all__ = [
    "VoiceCompanionService",
    "build_voice_companion_service",
]
