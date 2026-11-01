"""ElevenLabs text-to-speech provider with emotion-conditioned delivery."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

try:
    from elevenlabs import AsyncElevenLabs, VoiceSettings
except ImportError:  # pragma: no cover - exercised only without optional SDK
    AsyncElevenLabs = None  # type: ignore[assignment]
    VoiceSettings = None  # type: ignore[assignment]

from app.core.settings import settings
from app.services.speech.contracts import AudioResult


class ElevenLabsTTS:
    """ElevenLabs implementation of the voice companion TTS contract.

    Emotion is translated into per-request voice settings instead of mutating
    the stored voice configuration. This preserves a consistent character
    voice while changing stability, style, and speaking rate for the response.
    """

    name = "elevenlabs"

    _EMOTION_SETTINGS: dict[str, dict[str, float | bool]] = {
        "neutral": {
            "stability": 0.62,
            "similarity_boost": 0.90,
            "style": 0.02,
            "speed": 1.00,
            "use_speaker_boost": True,
        },
        "warm": {
            "stability": 0.58,
            "similarity_boost": 0.90,
            "style": 0.08,
            "speed": 1.00,
            "use_speaker_boost": True,
        },
        "empathetic": {
            "stability": 0.54,
            "similarity_boost": 0.90,
            "style": 0.18,
            "speed": 0.95,
            "use_speaker_boost": True,
        },
        "excited": {
            "stability": 0.58,
            "similarity_boost": 0.90,
            "style": 0.16,
            "speed": 1.03,
            "use_speaker_boost": True,
        },
        "calm": {
            "stability": 0.66,
            "similarity_boost": 0.90,
            "style": 0.05,
            "speed": 0.96,
            "use_speaker_boost": True,
        },
        "reassuring": {
            "stability": 0.56,
            "similarity_boost": 0.90,
            "style": 0.14,
            "speed": 0.96,
            "use_speaker_boost": True,
        },
    }

    def __init__(self) -> None:
        config = settings.elevenlabs

        if AsyncElevenLabs is None or VoiceSettings is None:
            raise RuntimeError(
                "The ElevenLabs Python SDK is not installed."
            )

        if not config.api_key:
            raise RuntimeError(
                "ElevenLabs API key is not configured."
            )

        if not config.voice_id:
            raise RuntimeError(
                "ElevenLabs voice ID is not configured."
            )

        self._voice_id = config.voice_id
        self._model = config.model

        self._client = AsyncElevenLabs(
            api_key=config.api_key,
        )

    def _voice_settings(
        self,
        emotion: str | None,
    ) -> Any | None:
        """Create a per-request ElevenLabs voice-settings object."""

        if not emotion:
            return None

        values = self._EMOTION_SETTINGS.get(
            emotion.casefold(),
            self._EMOTION_SETTINGS["warm"],
        )

        return VoiceSettings(**values)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AudioResult:
        """Generate complete MP3 audio using emotion-conditioned settings."""

        voice_id = self._voice_id
        voice_settings = self._voice_settings(emotion)

        request_kwargs: dict[str, Any] = {
            "voice_id": voice_id,
            "text": text,
            "model_id": self._model,
            "output_format": "mp3_44100_128",
        }

        if language:
            request_kwargs["language_code"] = language

        if voice_settings is not None:
            request_kwargs["voice_settings"] = voice_settings

        try:
            audio_stream = self._client.text_to_speech.convert(
                **request_kwargs,
            )

            chunks: list[bytes] = []
            async for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)

            return AudioResult(
                data=b"".join(chunks),
                content_type="audio/mpeg",
            )

        except Exception as error:
            raise RuntimeError(
                "ElevenLabs synthesis failed."
            ) from error

    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream MP3 audio chunks using emotion-conditioned settings."""

        voice_id = self._voice_id
        voice_settings = self._voice_settings(emotion)

        request_kwargs: dict[str, Any] = {
            "voice_id": voice_id,
            "text": text,
            "model_id": self._model,
            "output_format": "mp3_44100_128",
        }

        if language:
            request_kwargs["language_code"] = language

        if voice_settings is not None:
            request_kwargs["voice_settings"] = voice_settings

        try:
            audio_stream = self._client.text_to_speech.convert(
                **request_kwargs,
            )

            async for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    yield chunk

        except Exception as error:
            raise RuntimeError(
                "ElevenLabs streaming failed."
            ) from error

    async def close(self) -> None:
        """Close the ElevenLabs client."""

        if self._client is None:
            return

        close_method = getattr(self._client, "close", None)

        if close_method is not None:
            result = close_method()

            if hasattr(result, "__await__"):
                await result
