"""Groq Whisper speech-to-text adapter."""

from __future__ import annotations

import io

try:
    from groq import APIStatusError, AsyncGroq
except ImportError:  # optional SDK for local/dev startup
    APIStatusError = Exception  # type: ignore[assignment,misc]
    AsyncGroq = None  # type: ignore[assignment,misc]

from app.core.logger import get_logger

from app.core.settings import settings
from app.services.speech.contracts import TranscriptionResult


logger = get_logger(__name__)


class GroqWhisperSTT:
    """Speech-to-text provider backed by Groq Whisper Large V3 Turbo."""

    def __init__(self) -> None:
        if AsyncGroq is None:
            raise RuntimeError("Groq SDK is not installed.")

        api_key = settings.ai.groq.api_key

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self._client = AsyncGroq(api_key=api_key)
        self._model = "whisper-large-v3-turbo"

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe in-memory audio using Groq Whisper Large V3 Turbo."""

        extension_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
            "audio/flac": "flac",
            "audio/x-flac": "flac",
        }

        extension = extension_map.get(
            content_type,
            "wav",
        )

        filename = f"audio.{extension}"

        request_kwargs: dict[str, object] = {
            "file": (
                filename,
                io.BytesIO(audio),
                content_type,
            ),
            "model": self._model,
            "response_format": "json",
        }

        if language:
            request_kwargs["language"] = language

        try:
            response = await self._client.audio.transcriptions.create(
                **request_kwargs,
            )
        except APIStatusError as error:
            body = getattr(error, "body", None)
            logger.error(
                "Groq STT rejected audio: status=%s body=%r content_type=%s bytes=%d",
                getattr(error, "status_code", None),
                body,
                content_type,
                len(audio),
            )
            raise
        except Exception:
            logger.exception(
                "Groq STT request failed: content_type=%s bytes=%d",
                content_type,
                len(audio),
            )
            raise

        text = getattr(response, "text", "").strip()

        response_language = getattr(
            response,
            "language",
            None,
        )

        duration = getattr(
            response,
            "duration",
            None,
        )

        return TranscriptionResult(
            text=text,
            language=(
                str(response_language)
                if response_language
                else language
            ),
            duration_seconds=(
                float(duration)
                if duration is not None
                else None
            ),
        )

    async def close(self) -> None:
        """Close the underlying Groq client."""

        await self._client.close()
