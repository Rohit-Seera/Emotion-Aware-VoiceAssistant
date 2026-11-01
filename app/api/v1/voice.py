"""Authenticated HTTP API for the provider-neutral voice companion."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.services.speech.exceptions import (
    VoiceProcessingError,
    VoiceProviderUnavailableError,
    VoiceSessionAccessError,
    VoiceSessionNotFoundError,
    VoiceTurnCancelledError,
    VoiceTurnInProgressError,
)
from app.services.speech.service import VoiceCompanionService


router = APIRouter(prefix="/voice", tags=["voice"])

_bearer_scheme = HTTPBearer(auto_error=False)
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


def _header_text(value: object) -> str:
    """Encode arbitrary UTF-8 text so Starlette can emit a Latin-1 HTTP header."""

    return quote(str(value), safe="-_.~:/;,@")


class VoiceSessionResponse(BaseModel):
    """A caller-owned companion session."""

    session_id: UUID
    status: str = "idle"


class VoiceTurnRequest(BaseModel):
    """A transcript ready to be handled by the companion runtime."""

    transcript: str = Field(min_length=1, max_length=8_000)
    language: str | None = Field(default=None, max_length=16)


class ExpressionResponse(BaseModel):
    """Constrained frontend animation and delivery instructions."""

    emotion: str
    expression: str
    gesture: str
    gaze: str
    lip_sync: str
    voice_style: str


class TokenUsageResponse(BaseModel):
    """Provider token usage available after a completed runtime turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VoiceTurnResponse(BaseModel):
    """Companion text, emotion analysis, and animation instructions."""

    session_id: UUID
    request_id: UUID
    text: str
    expression: ExpressionResponse
    user_emotion: str
    emotion_confidence: float
    emotion_source: str
    emotion_top: list[dict[str, object]]
    token_usage: TokenUsageResponse
    llm_provider: str
    llm_model: str
    processing_ms: float
    emotion_acoustic: dict[str, object] | None
    emotion_text: dict[str, object] | None
    emotion_policy: str


class TranscriptionResponse(BaseModel):
    """A normalized STT response."""

    transcript: str
    language: str | None = None
    duration_seconds: float | None = None


class SynthesisRequest(BaseModel):
    """Parameters sent to a configured text-to-speech provider."""

    text: str = Field(min_length=1, max_length=8_000)
    voice: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=16)
    emotion: str | None = Field(default=None, max_length=64)


class CancelTurnResponse(BaseModel):
    """Result of a client barge-in cancellation request."""

    cancelled: bool


class VoiceSessionMetricsResponse(BaseModel):
    """Research/demo metrics for a caller-owned voice session."""

    session_id: UUID
    active: bool
    interruptions: int


async def require_voice_claims(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        _bearer_scheme,
    ),
) -> dict[str, object]:
    """Enforce the existing JWT and VOICE permission without eager secrets."""

    try:
        from app.security.auth import get_current_claims
        from app.security.permissions import Permission, has_permission

        claims = await get_current_claims(credentials)

        if not has_permission(claims, Permission.VOICE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )

        return claims

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable.",
        ) from error


def get_voice_service(request: Request) -> VoiceCompanionService:
    """Return the lifespan-owned companion service."""

    service = getattr(
        request.app.state,
        "voice_companion_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice service unavailable.",
        )

    return service


def get_subject(claims: dict[str, object]) -> str:
    """Extract the authenticated subject from JWT claims."""

    subject = claims.get("sub")

    if not isinstance(subject, str) or not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication claims.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return subject


def map_voice_error(error: Exception) -> HTTPException:
    """Map internal voice errors to safe HTTP responses."""

    if isinstance(
        error,
        (
            VoiceSessionNotFoundError,
            VoiceSessionAccessError,
        ),
    ):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice session not found.",
        )

    if isinstance(error, VoiceTurnInProgressError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A voice turn is already in progress.",
        )

    if isinstance(error, VoiceTurnCancelledError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Voice turn was interrupted.",
        )

    if isinstance(error, VoiceProviderUnavailableError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice provider unavailable.",
        )

    if isinstance(error, VoiceProcessingError):
        detail = "Voice request could not be completed."
        if settings.is_development and getattr(error, "detail", None):
            detail = error.detail
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )

    if isinstance(error, ValueError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid voice request.",
        )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Voice service unavailable.",
    )


@router.post(
    "/sessions",
    response_model=VoiceSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> VoiceSessionResponse:
    """Create a session scoped to the authenticated user."""

    session = await service.create_session(
        get_subject(claims),
    )

    return VoiceSessionResponse(
        session_id=session.id,
    )


@router.post(
    "/sessions/{session_id}/turns",
    response_model=VoiceTurnResponse,
)
async def process_turn(
    session_id: UUID,
    payload: VoiceTurnRequest,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> VoiceTurnResponse:
    """Process one text transcript through the companion runtime."""

    try:
        turn = await service.process_turn(
            session_id=session_id,
            owner_id=get_subject(claims),
            transcript=payload.transcript,
            language=payload.language,
        )

    except Exception as error:
        raise map_voice_error(error) from error

    return VoiceTurnResponse(
        session_id=turn.session_id,
        request_id=turn.request_id,
        text=turn.text,
        expression=ExpressionResponse(
            emotion=turn.expression.emotion,
            expression=turn.expression.expression,
            gesture=turn.expression.gesture,
            gaze=turn.expression.gaze,
            lip_sync=turn.expression.lip_sync,
            voice_style=turn.expression.voice_style,
        ),
        user_emotion=turn.user_emotion,
        emotion_confidence=turn.emotion_confidence,
        emotion_source=turn.emotion_source,
        emotion_top=[dict(item) for item in turn.emotion_top],
        token_usage=TokenUsageResponse(
            prompt_tokens=turn.token_usage.prompt_tokens,
            completion_tokens=turn.token_usage.completion_tokens,
            total_tokens=turn.token_usage.total_tokens,
        ),
        llm_provider=turn.llm_provider,
        llm_model=turn.llm_model,
        processing_ms=turn.processing_ms,
        emotion_acoustic=turn.emotion_acoustic,
        emotion_text=turn.emotion_text,
        emotion_policy=turn.emotion_policy,
    )


@router.post(
    "/sessions/{session_id}/audio-turn",
)
async def process_audio_turn(
    session_id: UUID,
    request: Request,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
    language: str | None = None,
    voice: str | None = None,
) -> Response:
    """
    Process one audio turn through:

    STT -> Aiko companion runtime -> ElevenLabs TTS.
    """

    content_length = request.headers.get("content-length")

    if content_length is not None:
        try:
            if int(content_length) > settings.voice.max_audio_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Audio payload is too large.",
                )

        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid content length.",
            ) from error

    try:
        audio = await request.body()

        if len(audio) > settings.voice.max_audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio payload is too large.",
            )

        result = await service.process_audio_turn(
            session_id=session_id,
            owner_id=get_subject(claims),
            audio=audio,
            content_type=request.headers.get(
                "content-type",
                "",
            ),
            language=language,
            voice=voice,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise map_voice_error(error) from error

    turn = result.turn

    headers = {
        **_NO_STORE_HEADERS,
        "X-Voice-Request-Id": str(
            turn.request_id,
        ),
        "X-Voice-Session-Id": str(
            turn.session_id,
        ),
        "X-Voice-Transcript": _header_text(result.transcript),
        "X-Voice-Response": _header_text(turn.text),
        "X-Voice-Emotion": turn.expression.emotion,
        "X-Voice-User-Emotion": turn.user_emotion,
        "X-Voice-Emotion-Confidence": f"{turn.emotion_confidence:.4f}",
        "X-Voice-Emotion-Source": turn.emotion_source,
        "X-Voice-Emotion-Top": ";".join(
            f"{item.get('emotion')}:{float(item.get('confidence', 0.0)):.3f}"
            for item in turn.emotion_top
        ),
        "X-Voice-Expression": turn.expression.expression,
        "X-Voice-Gesture": turn.expression.gesture,
        "X-Voice-Gaze": turn.expression.gaze,
        "X-Voice-Lip-Sync": turn.expression.lip_sync,
        "X-Voice-Voice-Style": turn.expression.voice_style,
        "X-Voice-Prompt-Tokens": str(
            turn.token_usage.prompt_tokens,
        ),
        "X-Voice-Completion-Tokens": str(
            turn.token_usage.completion_tokens,
        ),
        "X-Voice-Total-Tokens": str(
            turn.token_usage.total_tokens,
        ),
    }

    return Response(
        content=result.audio.data,
        media_type=result.audio.content_type,
        headers=headers,
    )



@router.post(
    "/sessions/{session_id}/audio-stream-turn",
)
async def process_audio_stream_turn(
    session_id: UUID,
    request: Request,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
    language: str | None = None,
    voice: str | None = None,
) -> StreamingResponse:
    """Stream a voice response so the active turn can be interrupted."""

    content_length = request.headers.get("content-length")

    if content_length is not None:
        try:
            if int(content_length) > settings.voice.max_audio_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Audio payload is too large.",
                )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid content length.",
            ) from error

    try:
        audio = await request.body()

        if len(audio) > settings.voice.max_audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio payload is too large.",
            )

        if not service.provider_status().get("tts_ready", False):
            turn, transcript, _ = await service.transcribe_and_process_turn(
                session_id=session_id,
                owner_id=get_subject(claims),
                audio=audio,
                content_type=request.headers.get("content-type", ""),
                language=language,
            )
            fallback_headers = {
                **_NO_STORE_HEADERS,
                "X-Voice-TTS-Fallback": "unavailable",
                "X-Voice-Session-Id": str(turn.session_id),
                "X-Voice-Request-Id": str(turn.request_id),
                "X-Voice-Transcript": _header_text(transcript),
                "X-Voice-Response": _header_text(turn.text),
                "X-Voice-Emotion": turn.expression.emotion,
                "X-Voice-User-Emotion": turn.user_emotion,
                "X-Voice-Emotion-Confidence": f"{turn.emotion_confidence:.4f}",
                "X-Voice-Emotion-Source": turn.emotion_source,
                "X-Voice-Emotion-Top": ";".join(
                    f"{item.get('emotion')}:{float(item.get('confidence', 0.0)):.3f}"
                    for item in turn.emotion_top
                ),
                "X-Voice-Expression": turn.expression.expression,
                "X-Voice-Gesture": turn.expression.gesture,
                "X-Voice-Gaze": turn.expression.gaze,
                "X-Voice-Lip-Sync": turn.expression.lip_sync,
                "X-Voice-Voice-Style": turn.expression.voice_style,
                "X-Voice-LLM-Provider": turn.llm_provider,
                "X-Voice-LLM-Model": turn.llm_model,
                "X-Voice-Processing-Ms": f"{turn.processing_ms:.1f}",
                "X-Voice-TTS-Provider": "unavailable",
                "X-Voice-Acoustic-Emotion": _header_text(
                    (turn.emotion_acoustic or {}).get("emotion", "")
                ),
                "X-Voice-Acoustic-Confidence": str(
                    (turn.emotion_acoustic or {}).get("confidence", "")
                ),
                "X-Voice-Lexical-Emotion": _header_text(
                    (turn.emotion_text or {}).get("emotion", "")
                ),
                "X-Voice-Lexical-Confidence": str(
                    (turn.emotion_text or {}).get("confidence", "")
                ),
                "X-Voice-Emotion-Policy": _header_text(turn.emotion_policy),
            }
            async def empty_stream():
                if False:
                    yield b""
            return StreamingResponse(
                empty_stream(),
                media_type="audio/mpeg",
                headers=fallback_headers,
            )

        result = await service.process_audio_stream_turn(
            session_id=session_id,
            owner_id=get_subject(claims),
            audio=audio,
            content_type=request.headers.get("content-type", ""),
            language=language,
            voice=voice,
        )

    except HTTPException:
        raise
    except Exception as error:
        raise map_voice_error(error) from error

    turn = result.turn

    headers = {
        **_NO_STORE_HEADERS,
        "X-Voice-Request-Id": str(turn.request_id),
        "X-Voice-Session-Id": str(turn.session_id),
        "X-Voice-Transcript": _header_text(result.transcript),
        "X-Voice-Response": _header_text(turn.text),
        "X-Voice-Emotion": turn.expression.emotion,
        "X-Voice-User-Emotion": turn.user_emotion,
        "X-Voice-Emotion-Confidence": f"{turn.emotion_confidence:.4f}",
        "X-Voice-Emotion-Source": turn.emotion_source,
        "X-Voice-Emotion-Top": ";".join(
            f"{item.get('emotion')}:{float(item.get('confidence', 0.0)):.3f}"
            for item in turn.emotion_top
        ),
        "X-Voice-Expression": turn.expression.expression,
        "X-Voice-Gesture": turn.expression.gesture,
        "X-Voice-Gaze": turn.expression.gaze,
        "X-Voice-Lip-Sync": turn.expression.lip_sync,
        "X-Voice-Voice-Style": turn.expression.voice_style,
        "X-Voice-LLM-Provider": turn.llm_provider,
        "X-Voice-LLM-Model": turn.llm_model,
        "X-Voice-Processing-Ms": f"{turn.processing_ms:.1f}",
        "X-Voice-TTS-Provider": str(
            service.provider_status().get("tts_provider") or "unknown"
        ),
        "X-Voice-Acoustic-Emotion": _header_text(
            (turn.emotion_acoustic or {}).get("emotion", "")
        ),
        "X-Voice-Acoustic-Confidence": str(
            (turn.emotion_acoustic or {}).get("confidence", "")
        ),
        "X-Voice-Lexical-Emotion": _header_text(
            (turn.emotion_text or {}).get("emotion", "")
        ),
        "X-Voice-Lexical-Confidence": str(
            (turn.emotion_text or {}).get("confidence", "")
        ),
        "X-Voice-Emotion-Policy": _header_text(turn.emotion_policy),
    }

    return StreamingResponse(
        result.audio_stream,
        media_type="audio/mpeg",
        headers=headers,
    )

@router.post(
    "/sessions/{session_id}/cancel",
    response_model=CancelTurnResponse,
)
async def cancel_turn(
    session_id: UUID,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> CancelTurnResponse:
    """Interrupt the caller's active response for barge-in support."""

    try:
        cancelled = await service.cancel_turn(
            session_id,
            get_subject(claims),
        )

    except Exception as error:
        raise map_voice_error(error) from error

    return CancelTurnResponse(
        cancelled=cancelled,
    )


@router.get(
    "/sessions/{session_id}/metrics",
    response_model=VoiceSessionMetricsResponse,
)
async def session_metrics(
    session_id: UUID,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> VoiceSessionMetricsResponse:
    """Return safe research metrics for the caller's own voice session."""

    try:
        metrics = await service.session_metrics(
            session_id,
            get_subject(claims),
        )
    except Exception as error:
        raise map_voice_error(error) from error

    return VoiceSessionMetricsResponse(**metrics)


@router.get(
    "/emotion/status",
)
async def emotion_status(
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> dict[str, object]:
    """Expose non-sensitive acoustic SER diagnostics for the demo."""

    del claims
    return service.emotion_status()


@router.post(
    "/emotion/warmup",
)
async def warmup_emotion(
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> dict[str, object]:
    """Load the acoustic SER model before a live research demo."""

    del claims
    try:
        return await service.warmup_emotion_model()
    except Exception as error:
        # Keep the endpoint usable even when the optional model cannot be
        # downloaded; the voice pipeline will transparently fall back to text.
        return {
            **service.emotion_status(),
            "warmup_error": str(error),
        }


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
)
async def transcribe_audio(
    request: Request,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
    language: str | None = None,
) -> TranscriptionResponse:
    """Transcribe raw audio without retaining the source bytes on disk."""

    del claims

    content_length = request.headers.get("content-length")

    if content_length is not None:
        try:
            if int(content_length) > settings.voice.max_audio_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Audio payload is too large.",
                )

        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid content length.",
            ) from error

    try:
        audio = await request.body()

        result = await service.transcribe(
            audio,
            content_type=request.headers.get(
                "content-type",
                "",
            ),
            language=language,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise map_voice_error(error) from error

    return TranscriptionResponse(
        transcript=result.text,
        language=result.language,
        duration_seconds=result.duration_seconds,
    )


@router.post(
    "/synthesis",
)
async def synthesize_audio(
    payload: SynthesisRequest,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> Response:
    """Generate a non-persistent audio response using configured TTS."""

    del claims

    try:
        audio = await service.synthesize(
            payload.text,
            voice=payload.voice,
            language=payload.language,
            emotion=payload.emotion,
        )

    except Exception as error:
        raise map_voice_error(error) from error

    return Response(
        content=audio.data,
        media_type=audio.content_type,
        headers=_NO_STORE_HEADERS,
    )


@router.post(
    "/synthesis/stream",
)
async def stream_synthesis_audio(
    payload: SynthesisRequest,
    claims: Annotated[
        dict[str, object],
        Depends(require_voice_claims),
    ],
    service: Annotated[
        VoiceCompanionService,
        Depends(get_voice_service),
    ],
) -> StreamingResponse:
    """Stream generated TTS audio without persisting it."""

    del claims

    if not service.provider_status().get("tts_ready", False):
        async def empty_stream() -> AsyncIterator[bytes]:
            if False:
                yield b""

        return StreamingResponse(
            empty_stream(),
            media_type="audio/mpeg",
            headers={
                **_NO_STORE_HEADERS,
                "X-Voice-TTS-Fallback": "unavailable",
                "X-Voice-TTS-Provider": "unavailable",
            },
        )

    try:
        service.ensure_synthesis_available()

    except Exception as error:
        raise map_voice_error(error) from error

    async def audio_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in service.stream_synthesis(
                payload.text,
                voice=payload.voice,
                language=payload.language,
                emotion=payload.emotion,
            ):
                yield chunk

        except Exception as error:
            raise map_voice_error(error) from error

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={
            **_NO_STORE_HEADERS,
            "X-Voice-TTS-Provider": str(
                service.provider_status().get("tts_provider") or "unknown"
            ),
        },
    )
