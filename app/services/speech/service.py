"""Voice-companion orchestration with safe, provider-neutral boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID, uuid4

from app.core.enums import Workflow
from app.core.settings import settings
from app.core.logger import get_logger

from app.engines.character.engine import CharacterEngine
from app.engines.emotion.engine import (
    EmotionEngine,
    KeywordEmotionDetector,
)
from app.engines.emotion.acoustic import Wav2Vec2AcousticEmotionDetector
from app.engines.emotion.hybrid import HybridEmotionDetector
from app.engines.personality.engine import PersonalityEngine
from app.engines.relationship.engine import RelationshipEngine

from app.providers.manager.provider_manager import ProviderManager
from app.providers.elevenlabs.provider import ElevenLabsTTS

from app.runtime.companion_workflow import CompanionWorkflow
from app.runtime.exceptions import RuntimeCancelledError
from app.runtime.executor import RuntimeExecutor
from app.runtime.runtime import Runtime
from app.runtime.types import RuntimeTokenUsage

from app.services.speech.contracts import (
    AudioResult,
    SpeechToTextProvider,
    TextToSpeechProvider,
    TranscriptionResult,
)

from app.services.speech.exceptions import (
    VoiceProcessingError,
    VoiceProviderUnavailableError,
    VoiceSessionAccessError,
    VoiceSessionNotFoundError,
    VoiceTurnCancelledError,
    VoiceTurnInProgressError,
)

from app.services.speech.expression import (
    ExpressionDirective,
    ExpressionPlanner,
)

from app.services.speech.groq_stt import GroqWhisperSTT


logger = get_logger(__name__)


@dataclass(slots=True)
class VoiceSession:
    """Process-local session ownership and interruption state."""

    id: UUID
    owner_id: str
    created_at: datetime
    active_task: asyncio.Task[object] | None = None
    interruptions: int = 0


@dataclass(frozen=True, slots=True)
class VoiceTurnResult:
    """A completed companion turn returned to a client."""

    session_id: UUID
    request_id: UUID
    text: str
    expression: ExpressionDirective
    token_usage: RuntimeTokenUsage
    user_emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_source: str = "unknown"
    emotion_top: tuple[dict[str, object], ...] = ()
    llm_provider: str = "unknown"
    llm_model: str = "unknown"
    processing_ms: float = 0.0
    emotion_acoustic: dict[str, object] | None = None
    emotion_text: dict[str, object] | None = None
    emotion_policy: str = ""


@dataclass(frozen=True, slots=True)
class VoiceAudioTurnResult:
    """A completed audio turn with generated speech audio."""

    turn: VoiceTurnResult
    audio: AudioResult
    transcript: str


@dataclass(frozen=True, slots=True)
class VoiceAudioStreamTurnResult:
    """A voice turn whose TTS remains active as an HTTP audio stream."""

    turn: VoiceTurnResult
    transcript: str
    audio_stream: AsyncIterator[bytes]


class VoiceSessionRegistry:
    """Coordinate same-process session ownership and barge-in requests."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, VoiceSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, owner_id: str) -> VoiceSession:
        """Create an idle session owned by the authenticated subject."""

        session = VoiceSession(
            id=uuid4(),
            owner_id=owner_id,
            created_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            self._sessions[session.id] = session

        return session

    async def get_owned(
        self,
        session_id: UUID,
        owner_id: str,
    ) -> VoiceSession:
        """Return a session only when it belongs to the caller."""

        async with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                raise VoiceSessionNotFoundError()

            if session.owner_id != owner_id:
                raise VoiceSessionAccessError()

            return session

    async def begin_turn(
        self,
        session_id: UUID,
        owner_id: str,
        task: asyncio.Task[object] | None,
    ) -> None:
        """Mark the current task as a session's only active voice turn."""

        async with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                raise VoiceSessionNotFoundError()

            if session.owner_id != owner_id:
                raise VoiceSessionAccessError()

            if (
                session.active_task is not None
                and not session.active_task.done()
            ):
                raise VoiceTurnInProgressError()

            session.active_task = task

    async def finish_turn(
        self,
        session_id: UUID,
        task: asyncio.Task[object] | None,
    ) -> None:
        """Clear a completed task without clobbering a newer turn."""

        async with self._lock:
            session = self._sessions.get(session_id)

            if session is not None and session.active_task is task:
                session.active_task = None

    async def cancel(
        self,
        session_id: UUID,
        owner_id: str,
    ) -> bool:
        """Cancel an active turn after enforcing object ownership."""

        async with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                raise VoiceSessionNotFoundError()

            if session.owner_id != owner_id:
                raise VoiceSessionAccessError()

            task = session.active_task

            if task is None or task.done():
                return False

            session.interruptions += 1
            task.cancel()
            return True


    async def metrics(
        self,
        session_id: UUID,
        owner_id: str,
    ) -> dict[str, object]:
        """Return non-sensitive session metrics for the owner."""

        session = await self.get_owned(session_id, owner_id)
        return {
            "session_id": session.id,
            "active": (
                session.active_task is not None
                and not session.active_task.done()
            ),
            "interruptions": session.interruptions,
        }


class VoiceCompanionService:
    """Coordinate transcript, companion runtime, expression, and speech."""

    _ALLOWED_AUDIO_TYPES = frozenset(
        {
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/ogg",
            "audio/webm",
            "audio/flac",
            "audio/x-flac",
        }
    )

    def __init__(
        self,
        *,
        runtime: Runtime,
        provider_manager: ProviderManager | None = None,
        transcriber: SpeechToTextProvider | None = None,
        synthesizer: TextToSpeechProvider | None = None,
        sessions: VoiceSessionRegistry | None = None,
        expression_planner: ExpressionPlanner | None = None,
        acoustic_emotion_detector: Wav2Vec2AcousticEmotionDetector | None = None,
    ) -> None:
        self._runtime = runtime
        self._provider_manager = provider_manager
        self._transcriber = transcriber
        self._synthesizer = synthesizer
        self._sessions = sessions or VoiceSessionRegistry()
        self._expression_planner = (
            expression_planner or ExpressionPlanner()
        )
        self._acoustic_emotion_detector = acoustic_emotion_detector

    async def create_session(
        self,
        owner_id: str,
    ) -> VoiceSession:
        """Create a new caller-owned voice session."""

        return await self._sessions.create(owner_id)

    async def process_turn(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        transcript: str,
        language: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> VoiceTurnResult:
        """Run an authenticated transcript through the companion runtime."""

        normalized_transcript = transcript.strip()
        started_at = perf_counter()

        if not normalized_transcript:
            raise ValueError("Transcript cannot be empty.")

        if (
            len(normalized_transcript)
            > settings.voice.max_transcript_characters
        ):
            raise ValueError(
                "Transcript exceeds the configured size limit."
            )

        task = asyncio.current_task()

        await self._sessions.begin_turn(
            session_id,
            owner_id,
            task,
        )

        runtime_metadata: dict[str, object] = {
            "owner_id": owner_id,
            "language": language,
        }
        if metadata:
            runtime_metadata.update(metadata)

        try:
            result = await asyncio.wait_for(
                self._runtime.execute(
                    workflow=Workflow.VOICE,
                    input=normalized_transcript,
                    session_id=session_id,
                    metadata=runtime_metadata,
                ),
                timeout=settings.voice.turn_timeout,
            )

        except (
            asyncio.CancelledError,
            RuntimeCancelledError,
        ) as error:
            raise VoiceTurnCancelledError() from error

        except TimeoutError as error:
            raise VoiceProcessingError() from error

        except Exception as error:
            logger.exception("Voice runtime turn failed: %s", error)
            detail = str(error).strip() or error.__class__.__name__
            raise VoiceProcessingError(f"Runtime failed: {detail}") from error

        finally:
            await self._sessions.finish_turn(
                session_id,
                task,
            )

        if not result.success or not result.output:
            raise VoiceProcessingError()

        emotion_metadata = result.metadata.get(
            "emotion",
            {},
        )

        user_emotion = "neutral"
        emotion_confidence = 0.0
        emotion_source = "unknown"
        emotion_top: tuple[dict[str, object], ...] = ()
        emotion_acoustic: dict[str, object] | None = None
        emotion_text: dict[str, object] | None = None

        if isinstance(emotion_metadata, dict):
            candidate = emotion_metadata.get("emotion")
            if isinstance(candidate, str):
                user_emotion = candidate

            confidence = emotion_metadata.get("confidence")
            if isinstance(confidence, (int, float)):
                emotion_confidence = float(confidence)

            source = emotion_metadata.get("source")
            if isinstance(source, str):
                emotion_source = source

            top = emotion_metadata.get("top_emotions")
            if isinstance(top, list):
                emotion_top = tuple(
                    item for item in top
                    if isinstance(item, dict)
                )

            acoustic = emotion_metadata.get("acoustic")
            if isinstance(acoustic, dict):
                emotion_acoustic = {
                    key: acoustic[key]
                    for key in ("emotion", "confidence", "model")
                    if key in acoustic
                }

            text_prediction = emotion_metadata.get("text")
            if isinstance(text_prediction, dict):
                emotion_text = {
                    key: text_prediction[key]
                    for key in ("emotion", "confidence", "source")
                    if key in text_prediction
                }
            elif "emotion" in emotion_metadata:
                emotion_text = {
                    "emotion": emotion_metadata.get("emotion"),
                    "confidence": emotion_metadata.get("confidence", 0.0),
                    "source": emotion_metadata.get("source", "unknown"),
                }

        expression = self._expression_planner.plan(
            user_emotion=user_emotion,
            response_text=result.output,
        )
        llm_provider = result.metadata.get("llm_provider", "unknown")
        llm_model = result.metadata.get("llm_model", "unknown")

        return VoiceTurnResult(
            session_id=session_id,
            request_id=result.request_id,
            text=result.output,
            expression=expression,
            token_usage=result.token_usage,
            user_emotion=user_emotion,
            emotion_confidence=emotion_confidence,
            emotion_source=emotion_source,
            emotion_top=emotion_top,
            llm_provider=str(llm_provider),
            llm_model=str(llm_model),
            processing_ms=(perf_counter() - started_at) * 1000,
            emotion_acoustic=emotion_acoustic,
            emotion_text=emotion_text,
            emotion_policy=str(
                result.metadata.get("emotion_policy", "")
            ),
        )

    async def process_audio_turn(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        audio: bytes,
        content_type: str,
        language: str | None = None,
        voice: str | None = None,
    ) -> VoiceAudioTurnResult:
        """
        Run one complete voice turn:

        audio -> Groq Whisper -> Aiko -> ElevenLabs.
        """

        transcription = await self.transcribe(
            audio,
            content_type=content_type,
            language=language,
        )

        transcript = transcription.text.strip()

        if not transcript:
            raise VoiceProcessingError()

        turn = await self.process_turn(
            session_id=session_id,
            owner_id=owner_id,
            transcript=transcript,
            language=language or transcription.language,
            metadata={
                "audio_bytes": audio,
                "audio_content_type": content_type,
                "audio_duration_seconds": transcription.duration_seconds,
            },
        )

        audio_result = await self.synthesize(
            turn.text,
            voice=voice,
            language=language or transcription.language,
            emotion=turn.expression.voice_style,
        )

        return VoiceAudioTurnResult(
            turn=turn,
            audio=audio_result,
            transcript=transcript,
        )


    async def process_audio_stream_turn(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        audio: bytes,
        content_type: str,
        language: str | None = None,
        voice: str | None = None,
    ) -> VoiceAudioStreamTurnResult:
        """Run STT -> companion runtime -> streaming TTS.

        The HTTP request remains the active session task while TTS chunks are
        streamed. Therefore the existing cancel_turn() endpoint can interrupt
        a live response during synthesis/streaming.
        """

        transcription = await self.transcribe(
            audio,
            content_type=content_type,
            language=language,
        )

        transcript = transcription.text.strip()

        if not transcript:
            raise VoiceProcessingError()

        turn = await self.process_turn(
            session_id=session_id,
            owner_id=owner_id,
            transcript=transcript,
            language=language or transcription.language,
            metadata={
                "audio_bytes": audio,
                "audio_content_type": content_type,
                "audio_duration_seconds": transcription.duration_seconds,
            },
        )

        async def audio_stream() -> AsyncIterator[bytes]:
            # StreamingResponse consumes this generator in its own task.
            # Register that task so the /cancel endpoint cancels the task that
            # is actually producing audio rather than the handler task that
            # already returned the StreamingResponse object.
            task = asyncio.current_task()

            if task is None:
                raise VoiceProcessingError()

            await self._sessions.begin_turn(
                session_id,
                owner_id,
                task,
            )

            try:
                async for chunk in self.stream_synthesis(
                    turn.text,
                    voice=voice,
                    language=language or transcription.language,
                    emotion=turn.expression.voice_style,
                ):
                    yield chunk
            except asyncio.CancelledError as error:
                raise VoiceTurnCancelledError() from error
            finally:
                await self._sessions.finish_turn(
                    session_id,
                    task,
                )

        return VoiceAudioStreamTurnResult(
            turn=turn,
            transcript=transcript,
            audio_stream=audio_stream(),
        )

    async def transcribe_and_process_turn(
        self,
        *,
        session_id: UUID,
        owner_id: str,
        audio: bytes,
        content_type: str,
        language: str | None = None,
    ) -> tuple[VoiceTurnResult, str, TranscriptionResult]:
        """Run STT and the conversational runtime without requiring TTS."""

        transcription = await self.transcribe(
            audio,
            content_type=content_type,
            language=language,
        )
        transcript = transcription.text.strip()
        if not transcript:
            raise VoiceProcessingError("STT returned an empty transcript.")

        turn = await self.process_turn(
            session_id=session_id,
            owner_id=owner_id,
            transcript=transcript,
            language=language or transcription.language,
            metadata={
                "audio_bytes": audio,
                "audio_content_type": content_type,
                "audio_duration_seconds": transcription.duration_seconds,
            },
        )
        return turn, transcript, transcription

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe bounded in-memory audio without writing it to disk."""

        if not audio:
            raise ValueError("Audio payload cannot be empty.")

        if len(audio) > settings.voice.max_audio_bytes:
            raise ValueError(
                "Audio payload exceeds the configured size limit."
            )

        normalized_content_type = (
            content_type.split(";", 1)[0]
            .strip()
            .lower()
        )

        if normalized_content_type not in self._ALLOWED_AUDIO_TYPES:
            raise ValueError(
                "Unsupported audio content type."
            )

        if self._transcriber is None:
            raise VoiceProviderUnavailableError()

        try:
            async with asyncio.timeout(
                settings.voice.voice_timeout
            ):
                return await self._transcriber.transcribe(
                    audio,
                    content_type=normalized_content_type,
                    language=language,
                )

        except TimeoutError as error:
            logger.error("Voice STT timed out after %.1fs", settings.voice.voice_timeout)
            raise VoiceProcessingError("STT provider timed out.") from error

        except VoiceProviderUnavailableError:
            raise

        except Exception as error:
            logger.exception("Voice STT failed: %s", error)
            detail = str(error).strip() or error.__class__.__name__
            raise VoiceProcessingError(f"STT provider error: {detail}") from error

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AudioResult:
        """Synthesize a bounded response in memory."""

        self._validate_synthesis_text(text)

        synthesizer = self._require_synthesizer()

        try:
            async with asyncio.timeout(
                settings.voice.voice_timeout
            ):
                return await synthesizer.synthesize(
                    text,
                    voice=voice,
                    language=language,
                    emotion=emotion,
                )

        except TimeoutError as error:
            logger.exception("Voice TTS timed out")
            raise VoiceProcessingError("TTS provider timed out.") from error

        except Exception as error:
            logger.exception("Voice TTS failed: %s", error)
            detail = str(error).strip() or error.__class__.__name__
            raise VoiceProcessingError(f"TTS provider error: {detail}") from error

    async def stream_synthesis(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield provider audio chunks while bounding total stream duration."""

        self._validate_synthesis_text(text)

        synthesizer = self._require_synthesizer()

        try:
            async with asyncio.timeout(
                settings.voice.voice_timeout
            ):
                async for chunk in synthesizer.stream(
                    text,
                    voice=voice,
                    language=language,
                    emotion=emotion,
                ):
                    if not isinstance(chunk, bytes):
                        raise VoiceProcessingError()

                    yield chunk

        except TimeoutError as error:
            logger.exception("Voice streaming TTS timed out")
            raise VoiceProcessingError("TTS streaming timed out.") from error

        except VoiceProcessingError:
            raise

        except Exception as error:
            logger.exception("Voice streaming TTS failed: %s", error)
            detail = str(error).strip() or error.__class__.__name__
            raise VoiceProcessingError(f"TTS streaming error: {detail}") from error

    async def session_metrics(
        self,
        session_id: UUID,
        owner_id: str,
    ) -> dict[str, object]:
        """Return safe interruption metrics for the caller's own session."""

        return await self._sessions.metrics(
            session_id,
            owner_id,
        )

    def provider_status(self) -> dict[str, object]:
        """Return safe voice/provider diagnostics without secrets."""

        providers = list(self._provider_manager.providers.keys()) if self._provider_manager is not None else []
        return {
            "llm_providers": providers,
            "configured_llm": settings.ai.default_provider,
            "configured_llm_model": getattr(
                getattr(settings.ai, settings.ai.default_provider, None),
                "model",
                "unknown",
            ),
            "stt_ready": self._transcriber is not None,
            "tts_ready": self._synthesizer is not None,
            "tts_provider": getattr(self._synthesizer, "name", None),
            "stt_provider": getattr(
                self._transcriber,
                "name",
                settings.voice.stt_provider if self._transcriber else None,
            ),
            "development_fallback": "demo_fallback" in providers,
        }

    def emotion_status(self) -> dict[str, object]:
        """Return safe diagnostics for the acoustic emotion research path."""

        if self._acoustic_emotion_detector is None:
            return {
                "enabled": False,
                "status": "disabled",
            }

        return self._acoustic_emotion_detector.status()

    async def warmup_emotion_model(self) -> dict[str, object]:
        """Load the acoustic emotion model before a live demonstration."""

        if self._acoustic_emotion_detector is None:
            return self.emotion_status()

        return await self._acoustic_emotion_detector.warmup()

    async def cancel_turn(
        self,
        session_id: UUID,
        owner_id: str,
    ) -> bool:
        """Request a barge-in cancellation for the caller's active turn."""

        return await self._sessions.cancel(
            session_id,
            owner_id,
        )

    async def close(self) -> None:
        """Release providers owned by the default service composition."""

        if self._provider_manager is not None:
            await self._provider_manager.close()

        if self._transcriber is not None:
            close = getattr(
                self._transcriber,
                "close",
                None,
            )

            if close is not None:
                await close()

        if self._synthesizer is not None:
            close = getattr(
                self._synthesizer,
                "close",
                None,
            )

            if close is not None:
                result = close()

                if hasattr(result, "__await__"):
                    await result

    def ensure_synthesis_available(self) -> None:
        """Check TTS configuration before an HTTP streaming response starts."""

        self._require_synthesizer()

    def _require_synthesizer(
        self,
    ) -> TextToSpeechProvider:
        if self._synthesizer is None:
            raise VoiceProviderUnavailableError()

        return self._synthesizer

    @staticmethod
    def _validate_synthesis_text(
        text: str,
    ) -> None:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        if len(text) > settings.voice.max_transcript_characters:
            raise ValueError(
                "Text exceeds the configured size limit."
            )


def build_voice_companion_service() -> VoiceCompanionService:
    """Build the production composition with configured STT and TTS."""

    provider_manager = ProviderManager()

    character_engine = CharacterEngine()

    acoustic_emotion_detector = Wav2Vec2AcousticEmotionDetector(
        model_name=settings.voice.emotion_model,
        enabled=(
            settings.features.emotion
            and settings.voice.emotion_acoustic_enabled
        ),
        min_confidence=settings.voice.emotion_min_confidence,
    )

    workflow = CompanionWorkflow(
        provider_manager=provider_manager,
        character_engine=character_engine,
        emotion_engine=EmotionEngine(
            detector=HybridEmotionDetector(
                acoustic=acoustic_emotion_detector,
                lexical=KeywordEmotionDetector(),
            ),
        ),
        personality_engine=PersonalityEngine(),
        relationship_engine=RelationshipEngine(),
    )

    runtime = Runtime(
        RuntimeExecutor(
            workflow,
            timeout=float(settings.runtime.timeout),
        )
    )

    transcriber: SpeechToTextProvider | None = None

    if settings.voice.stt_provider.lower() == "groq":
        if settings.ai.groq.api_key:
            try:
                transcriber = GroqWhisperSTT()
            except (RuntimeError, ImportError) as error:
                logger.warning("Groq STT unavailable; continuing without STT: %s", error)

    synthesizer: TextToSpeechProvider | None = None

    if settings.voice.tts_provider.lower() == "elevenlabs":
        if (
            settings.elevenlabs.api_key
            and settings.elevenlabs.voice_id
        ):
            try:
                synthesizer = ElevenLabsTTS()
            except (RuntimeError, ImportError) as error:
                logger.warning("ElevenLabs TTS unavailable; continuing without TTS: %s", error)

    return VoiceCompanionService(
        runtime=runtime,
        provider_manager=provider_manager,
        transcriber=transcriber,
        synthesizer=synthesizer,
        acoustic_emotion_detector=acoustic_emotion_detector,
    )
