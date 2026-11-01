"""Application lifecycle."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logger import get_logger
from app.core.settings import settings

logger = get_logger(__name__)


async def startup(app: FastAPI) -> None:
    """Run application startup tasks."""

    logger.info("Starting Emotion Aware Voice Assistant backend...")

    settings.validate_runtime_configuration()

    from app.services.speech.service import build_voice_companion_service

    app.state.voice_companion_service = (
        build_voice_companion_service()
    )

    service = app.state.voice_companion_service
    if settings.voice.emotion_preload:
        try:
            await service.warmup_emotion_model()
            logger.info("Acoustic emotion model preloaded.")
        except Exception as error:
            logger.warning(
                "Acoustic emotion model preload failed; text fallback remains available: %s",
                error,
            )

    logger.info("Startup completed.")


async def shutdown(app: FastAPI) -> None:
    """Run application shutdown tasks."""

    logger.info("Shutting down Emotion Aware Voice Assistant backend...")

    service = getattr(
        app.state,
        "voice_companion_service",
        None,
    )

    if service is not None:
        await service.close()

    # Close optional infrastructure clients when they are actually available.
    # The local voice demo does not require PostgreSQL/Redis/Qdrant just to
    # start, so shutdown must not fail when those optional dependencies are
    # absent.
    try:
        from app.infrastructure.cache.client import redis_client
        await redis_client.aclose()
    except Exception as error:
        logger.debug("Redis shutdown skipped: %s", error)

    try:
        from app.infrastructure.vectorstore.client import qdrant_client
        await qdrant_client.close()
    except Exception as error:
        logger.debug("Qdrant shutdown skipped: %s", error)

    try:
        from app.infrastructure.database.engine import engine
        await engine.dispose()
    except Exception as error:
        logger.debug("Database shutdown skipped: %s", error)

    logger.info("Shutdown completed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""

    await startup(app)

    yield

    await shutdown(app)
