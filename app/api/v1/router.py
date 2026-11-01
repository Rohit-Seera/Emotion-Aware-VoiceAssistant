"""Version 1 API router."""

from fastapi import APIRouter

from app.api.v1.demo import router as demo_router
from app.api.v1.health import router as health_router
from app.api.v1.voice import router as voice_router

router = APIRouter(
    prefix="/v1",
)

router.include_router(health_router)
router.include_router(demo_router)
router.include_router(voice_router)
