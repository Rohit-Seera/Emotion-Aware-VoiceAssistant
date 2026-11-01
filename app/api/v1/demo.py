"""Local development UI for the Aiko voice demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.settings import settings
from app.security.jwt import jwt_service

router = APIRouter(prefix="/demo", tags=["demo"])

_DEMO_USER = "demo-user"
_DEMO_HTML = Path(__file__).with_name("demo.html")


class DemoTokenResponse(BaseModel):
    """Short-lived token used only by the local development demo."""

    access_token: str
    token_type: str = "bearer"


def _ensure_development() -> None:
    """Keep the demo surface out of production deployments."""

    if settings.is_production:
        raise HTTPException(
            status_code=404,
            detail="Demo is available only in development.",
        )


@router.get("", include_in_schema=False)
async def demo_page() -> FileResponse:
    """Serve the local Aiko voice-companion demo."""

    _ensure_development()
    return FileResponse(
        _DEMO_HTML,
        media_type="text/html; charset=utf-8",
    )


@router.get("/token", response_model=DemoTokenResponse, include_in_schema=False)
async def demo_token() -> DemoTokenResponse:
    """Issue a development-only user token for the demo browser."""

    _ensure_development()

    return DemoTokenResponse(
        access_token=jwt_service.create_access_token(
            _DEMO_USER,
            extra_claims={"role": "user"},
        ),
    )


@router.get("/diagnostics", include_in_schema=False)
async def demo_diagnostics(request: Request) -> dict[str, object]:
    """Return safe demo diagnostics without exposing credentials."""

    _ensure_development()
    service = getattr(request.app.state, "voice_companion_service", None)
    if service is None:
        return {"status": "starting"}
    return {
        "status": "ready",
        "providers": service.provider_status(),
        "emotion": service.emotion_status(),
    }
