"""Emotion Aware Voice Assistant backend application."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse

from app.api.router import router as api_router
from app.core.lifecycle import lifespan


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    application = FastAPI(
        title="Emotion Aware Voice Assistant",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Open the local research demo from the root URL."""
        return RedirectResponse(url="/api/v1/demo", status_code=307)

    @application.get("/demo", include_in_schema=False)
    async def demo_shortcut() -> RedirectResponse:
        """Convenience shortcut for the local research demo."""
        return RedirectResponse(url="/api/v1/demo", status_code=307)

    @application.get("/status", include_in_schema=False)
    async def status() -> JSONResponse:
        """Simple startup/status endpoint for Windows demo troubleshooting."""
        return JSONResponse({
            "status": "running",
            "demo": "/api/v1/demo",
            "docs": "/docs",
            "health": "/api/v1/health/live",
        })

    application.include_router(api_router)
    return application


app = create_application()
