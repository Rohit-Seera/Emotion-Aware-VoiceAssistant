"""API behavior tests for authenticated voice-companion endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.voice import router
from app.core.enums import RuntimeStatus, Workflow
from app.runtime.result import RuntimeResult
from app.services.speech.service import VoiceCompanionService


class FakeRuntime:
    """Runtime double that avoids any external LLM request."""

    async def execute(self, **kwargs: object) -> RuntimeResult:
        return RuntimeResult(
            request_id=uuid4(),
            workflow=Workflow.VOICE,
            status=RuntimeStatus.COMPLETED,
            output="Welcome back, Rohit.",
            metadata={
                "llm_provider": "test-provider",
                "llm_model": "test-model",
            },
        )


def create_test_app() -> FastAPI:
    """Create a voice API app with a deterministic runtime composition."""

    app = FastAPI()
    app.include_router(router)
    app.state.voice_companion_service = VoiceCompanionService(
        runtime=FakeRuntime(),
    )
    return app


def authorization_header(subject: str) -> dict[str, str]:
    """Create a test-only user token with the existing permission model."""

    from app.security.jwt import jwt_service

    token = jwt_service.create_access_token(
        subject,
        extra_claims={"role": "user"},
    )
    return {"Authorization": f"Bearer {token}"}


def test_voice_session_and_turn_are_authenticated() -> None:
    """A user can create and use only their own session."""

    client = TestClient(create_test_app())
    headers = authorization_header("user-1")

    created = client.post("/voice/sessions", headers=headers)

    assert created.status_code == 201
    session_id = created.json()["session_id"]

    turn = client.post(
        f"/voice/sessions/{session_id}/turns",
        headers=headers,
        json={"transcript": "Hello Aiko"},
    )

    assert turn.status_code == 200
    assert turn.json()["text"] == "Welcome back, Rohit."
    assert turn.json()["expression"]["lip_sync"] == "speech"
    assert turn.json()["llm_provider"] == "test-provider"
    assert turn.json()["llm_model"] == "test-model"
    assert turn.json()["processing_ms"] >= 0


def test_voice_endpoint_rejects_missing_authentication() -> None:
    """Voice operations must not be exposed as anonymous endpoints."""

    client = TestClient(create_test_app())

    response = client.post("/voice/sessions")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_voice_session_is_hidden_from_another_user() -> None:
    """A session ID must not become an IDOR vector."""

    client = TestClient(create_test_app())
    first_headers = authorization_header("user-1")
    second_headers = authorization_header("user-2")
    created = client.post("/voice/sessions", headers=first_headers)
    session_id = created.json()["session_id"]

    response = client.post(
        f"/voice/sessions/{session_id}/turns",
        headers=second_headers,
        json={"transcript": "Open this session."},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Voice session not found."
