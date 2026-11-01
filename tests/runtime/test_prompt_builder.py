"""Tests for the prompt-builder runtime node."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.prompt_builder import PromptBuilderNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="How are you, Weings?",
        memories=[
            "User likes Python.",
            "User is building Emotion Aware Voice Assistant.",
        ],
    )


@pytest.fixture
def context(
    state: RuntimeState,
) -> RuntimeContext:
    """Return a sample runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "personality": {
                "tone": "warm",
            },
            "relationship": {
                "level": 3,
            },
            "emotion": {
                "label": "happy",
            },
        },
    )


def test_build_prompt_contains_user_input(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Prompt should contain the user's input."""

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "How are you, Weings?" in prompt


def test_build_prompt_requests_natural_concise_spoken_replies(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    prompt = PromptBuilderNode().build_prompt(state, context)

    assert "warm, emotionally attentive voice companion" in prompt
    assert "1–3 natural sentences" in prompt
    assert "avoid headings, lists" in prompt


def test_sad_response_policy_matches_tone_without_forced_optimism() -> None:
    policy = PromptBuilderNode.response_policy_for("sad")

    assert "lower-energy" in policy
    assert "avoid cheerleading" in policy


def test_build_prompt_contains_personality(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Prompt should contain personality context."""

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "Personality:" in prompt
    assert "warm" in prompt


def test_build_prompt_contains_relationship(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Prompt should contain relationship context."""

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "Relationship:" in prompt
    assert "level" in prompt


def test_build_prompt_contains_emotion(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Prompt should contain emotion context."""

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "Emotion:" in prompt
    assert "happy" in prompt


def test_build_prompt_contains_memories(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Prompt should contain retrieved memories."""

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "Memories:" in prompt
    assert "User likes Python." in prompt
    assert "User is building Emotion Aware Voice Assistant." in prompt


@pytest.mark.asyncio
async def test_execute_stores_prompt(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """execute should store the generated prompt."""

    node = PromptBuilderNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert "prompt" in context.metadata
    assert context.metadata["prompt"] == (
        node.build_prompt(
            state,
            context,
        )
    )


def test_build_prompt_handles_missing_context(
    state: RuntimeState,
) -> None:
    """Prompt should work when optional context is unavailable."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )

    node = PromptBuilderNode()

    prompt = node.build_prompt(
        state,
        context,
    )

    assert "How are you, Weings?" in prompt
    assert "Personality:" not in prompt
    assert "Relationship:" not in prompt
    assert "Emotion:" not in prompt
