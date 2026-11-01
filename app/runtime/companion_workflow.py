"""Default voice-companion workflow built from the runtime's existing nodes."""

from __future__ import annotations

from app.engines.character.engine import CharacterEngine
from app.engines.emotion.engine import EmotionEngine
from app.engines.personality.engine import PersonalityEngine
from app.engines.relationship.engine import RelationshipEngine
from app.providers.manager.provider_manager import ProviderManager
from app.runtime.context import RuntimeContext
from app.runtime.nodes.context_builder import ContextBuilderNode
from app.runtime.nodes.detect_emotion import DetectEmotionNode
from app.runtime.nodes.finish import FinishNode
from app.runtime.nodes.llm import LLMNode
from app.runtime.nodes.model_router import ModelRouterNode
from app.runtime.nodes.personality import PersonalityNode
from app.runtime.nodes.prompt_builder import PromptBuilderNode
from app.runtime.nodes.relationship import RelationshipNode
from app.runtime.state import RuntimeState


class CompanionWorkflow:
    """Execute the default companion path through established runtime nodes."""

    def __init__(
        self,
        *,
        provider_manager: ProviderManager,
        character_engine: CharacterEngine,
        emotion_engine: EmotionEngine,
        personality_engine: PersonalityEngine,
        relationship_engine: RelationshipEngine,
    ) -> None:
        self._character = character_engine
        self._context_builder = ContextBuilderNode()
        self._emotion = DetectEmotionNode(emotion_engine)
        self._personality = PersonalityNode(personality_engine)
        self._relationship = RelationshipNode(relationship_engine)
        self._model_router = ModelRouterNode(max_tokens=512)
        self._prompt_builder = PromptBuilderNode()
        self._llm = LLMNode(provider_manager)
        self._finish = FinishNode()

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Run a complete conversation without parallel orchestration."""

        await self._context_builder.execute(state, context)
        context.metadata["character"] = self._character.get_character()
        await self._emotion.execute(state, context)
        await self._personality.execute(state, context)
        await self._relationship.execute(state, context)
        await self._model_router.execute(state, context)
        await self._prompt_builder.execute(state, context)
        await self._llm.execute(state, context)
        await self._finish.execute(state, context)

        # RuntimeResult serializes state metadata, not RuntimeContext. Only
        # expose structured data that the client needs to animate a response.
        for name in (
            "character",
            "emotion",
            "personality",
            "relationship",
            "model_route",
            "llm_provider",
            "llm_model",
            "emotion_policy",
        ):
            value = context.metadata.get(name)
            if value is not None:
                state.metadata[name] = value

        return state
