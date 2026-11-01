"""Build the prompt used by downstream LLM nodes."""

from __future__ import annotations

from typing import Any

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class PromptBuilderNode:
    """Build a structured prompt from runtime context."""

    @staticmethod
    def response_policy_for(emotion: str) -> str:
        """Return the brief emotional response policy for a label."""

        return {
            "sad": "Meet the user's sadness with a softer, lower-energy and caring tone. Acknowledge it sincerely, avoid cheerleading or forced optimism, and offer one gentle next step without mirroring hopelessness.",
            "anxious": "Use calm, reassuring language and reduce cognitive load; give clear next steps.",
            "fearful": "Reassure the user, avoid alarmist phrasing, and keep the response grounded and clear.",
            "angry": "Stay calm and non-defensive; validate the frustration and focus on a constructive next step.",
            "frustrated": "Acknowledge the difficulty, stay patient, and give a concise actionable path forward.",
            "happy": "Match the positive energy naturally without becoming exaggerated or distracting.",
            "neutral": "Use a warm, natural conversational tone.",
        }.get(
            emotion.casefold(),
            "Use a warm, natural conversational tone appropriate to the user's state.",
        )

    def build_prompt(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> str:
        """Build a prompt from the current runtime information."""

        sections: list[str] = []

        character = context.metadata.get("character")

        if isinstance(character, dict):
            name = character.get("name")
            description = character.get("description")

            if name:
                sections.append(f"Character: {name}")

            if description:
                sections.append(
                    f"Character identity: {description}"
                )

        sections.append(
            f"User: {state.input}"
        )

        sections.append(
            "Spoken response style: Reply like a warm, emotionally attentive voice companion. "
            "Answer directly in 1–3 natural sentences (usually under 45 words), with smooth, "
            "connected phrasing that is easy to say aloud. Use simple words, natural contractions, "
            "and gentle emotional expression; avoid headings, lists, repetition, stage directions, "
            "and choppy fragments. Never claim to know feelings with certainty."
        )

        personality = context.metadata.get(
            "personality",
        )

        if personality:
            sections.append(
                f"Personality: {personality}"
            )

        relationship = context.metadata.get(
            "relationship",
        )

        if relationship:
            sections.append(
                f"Relationship: {relationship}"
            )

        emotion = context.metadata.get(
            "emotion",
        )

        if isinstance(emotion, dict):
            detected = emotion.get(
                "emotion",
                emotion.get("label", "neutral"),
            )
            confidence = emotion.get("confidence", 0.0)
            source = emotion.get("source", "unknown")
            sections.append(
                "Emotion: "
                f"{detected} (confidence={confidence}, source={source})"
            )

            response_policy = self.response_policy_for(str(detected))
            sections.append(
                f"Emotion-aware response policy: {response_policy}"
            )
        elif emotion:
            sections.append(
                f"Emotion: {emotion}"
            )

        if state.memories:
            sections.append(
                f"Memories: {state.memories}"
            )

        return "\n\n".join(sections)

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Build and store the prompt."""

        prompt = self.build_prompt(
            state,
            context,
        )

        context.metadata["prompt"] = prompt
        emotion = context.metadata.get("emotion")
        label = (
            str(emotion.get("emotion", "neutral"))
            if isinstance(emotion, dict)
            else "neutral"
        )
        context.metadata["emotion_policy"] = self.response_policy_for(label)

        return state
