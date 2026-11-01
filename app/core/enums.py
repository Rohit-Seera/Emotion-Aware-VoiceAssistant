"""Shared application enums."""

from enum import Enum


class StrEnum(str, Enum):
    """Base enum that behaves like a string."""

    def __str__(self) -> str:
        return self.value


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class BuildStage(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    RC = "release-candidate"
    STABLE = "stable"


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OPENROUTER = "openrouter"


class Workflow(StrEnum):
    CHAT = "chat"
    VOICE = "voice"
    RESEARCH = "research"
    AUTOMATION = "automation"


class RuntimeStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"