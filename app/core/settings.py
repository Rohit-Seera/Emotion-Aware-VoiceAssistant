"""Application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import Environment


class BaseConfig(BaseSettings):
    """Base settings configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

class AppSettings(BaseConfig):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        extra="ignore",
    )

    name: str = "Emotion Aware Voice Assistant"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True


class DatabaseSettings(BaseConfig):
    """PostgreSQL configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 5432

    username: str = "postgres"
    password: str = "postgres"

    database: str = "emotion_aware_assistant"

    echo: bool = False

    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    pool_recycle: int = 3600

    @property
    def url(self) -> str:
        """Return the async PostgreSQL connection URL."""

        return (
            f"postgresql+asyncpg://"
            f"{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseConfig):
    """Redis configuration."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 6379

    password: str | None = None

    db: int = 0

    decode_responses: bool = False
    ssl: bool = False
    socket_timeout: int = 5

    @property
    def url(self) -> str:
        """Return the Redis connection URL."""

        if self.password:
            return (
                f"redis://:{self.password}"
                f"@{self.host}:{self.port}/{self.db}"
            )

        return f"redis://{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseConfig):
    """Qdrant configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 6333

    api_key: str | None = None

    collection: str = "assistant-memory"

    https: bool = False
    prefer_grpc: bool = False

    @property
    def url(self) -> str:
        """Return the Qdrant connection URL."""

        protocol = "https" if self.https else "http"

        return f"{protocol}://{self.host}:{self.port}"


class ProviderSettings(BaseConfig):
    """Base provider configuration."""

    enabled: bool = True

    api_key: str = ""

    base_url: str | None = None

    model: str = ""

    timeout: int = 60

    max_retries: int = 3

    temperature: float = 0.7

    max_tokens: int = 4096


class OpenAISettings(ProviderSettings):
    """OpenAI configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        extra="ignore",
    )

    model: str = "gpt-5"


class AnthropicSettings(ProviderSettings):
    """Anthropic configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        extra="ignore",
    )

    model: str = "claude-sonnet-4"


class GeminiSettings(ProviderSettings):
    """Google Gemini configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        extra="ignore",
    )

    model: str = "gemini-2.5-pro"


class GroqSettings(ProviderSettings):
    """Groq configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        extra="ignore",
    )

    model: str = "llama-3.3-70b-versatile"


class DeepSeekSettings(ProviderSettings):
    """DeepSeek configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_",
        extra="ignore",
    )

    model: str = "deepseek-v4-flash"


class QwenSettings(ProviderSettings):
    """Qwen configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QWEN_",
        extra="ignore",
    )

    model: str = "qwen-plus"


class OpenRouterSettings(ProviderSettings):
    """OpenRouter configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        extra="ignore",
    )

    base_url: str = "https://openrouter.ai/api/v1"

    embedding_model: str = (
        "openai/text-embedding-3-small"
    )


class VoyageSettings(ProviderSettings):
    """Voyage AI embedding configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VOYAGE_",
        extra="ignore",
    )

    model: str = "voyage-4-large"


class ElevenLabsSettings(BaseConfig):
    """ElevenLabs text-to-speech configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ELEVENLABS_",
        extra="ignore",
    )

    api_key: str = ""
    voice_id: str = ""
    model: str = "eleven_flash_v2_5"
    timeout: int = 30


class VoiceSettings(BaseConfig):
    """Voice configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VOICE_",
        extra="ignore",
    )

    enabled: bool = True

    language: str = "en"

    sample_rate: int = 16000

    streaming: bool = True

    silence_timeout: float = 1.5

    interrupt_enabled: bool = True

    tts_provider: str = "elevenlabs"

    stt_provider: str = "whisper"

    voice_timeout: float = 45.0

    wake_word: str = "aiko"

    chunk_size: int = 1024

    response_format: str = "pcm"

    # Audio is handled in memory only. Keep this ceiling conservative so an
    # unauthenticated transport cannot exhaust the worker's memory.
    max_audio_bytes: int = 10 * 1024 * 1024

    max_transcript_characters: int = 8_000

    turn_timeout: float = 75.0

    # Research extension: acoustic speech-emotion recognition. The public
    # SUPERB Wav2Vec2 checkpoint is lazy-loaded on first audio turn.
    emotion_acoustic_enabled: bool = True

    emotion_model: str = "superb/wav2vec2-base-superb-er"

    emotion_min_confidence: float = 0.45

    emotion_preload: bool = False

    # Browser-side VAD uses these values for responsive barge-in.
    interrupt_min_speech_ms: int = 260

    interrupt_silence_ms: int = 850


class RuntimeSettings(BaseConfig):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RUNTIME_",
        extra="ignore",
    )

    timeout: int = 60

    max_retries: int = 3

    max_parallel_tasks: int = 10

    stream_tokens: bool = True

    save_checkpoints: bool = True

    worker_count: int = 4

    max_queue_size: int = 100


class MemorySettings(BaseConfig):
    """Memory configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        extra="ignore",
    )

    top_k: int = 10

    max_results: int = 20

    similarity_threshold: float = 0.75

    summarization_threshold: int = 30

    consolidation_interval: int = 3600

    embedding_dimension: int = 1536

    auto_summarize: bool = True


class FeatureSettings(BaseConfig):
    """Feature flags."""

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        extra="ignore",
    )

    voice: bool = True

    memory: bool = True

    emotion: bool = True

    relationship: bool = True

    research: bool = False

    automation: bool = False

    vision: bool = True

    tools: bool = True

    streaming: bool = True


class AISettings(BaseConfig):
    """AI provider configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    default_provider: str = "groq"

    embedding_provider: str = "voyage"

    vision_provider: str = "openai"

    fallback_provider: str = "gemini"

    openai: OpenAISettings = Field(
        default_factory=OpenAISettings,
    )

    anthropic: AnthropicSettings = Field(
        default_factory=AnthropicSettings,
    )

    gemini: GeminiSettings = Field(
        default_factory=GeminiSettings,
    )

    groq: GroqSettings = Field(
        default_factory=GroqSettings,
    )

    deepseek: DeepSeekSettings = Field(
        default_factory=DeepSeekSettings,
    )

    qwen: QwenSettings = Field(
        default_factory=QwenSettings,
    )

    openrouter: OpenRouterSettings = Field(
        default_factory=OpenRouterSettings,
    )

    voyage: VoyageSettings = Field(
        default_factory=VoyageSettings,
    )


class SecuritySettings(BaseConfig):
    """Authentication and security configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # JWT
    # ------------------------------------------------------------------

    # Only algorithms explicitly allowlisted by JWTService are permitted.
    jwt_algorithm: str = "HS256"

    # Never hardcode a real secret here.
    # Development/testing/production must provide it through
    # environment variables or a secret manager.
    jwt_secret_key: str = ""

    jwt_issuer: str = "emotion-aware-voice-assistant"

    jwt_audience: str = "emotion-aware-voice-assistant-client"

    # Short-lived access token.
    access_token_expire_minutes: int = 15

    # Longer-lived refresh token.
    refresh_token_expire_days: int = 30

    # ------------------------------------------------------------------
    # Authentication protection
    # ------------------------------------------------------------------

    max_login_attempts: int = 5

    login_lockout_minutes: int = 15

    # ------------------------------------------------------------------
    # Password policy
    # ------------------------------------------------------------------

    password_min_length: int = 12

    password_max_length: int = 128

    # ------------------------------------------------------------------
    # Transport security
    # ------------------------------------------------------------------

    # Keep False for local development.
    # Must be enabled for production deployment.
    require_https: bool = False

    # ------------------------------------------------------------------
    # Token security
    # ------------------------------------------------------------------

    require_jti: bool = True

    # ------------------------------------------------------------------
    # Production hardening
    # ------------------------------------------------------------------

    require_secure_secret_in_production: bool = True

    # Minimum JWT signing secret length.
    jwt_secret_min_length: int = 64


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(
        default_factory=AppSettings,
    )

    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
    )

    redis: RedisSettings = Field(
        default_factory=RedisSettings,
    )

    qdrant: QdrantSettings = Field(
        default_factory=QdrantSettings,
    )

    runtime: RuntimeSettings = Field(
        default_factory=RuntimeSettings,
    )

    memory: MemorySettings = Field(
        default_factory=MemorySettings,
    )

    features: FeatureSettings = Field(
        default_factory=FeatureSettings,
    )

    voice: VoiceSettings = Field(
        default_factory=VoiceSettings,
    )

    elevenlabs: ElevenLabsSettings = Field(
        default_factory=ElevenLabsSettings,
    )

    ai: AISettings = Field(
        default_factory=AISettings,
    )

    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
    )

    @property
    def is_development(self) -> bool:
        """Return whether the application is in development."""

        return self.app.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Return whether the application is in testing."""

        return self.app.environment == Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Return whether the application is in production."""

        return self.app.environment == Environment.PRODUCTION

    @property
    def database_url(self) -> str:
        """Return the database connection URL."""

        return self.database.url

    @property
    def redis_url(self) -> str:
        """Return the Redis connection URL."""

        return self.redis.url

    def validate_runtime_configuration(self) -> None:
        """Validate configuration that is required to serve production traffic."""

        if not self.is_production:
            return

        # JWTService owns the security-critical algorithm and secret checks.
        # Import lazily so an unconfigured development shell can still expose
        # non-authenticated diagnostics and OpenAPI documentation.
        from app.security.jwt import JWTService

        JWTService()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


settings = get_settings()
