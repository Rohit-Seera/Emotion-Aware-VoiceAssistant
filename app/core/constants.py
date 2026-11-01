"""Shared application constants."""

from typing import Final

# Encoding

UTF8: Final = "utf-8"

# Localization

DEFAULT_TIMEZONE: Final = "UTC"
DEFAULT_LOCALE: Final = "en"

# API

API_PREFIX: Final = "/api"
API_VERSION_PREFIX: Final = f"{API_PREFIX}/v1"

HEALTH_ENDPOINT: Final = "/health"
VERSION_ENDPOINT: Final = "/version"

# Pagination

DEFAULT_PAGE_SIZE: Final = 20
MAX_PAGE_SIZE: Final = 100

# Runtime

DEFAULT_TIMEOUT: Final = 30
DEFAULT_RETRY_COUNT: Final = 3

# Memory

DEFAULT_MEMORY_LIMIT: Final = 10
MAX_MEMORY_LIMIT: Final = 100
DEFAULT_CONVERSATION_HISTORY: Final = 20

# Cache

DEFAULT_CACHE_TTL: Final = 300
SHORT_CACHE_TTL: Final = 60
LONG_CACHE_TTL: Final = 3600

# Common values

EMPTY_STRING: Final = ""
UNKNOWN: Final = "unknown"
ELLIPSIS: Final = "..."