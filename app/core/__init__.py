"""Core application components."""

from .exceptions import AssistantError
from .lifecycle import lifespan
from .logger import get_logger
from .settings import get_settings, settings
from .version import APP_NAME, API_VERSION, VERSION

__all__ = [
    "APP_NAME",
    "API_VERSION",
    "VERSION",
    "AssistantError",
    "get_logger",
    "get_settings",
    "lifespan",
    "settings",
]
