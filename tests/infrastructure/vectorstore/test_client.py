from importlib import import_module
from unittest.mock import MagicMock, patch

client_module = import_module("app.infrastructure.vectorstore.client")

def test_create_qdrant_client_uses_settings():
    fake_client = MagicMock()
    with (
        patch(
            "app.infrastructure.vectorstore.client.AsyncQdrantClient",
            return_value=fake_client,
        ) as qdrant,
        patch(
            "app.infrastructure.vectorstore.client.settings"
        ) as settings,
    ):
        settings.qdrant.url = "http://localhost:6333"
        settings.qdrant.api_key = "test-api-key"
        settings.qdrant.prefer_grpc = True

        result = client_module.create_qdrant_client()

    assert result is fake_client
    qdrant.assert_called_once_with(
        url="http://localhost:6333",
        api_key="test-api-key",
        prefer_grpc=True,
    )

def test_global_qdrant_client_exists():
    assert client_module.qdrant_client is not None
