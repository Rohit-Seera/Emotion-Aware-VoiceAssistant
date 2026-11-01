import app.infrastructure.vectorstore as package
from app.infrastructure.vectorstore.client import qdrant_client
from app.infrastructure.vectorstore.collections import CollectionManager, collections

def test_public_exports():
    assert package.CollectionManager is CollectionManager
    assert package.collections is collections
    assert package.qdrant_client is qdrant_client

def test_all_exports():
    assert set(package.__all__) == {
        "CollectionManager",
        "collections",
        "qdrant_client",
    }
