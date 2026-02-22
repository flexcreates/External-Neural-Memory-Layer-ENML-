from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config import (
    QDRANT_URL, QDRANT_API_KEY, 
    QDRANT_RESEARCH_COLLECTION, QDRANT_PROJECT_COLLECTION, QDRANT_CONVERSATION_COLLECTION,
    QDRANT_PROFILE_COLLECTION, QDRANT_KNOWLEDGE_COLLECTION, EMBED_DIM
)
from core.logger import get_logger

logger = get_logger(__name__)

class QdrantManager:
    """Manages Qdrant collections ensuring consistent dimension and existence."""
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        self.collections = [
            QDRANT_RESEARCH_COLLECTION,
            QDRANT_PROJECT_COLLECTION,
            QDRANT_CONVERSATION_COLLECTION,
            QDRANT_PROFILE_COLLECTION,
            QDRANT_KNOWLEDGE_COLLECTION
        ]
        self._ensure_collections()
        
    def _ensure_collections(self):
        for collection_name in self.collections:
            if not self.client.collection_exists(collection_name):
                logger.info(f"Creating missing Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBED_DIM,
                        distance=models.Distance.COSINE
                    )
                )
