from typing import Dict, Any, List, Optional
from core.vector.qdrant_client import QdrantManager
from core.vector.embeddings import EmbeddingService
from qdrant_client.http import models
from core.logger import get_logger
import uuid

logger = get_logger(__name__)

class Retriever:
    """Handles insertions and semantic search across specific Qdrant collections."""
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.embedding_service = EmbeddingService()

    def add_memory(self, collection: str, text: str, payload: Dict[str, Any], memory_id: Optional[str] = None):
        """Adds text context and its metadata to the specified collection."""
        if memory_id is None:
            memory_id = str(uuid.uuid4())
            
        vector = self.embedding_service.embed(text)
        
        # Add plain text for retrieval back to original form
        if "text" not in payload:
            payload["text"] = text
            
        self.qdrant_manager.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Added memory to {collection} with ID {memory_id}")

    def search(self, collection: str, query: str, limit: int = 3, filter_conditions: Optional[list] = None) -> List[Dict[str, Any]]:
        """Semantic search against a specific collection."""
        query_vector = self.embedding_service.embed(query)
        
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)
            
        response = self.qdrant_manager.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        
        results = response.points
                
        parsed_results = []
        for r in results:
            parsed_results.append({
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            })
                
        return parsed_results
