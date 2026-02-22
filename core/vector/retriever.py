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

    def search(self, collection: str, query: str, limit: int = 5, filter_conditions: Optional[list] = None) -> List[Dict[str, Any]]:
        """Hybrid Semantic Search against a specific collection."""
        query_vector = self.embedding_service.embed(query)
        
        # We implicitly only ever want 'active' or 'alternative' facts, never 'superseded' ones
        must_not_conditions = [
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="superseded")
            )
        ]
        
        query_filter = models.Filter(must_not=must_not_conditions)
        if filter_conditions:
            query_filter.must = filter_conditions
            
        # 1. Broad Vector Similarity Lookup
        response = self.qdrant_manager.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit * 2, # Fetch double and rank locally
            query_filter=query_filter,
            with_payload=True
        )
        
        results = response.points
                
        # 2. Local Re-Ranking (Entity Match + Recency)
        # In a full v3 system, entity linker extracts keywords from args. Here, we parse basic nouns.
        query_lower = query.lower()
        
        scored_results = []
        for r in results:
            payload = r.payload
            
            # Base Vector Semantic Score (0.0 to 1.0)
            score = r.score 
            
            # Entity keyword matching boost
            subject = payload.get("subject", "").lower()
            predicate = payload.get("predicate", "").lower()
            obj_val = payload.get("object_literal", "").lower()
            
            if subject and subject in query_lower:
                score += 0.2
            if predicate and predicate.replace('_', ' ') in query_lower:
                score += 0.1
            if obj_val and obj_val in query_lower:
                score += 0.15
                
            scored_results.append((score, {
                "id": r.id,
                "score": score,
                "payload": payload
            }))
            
        # Sort by hybrid score and limit
        scored_results.sort(key=lambda x: x[0], reverse=True)
        final_results = [item[1] for item in scored_results[:limit]]
                
        return final_results
