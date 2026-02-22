from typing import Dict, Any
from core.config import (
    QDRANT_RESEARCH_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_CONVERSATION_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION
)
from core.logger import get_logger

logger = get_logger(__name__)

class QueryRouter:
    """Routes queries to correct Qdrant collection or Identity memory."""
    def __init__(self):
        pass

    def route(self, query: str) -> str:
        """
        Determines the target collection or domain based on query analysis.
        Returns: 'profile_collection', 'project_collection', 'research_collection', or 'conversation_collection'
        """
        query_lower = query.lower()
        
        # Identity / Profile routing (Hits vector semantic memory + deterministic auto-injection downstream)
        # Highest priority to prevent 'what is my' being hijacked by 'what is' research generic
        identity_keywords = [
            "who am i", "my name", "my pc", "my specs", "my father", 
            "my mother", "my condition", "what do i like", "do i have", 
            "my hobbies", "my friend", "my best friend", "what is my"
        ]
        if any(kw in query_lower for kw in identity_keywords):
            return QDRANT_KNOWLEDGE_COLLECTION
            
        # Project routing
        project_keywords = ["codebase", "implemented", "script", "folder", "architecture", ".py", "function", "class", "module"]
        if any(kw in query_lower for kw in project_keywords):
            return QDRANT_PROJECT_COLLECTION
            
        # Research routing
        research_keywords = ["explain", "how does", "what is", "documentation", "article", "theory", "paper"]
        if any(kw in query_lower for kw in research_keywords):
            return QDRANT_RESEARCH_COLLECTION
            
        # Fallback to general conversation memory
        return QDRANT_CONVERSATION_COLLECTION
