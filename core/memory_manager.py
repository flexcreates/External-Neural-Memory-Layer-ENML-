from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import CONVERSATIONS_DIR, QDRANT_CONVERSATION_COLLECTION, QDRANT_PROFILE_COLLECTION, QDRANT_KNOWLEDGE_COLLECTION
from .storage.json_storage import JSONStorage
from .vector.retriever import Retriever
from .router.query_router import QueryRouter
from .memory.authority_memory import AuthorityMemory
from .memory.extractor import MemoryExtractor
from .memory.triple_memory import MemoryTriple
from .logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self):
        """
        Initializes the MemoryManager with Qdrant and JSON backends.
        """
        self.json_storage = JSONStorage(sessions_dir=CONVERSATIONS_DIR)
        self.retriever = Retriever()
        self.query_router = QueryRouter()
        self.authority_memory = AuthorityMemory()
        self.extractor = MemoryExtractor()

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        """Saves a full conversation session."""
        return self.json_storage.save_session(session_id, messages)

    def retrieve_context(self, query: str, n_results: int = 3) -> dict:
        """Routes the query and retrieves specific domain context."""
        collection = self.query_router.route(query)
        
        # Step 2: Force profile collection for self-referential queries
        query_lower = query.lower()
        self_words = ["my", "i", "me", "i'm"]
        if collection == QDRANT_CONVERSATION_COLLECTION:
            if any(f" {w} " in f" {query_lower} " or query_lower.startswith(w) for w in self_words):
                collection = QDRANT_KNOWLEDGE_COLLECTION
            
        results = self.retriever.search(collection, query, limit=n_results)
        docs = []
        for r in results:
            payload = r.get("payload", {})
            # Read triple object format
            if "subject" in payload and "predicate" in payload and "object" in payload:
                docs.append(f"- {payload.get('subject')} {payload.get('predicate')} {payload.get('object')}.")
            else:
                docs.append(f"- {payload.get('text', '')}")
                
        return {
            "type": collection,
            "documents": docs
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a full session log."""
        return self.json_storage.load_session(session_id)

    def get_profile_summary(self) -> str:
        # Phase 5 placeholder
        return ""
        
    def update_profile(self, user_interaction: str):
        """Phase 2: Extract semantic triples, construct MemoryTriple, and save to knowledge collection."""
        facts = self.extractor.extract_facts(user_interaction)
        
        for fact in facts:
            subject = fact.get("subject", "user").lower()
            predicate = fact.get("predicate", "").lower().replace(' ', '_')
            obj = fact.get("object", "")
            confidence = float(fact.get("confidence", 0.0))
            
            # Temporal constraints aren't needed at this layer; extractor prompt handles graph logic.
            
            if confidence < 0.75:
                logger.info(f"MemoryManager: Ignored triple '{predicate}' due to low confidence ({confidence})")
                continue
                
            if not predicate or not obj:
                continue
                
            triple = MemoryTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                timestamp=datetime.now(),
                source="user"
            )
            
            logger.info(f"MemoryManager: Saving Semantic Triple Fact -> {triple.natural_sentence}")
            
            payload = triple.to_dict()
            payload["text"] = triple.natural_sentence
            
            self.retriever.add_memory(
                collection=QDRANT_KNOWLEDGE_COLLECTION,
                text=payload["text"],
                payload=payload
            )
