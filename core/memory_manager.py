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
        Initializes the MemoryManager with Qdrant, JSON backends, and Feedback.
        """
        from .memory_feedback import MemoryFeedbackSystem
        
        self.json_storage = JSONStorage(sessions_dir=CONVERSATIONS_DIR)
        self.retriever = Retriever()
        self.query_router = QueryRouter()
        self.authority_memory = AuthorityMemory()
        self.extractor = MemoryExtractor()
        self.feedback = MemoryFeedbackSystem()

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
        """Phase 2: Extract semantic triples, construct EnrichedFacts via EntityLinker, and save to knowledge collection."""
        from .knowledge_graph import EntityLinker, MULTI_VALUE_PREDICATES
        
        # Instantiate Linker with the embedding service instance from retriever
        entity_linker = EntityLinker(embedding_service=self.retriever.embedding_service)
        
        facts = self.extractor.extract_facts(user_input=user_interaction)
        
        for fact in facts:
            subject = fact.get("subject", "user").lower()
            predicate = fact.get("predicate", "").lower().replace(' ', '_')
            obj = fact.get("object", "")
            confidence = float(fact.get("confidence", 0.0))
            
            if not predicate or not obj:
                continue
            
            # CRITICAL FIX: Check if this is a multi-value predicate
            # If so, check for exact duplicates but allow multiple different values
            is_multi_value = predicate in MULTI_VALUE_PREDICATES
            
            if is_multi_value:
                # Search for existing similar facts to avoid exact duplicates
                existing = self._find_existing_fact(subject, predicate, obj)
                if existing:
                    logger.debug(f"Skipping duplicate: {subject} {predicate} {obj}")
                    continue
                # For multi-value, we don't check for "contradictions" - we just add
                status = "active"
            else:
                # For single-value, use the entity linker's contradiction detection
                enriched_fact = entity_linker.store_fact({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence
                })
                status = enriched_fact.status
            
            # Build payload
            payload = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "status": status
            }
            payload["text"] = f"{subject} {predicate} {obj}."
            
            logger.info(f"MemoryManager: Processed Fact -> {subject} {predicate} {obj} [Status: {status}]")
            
            # Store in Qdrant with status metadata
            self.retriever.add_memory(
                collection=QDRANT_KNOWLEDGE_COLLECTION,
                text=payload["text"],
                payload=payload
            )
    
    def _find_existing_fact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        """Check if exact fact already exists to prevent duplicates."""
        try:
            # Search for subject+predicate combinations
            query = f"{subject} {predicate}"
            results = self.retriever.search(
                QDRANT_KNOWLEDGE_COLLECTION, 
                query, 
                limit=5,
                filter_dict={"subject": subject, "predicate": predicate}
            )
            
            for r in results:
                payload = r.get("payload", {})
                if payload.get("object") == obj:
                    return payload
            return None
        except Exception as e:
            logger.debug(f"Error checking for existing fact: {e}")
            return None