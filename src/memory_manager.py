
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import MEMORY_DIR, SESSIONS_DIR, VECTORS_DIR
from .storage.json_storage import JSONStorage
from .storage.vector_storage import VectorStorage
from .logger import get_logger

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self):
        """
        Initializes the MemoryManager with storage backends.
        """
        self.json_storage = JSONStorage(sessions_dir=SESSIONS_DIR)
        self.vector_storage = VectorStorage(persist_directory=str(VECTORS_DIR))

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        """
        Saves a full conversation session.
        
        Args:
            session_id: Unique session identifier.
            messages: List of message dictionaries.
        """
        return self.json_storage.save_session(session_id, messages)

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a specific memory to the vector store for retrieval.
        
        Args:
            text: The content to remember.
            metadata: Additional info (e.g., timestamp, source, session_id).
        """
        self.vector_storage.add_memory(text, metadata)

    def retrieve_context(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieves relevant context based on a query.
        
        Args:
            query: The text to search for.
            n_results: Number of results to return.
            
        Returns:
            List of relevant text snippets.
        """
        results = self.vector_storage.search_memory(query, n_results)
        if results and 'documents' in results:
            # Flatten the list of lists returned by chromadb
            return [doc for sublist in results['documents'] for doc in sublist]
        return []

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a full session log.
        """
        return self.json_storage.load_session(session_id)
