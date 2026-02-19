
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from ..logger import get_logger

logger = get_logger(__name__)

class VectorStorage:
    def __init__(self, persist_directory: str, collection_name: str = "memory_collection"):
        """
        Initializes the VectorStorage with ChromaDB.
        
        Args:
            persist_directory: Directory where ChromaDB stores its data.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        try:
            self.client = chromadb.PersistentClient(path=str(persist_directory))
            
            # Use default embedding function (all-MiniLM-L6-v2)
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Initialized ChromaDB collection '{collection_name}' in {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None):
        """
        Adds a memory to the vector store.
        
        Args:
            text: The text content of the memory.
            metadata: Optional metadata dictionary.
            memory_id: Optional unique ID. If None, one will be generated.
        """
        if memory_id is None:
            import uuid
            memory_id = str(uuid.uuid4())
            
        if metadata is None:
            metadata = {}
            
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            logger.info(f"Added memory {memory_id} to vector store.")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    def search_memory(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Searches for relevant memories.
        
        Args:
            query_text: The query string.
            n_results: Number of results to return.
            
        Returns:
            Dictionary containing 'ids', 'documents', 'metadatas', 'distances'.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return {}

    def count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()
