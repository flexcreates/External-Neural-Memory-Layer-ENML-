import threading
from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL
from core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class EmbeddingService:
    """Singleton wrapper around SentenceTransformer.
    
    The model is loaded once and shared across all callers (Retriever,
    EntityLinker, ingestion scripts) to avoid duplicating the ~90 MB
    in-memory model.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
                    instance.model = SentenceTransformer(EMBEDDING_MODEL)
                    cls._instance = instance
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Generates a dense vector embedding for the given text."""
        return self.model.encode(text).tolist()
