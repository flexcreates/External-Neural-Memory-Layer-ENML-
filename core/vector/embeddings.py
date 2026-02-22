from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL
from core.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> list[float]:
        """Generates embedding for given text."""
        return self.model.encode(text).tolist()
