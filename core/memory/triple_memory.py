from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class MemoryTriple:
    """A semantic triple representing a single fact in the Knowledge Graph.
    
    Format: Subject → Predicate → Object  (e.g., "user has_hobby paragliding")
    
    Attributes:
        subject: The entity the fact is about (usually "user").
        predicate: The relationship type in snake_case (e.g., "has_name").
        object: The value or target entity.
        confidence: Extraction confidence score (0.0–1.0).
        fact_type: Category — "identity", "preference", "fact", "interest", "property".
        timestamp: When the fact was extracted.
        source: Origin of the fact (e.g., "user", "web", "ingest").
    """
    subject: str
    predicate: str
    object: str
    confidence: float
    fact_type: str = "fact"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "user"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = d['timestamp'].isoformat()
        return d

    @property
    def natural_sentence(self) -> str:
        """Format for vector semantic search embedding."""
        return f"{self.subject} {self.predicate} {self.object}."
