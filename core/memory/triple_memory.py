from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

@dataclass
class MemoryTriple:
    subject: str
    predicate: str
    object: str
    confidence: float
    timestamp: datetime
    source: str
    
    def to_dict(self):
        d = asdict(self)
        d['timestamp'] = d['timestamp'].isoformat()
        return d
        
    @property
    def natural_sentence(self):
        # Format for vector semantic search
        return f"{self.subject} {self.predicate} {self.object}."
