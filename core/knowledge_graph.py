import json
import uuid
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from .config import GRAPH_DIR
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class EnrichedFact:
    id: str
    subject_id: str
    predicate: str
    object_literal: Optional[str] = None
    object_id: Optional[str] = None
    confidence: float = 1.0
    status: str = "active" # active, superseded, alternative
    superseded_by: Optional[str] = None
    timestamp: datetime = datetime.now()
    source: str = "user"
    
    def to_dict(self):
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_literal": self.object_literal,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }

@dataclass
class Entity:
    id: str
    canonical_name: str
    aliases: List[str]
    entity_type: str
    
    def to_dict(self):
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "entity_type": self.entity_type
        }

class EntityLinker:
    """
    Manages Entity Resolution and Temporal Fact Versioning (Contradiction Detection).
    """
    def __init__(self, embedding_service):
        self.entities_path = GRAPH_DIR / "entities.json"
        self.facts_path = GRAPH_DIR / "facts_ledger.json"
        self.embedding_service = embedding_service
        
        self.entities: Dict[str, Entity] = {}
        self.fact_versions: Dict[str, List[EnrichedFact]] = {}
        
        self._ensure_files()
        self._load_state()

    def _ensure_files(self):
        if not self.entities_path.exists():
            self._save_json(self.entities_path, {})
        if not self.facts_path.exists():
            self._save_json(self.facts_path, {})

    def _save_json(self, path: Path, data: Any):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"EntityLinker: Failed to save {path}: {e}")

    def _load_state(self):
        try:
            raw_ents = self._load_json(self.entities_path)
            for k, v in raw_ents.items():
                self.entities[k] = Entity(**v)
                
            raw_facts = self._load_json(self.facts_path)
            for key, versions in raw_facts.items():
                self.fact_versions[key] = [
                    EnrichedFact(
                        **{**f, "timestamp": datetime.fromisoformat(f["timestamp"])}
                    ) for f in versions
                ]
        except Exception as e:
            logger.error(f"EntityLinker: Failed to load state: {e}")

    def _load_json(self, path: Path) -> Any:
        try:
            if not path.exists(): return {}
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def resolve_or_create(self, mention: str) -> Entity:
        """Links a text mention to an existing entity or creates a new one using exact/alias matching."""
        mention_lower = mention.lower()
        
        # 1. Check existing entities
        for entity in self.entities.values():
            if mention_lower == entity.canonical_name.lower():
                return entity
            for alias in entity.aliases:
                if mention_lower == alias.lower():
                    return entity
                    
        # 2. Create new Entity
        new_entity = Entity(
            id=str(uuid.uuid4()),
            canonical_name=mention_lower,
            aliases=[mention_lower],
            entity_type="unknown"
        )
        self.entities[new_entity.id] = new_entity
        
        # Save state
        self._save_json(self.entities_path, {k: v.to_dict() for k, v in self.entities.items()})
        return new_entity

    def is_contradiction(self, old: EnrichedFact, new: EnrichedFact) -> bool:
        """Determines if two facts conflict based on semantics."""
        if old.predicate != new.predicate:
            return False
            
        old_val = old.object_literal or ""
        new_val = new.object_literal or ""
        
        if old_val.lower() != new_val.lower():
            # Calculate semantic similarity (e.g. RTX 3050 vs RTX 3060)
            try:
                v1 = self.embedding_service.embed(old_val)
                v2 = self.embedding_service.embed(new_val)
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if similarity < 0.75:
                    return True # They are talking about different explicit things
            except Exception:
                return True
                
        return False

    def store_fact(self, fact_kwargs: dict) -> EnrichedFact:
        """Handles temporal versioning of facts instead of dropping contradictions."""
        # Resolve subject
        subject_entity = self.resolve_or_create(fact_kwargs["subject"])
        
        new_fact = EnrichedFact(
            id=str(uuid.uuid4()),
            subject_id=subject_entity.id,
            predicate=fact_kwargs["predicate"],
            object_literal=fact_kwargs["object"],
            confidence=fact_kwargs["confidence"],
            timestamp=datetime.now()
        )
        
        key = f"{subject_entity.id}_{new_fact.predicate}"
        existing_versions = self.fact_versions.get(key, [])
        
        if existing_versions:
            # Check the latest active fact
            active_facts = [f for f in existing_versions if f.status == "active"]
            if active_facts:
                latest = active_facts[-1]
                if self.is_contradiction(latest, new_fact):
                    logger.info(f"EntityLinker: Contradiction detected. Superseding {latest.object_literal} with {new_fact.object_literal}")
                    latest.status = "superseded"
                    latest.superseded_by = new_fact.id
                    new_fact.status = "active"
                else:
                    new_fact.status = "active"
            else:
                new_fact.status = "active"
        else:
            new_fact.status = "active"
            
        existing_versions.append(new_fact)
        self.fact_versions[key] = existing_versions
        
        # Save state
        raw_facts = {k: [f.to_dict() for f in versions] for k, versions in self.fact_versions.items()}
        self._save_json(self.facts_path, raw_facts)
        
        return new_fact
