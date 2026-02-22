import json
import uuid
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from .config import GRAPH_DIR
from .logger import get_logger

logger = get_logger(__name__)

# CRITICAL FIX: Define which predicates support multiple values (list semantics)
MULTI_VALUE_PREDICATES: Set[str] = {
    # Interests & hobbies
    'has_interest', 'has_hobby', 'likes', 'enjoys', 'loves', 'prefers',
    # Pets
    'has_pet', 'owns_pet', 'has_dog', 'has_cat', 'has_pet_name', 'pet_name', 'has_name_of_pet',
    # Devices & tech
    'has_computer', 'uses_computer', 'has_device',
    # Usage (generic - user can use multiple things)
    'uses', 'uses_os', 'uses_tool', 'runs',
    # Skills & knowledge
    'knows', 'speaks', 'has_skill', 'works_with', 'creates', 'makes',
    # Projects & work
    'has_project', 'working_on', 'is_working_on', 'works_on',
    # Food & preferences
    'has_preferred_dish', 'has_favorite_food', 'likes_food', 'eats',
    # Physical traits (can have multiple)
    'has_moles', 'has_tattoo', 'has_scar',
    # Relationships (can have multiple)
    'has_friend', 'has_sibling', 'has_colleague',
}

# Predicates that should only have ONE value (replace semantics)
SINGLE_VALUE_PREDICATES: Set[str] = {
    'has_name', 'preferred_name', 'legal_name', 'first_name', 'last_name',
    'birthdate', 'is_a', 'has_type', 'is_type', 'gender', 'age',
    'has_id', 'email', 'phone'
}

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
    timestamp: datetime = field(default_factory=datetime.now)
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
        
        # CRITICAL FIX: Skip contradiction check for multi-value predicates
        if old.predicate in MULTI_VALUE_PREDICATES:
            # Check if it's the EXACT same value (duplicate)
            old_val = (old.object_literal or "").lower().strip()
            new_val = (new.object_literal or "").lower().strip()
            if old_val == new_val:
                return False  # Exact duplicate, will be filtered elsewhere
            return False  # Different values for multi-value predicate = NOT a contradiction
        
        # For single-value predicates, check semantic similarity
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

    def _check_exact_duplicate(self, existing_versions: List[EnrichedFact], new_fact: EnrichedFact) -> bool:
        """Check if exact same fact already exists."""
        for fact in existing_versions:
            if (fact.object_literal or "").lower().strip() == (new_fact.object_literal or "").lower().strip():
                if fact.status == "active":
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
        
        # CRITICAL FIX: Check for exact duplicates first
        if self._check_exact_duplicate(existing_versions, new_fact):
            logger.debug(f"EntityLinker: Exact duplicate found for {new_fact.predicate} {new_fact.object_literal}")
            # Return the existing active fact
            for fact in reversed(existing_versions):
                if fact.status == "active":
                    return fact
        
        # CRITICAL FIX: Handle multi-value predicates (hobbies, interests, pets)
        if new_fact.predicate in MULTI_VALUE_PREDICATES:
            # Just add as new active fact, don't check for contradictions
            new_fact.status = "active"
            existing_versions.append(new_fact)
            self.fact_versions[key] = existing_versions
            
            # Save state
            raw_facts = {k: [f.to_dict() for f in versions] for k, versions in self.fact_versions.items()}
            self._save_json(self.facts_path, raw_facts)
            
            logger.info(f"EntityLinker: Added multi-value fact -> {new_fact.subject_id} {new_fact.predicate} {new_fact.object_literal}")
            return new_fact
        
        # Handle single-value predicates (names, identity) with contradiction detection
        if existing_versions:
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

    def get_current_facts(self, subject_id: str, predicate: Optional[str] = None) -> List[EnrichedFact]:
        """Get all active facts for a subject, optionally filtered by predicate."""
        results = []
        
        for key, versions in self.fact_versions.items():
            if key.startswith(f"{subject_id}_"):
                if predicate and not key.endswith(f"_{predicate}"):
                    continue
                    
                active = [f for f in versions if f.status == "active"]
                results.extend(active)
        
        return results