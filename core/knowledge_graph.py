
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import GRAPH_DIR
from .logger import get_logger

logger = get_logger(__name__)

class KnowledgeGraph:
    def __init__(self):
        self.entities_path = GRAPH_DIR / "entities.json"
        self.relations_path = GRAPH_DIR / "relations.json"
        self._ensure_files()
        
    def _ensure_files(self):
        if not self.entities_path.exists():
            self._save_json(self.entities_path, {})
        if not self.relations_path.exists():
            self._save_json(self.relations_path, [])
            
    def _save_json(self, path: Path, data: Any):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save graph file {path}: {e}")

    def _load_json(self, path: Path) -> Any:
        try:
            if not path.exists(): return {} if path == self.entities_path else []
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load graph file {path}: {e}")
            return {} if path == self.entities_path else []

    def add_entity(self, name: str, entity_type: str, metadata: Dict[str, Any] = {}):
        entities = self._load_json(self.entities_path)
        if name not in entities:
            entities[name] = {"type": entity_type, **metadata}
            self._save_json(self.entities_path, entities)
            logger.info(f"Added entity: {name} ({entity_type})")

    def add_relation(self, source: str, target: str, rel_type: str):
        relations = self._load_json(self.relations_path)
        # Check if exists
        for r in relations:
            if r["from"] == source and r["to"] == target and r["type"] == rel_type:
                return
        
        relations.append({"from": source, "to": target, "type": rel_type})
        self._save_json(self.relations_path, relations)
        logger.info(f"Added relation: {source} -> {target} ({rel_type})")

    def get_related(self, entity_name: str) -> List[Dict[str, str]]:
        relations = self._load_json(self.relations_path)
        related = []
        for r in relations:
            if r["from"] == entity_name:
                related.append({"entity": r["to"], "relation": r["type"], "direction": "out"})
            elif r["to"] == entity_name:
                related.append({"entity": r["from"], "relation": r["type"], "direction": "in"})
        return related
