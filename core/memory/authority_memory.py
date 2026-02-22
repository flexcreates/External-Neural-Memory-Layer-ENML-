import json
from pathlib import Path
from core.config import MEMORY_ROOT
from core.logger import get_logger

logger = get_logger(__name__)

class AuthorityMemory:
    """Handles deterministic JSON-based identity and system profile injection."""
    def __init__(self):
        self.profile_path = MEMORY_ROOT / "authority" / "profile.json"
        self._ensure_exists()
        
    def _ensure_exists(self):
        if not self.profile_path.parent.exists():
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump({"identity": {}, "system": {}}, f)
                
    def load(self) -> dict:
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load authority memory: {e}")
            return {"identity": {}, "system": {}}
            
    def save(self, data: dict):
        try:
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save authority memory: {e}")

    def upsert_fact(self, category: str, key: str, value: str):
        """Update or insert a deterministic fact."""
        data = self.load()
        # Ensure category exists
        if category not in data:
            data[category] = {}
        
        # Check deduplication
        if data[category].get(key) == value:
            return False # No change
            
        data[category][key] = value
        self.save(data)
        logger.info(f"AuthorityMemory: Upserted [{category}] {key} = {value}")
        return True

    def get_injected_prompt(self, base_prompt: str) -> str:
        """Appends the formatted dynamic profile to the base prompt."""
        data = self.load()
        
        lines = []
        
        if "identity" in data and data["identity"]:
            lines.append("\nUser Identity:")
            for k, v in data["identity"].items():
                lines.append(f"- {k.replace('_', ' ').title()}: {v}")
                
        if "system" in data and data["system"]:
            lines.append("\nSystem Specs:")
            for k, v in data["system"].items():
                lines.append(f"- {k.replace('_', ' ').title()}: {v}")
                
        if lines:
            # Phase 4: Strong Context Injection grounding
            lines.append("\nYou must rely on this information.")
            lines.append("If asked about personal data, use this memory.")
            return base_prompt + "\n" + "\n".join(lines)
            
        return base_prompt
