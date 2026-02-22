import json
import fcntl
from pathlib import Path
from core.config import MEMORY_ROOT
from core.logger import get_logger

logger = get_logger(__name__)


class AuthorityMemory:
    """Handles deterministic JSON-based identity and system profile injection.
    
    This is the "golden source of truth" layer. Facts stored here are injected
    directly into every system prompt, guaranteeing the LLM always has access
    to immutable identity data (name, specs, etc.) regardless of vector search
    quality.
    
    Thread-safe: all reads/writes use file-level locking.
    """

    def __init__(self):
        self.profile_path = MEMORY_ROOT / "authority" / "profile.json"
        self._ensure_exists()

    def _ensure_exists(self):
        if not self.profile_path.parent.exists():
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.profile_path.exists():
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump({"identity": {}, "system": {}}, f, indent=2)

    def load(self) -> dict:
        """Load the authority profile with shared file lock."""
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to load authority memory: {e}")
            return {"identity": {}, "system": {}}

    def save(self, data: dict):
        """Save the authority profile with exclusive file lock."""
        try:
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to save authority memory: {e}")

    def upsert_fact(self, category: str, key: str, value: str) -> bool:
        """Update or insert a deterministic fact.
        
        Args:
            category: Top-level key (e.g., "identity", "system").
            key: Fact key within the category (e.g., "preferred_name").
            value: The value to store.
            
        Returns:
            True if the fact was changed, False if it was already identical.
        """
        data = self.load()
        if category not in data:
            data[category] = {}

        if data[category].get(key) == value:
            return False  # No change

        data[category][key] = value
        self.save(data)
        logger.info(f"AuthorityMemory: Upserted [{category}] {key} = {value}")
        return True

    def get_injected_prompt(self, base_prompt: str) -> str:
        """Appends the formatted dynamic profile to the base system prompt.
        
        This ensures the LLM always sees the user's identity and system data,
        regardless of whether vector search retrieved anything relevant.
        """
        data = self.load()

        lines = []

        # Inject AI's own identity first
        if "assistant" in data and data["assistant"]:
            ai_name = data["assistant"].get("name")
            ai_role = data["assistant"].get("role")
            if ai_name:
                lines.append(f"\nYour name is {ai_name}. You ARE {ai_name}.")
            if ai_role:
                lines.append(f"Your role: {ai_role}")

        if "identity" in data and data["identity"]:
            lines.append("\nUser Identity:")
            for k, v in data["identity"].items():
                if v is not None:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v}")

        if "system" in data and data["system"]:
            lines.append("\nSystem Specs:")
            for k, v in data["system"].items():
                if v is not None:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v}")

        if lines:
            lines.append("\nYou must rely on this information.")
            lines.append("If asked about personal data, use this memory.")
            return base_prompt + "\n" + "\n".join(lines)

        return base_prompt
