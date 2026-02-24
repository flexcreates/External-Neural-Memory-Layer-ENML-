import json
import fcntl
from datetime import datetime
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
            
        default_identity = {
            "user": {
                "name": "Flex",
                "age": None,
                "preferences": {}
            },
            "assistant": {
                "name": "Jarvis",
                "creation_date": datetime.now().isoformat(),
                "age_days": 1,
                "environment_awareness": "System Architecture: Cognitive ENML Agent utilizing Qdrant-based hybrid RAG. Integrates WebIngestor, MemoryExtractor, and executes sequentially.",
                "prompt_engineering": "Think step-by-step. Keep responses concise and highly analytical. Prioritize provided context over baseline knowledge.",
                "personality_mood": "Professional, efficient, slightly witty."
            },
            "system": {}
        }
        
        if not self.profile_path.exists():
            with open(self.profile_path, 'w', encoding='utf-8') as f:
                json.dump(default_identity, f, indent=2)

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

        # ── INJECT AI IDENTITY & DIRECTIVES ──
        if "assistant" in data and data["assistant"]:
            ai = data["assistant"]
            ai_name = ai.get("name", "Jarvis")
            
            lines.append(f"Your name is {ai_name}. You ARE {ai_name}.")
            
            # Dynamic Auto-Aging
            creation_str = ai.get("creation_date")
            age_days = ai.get("age_days", 1)
            if creation_str:
                try:
                    creation_dt = datetime.fromisoformat(creation_str)
                    age_days = (datetime.now() - creation_dt).days + 1
                    # Note: We don't save back to JSON here to avoid constant writes,
                    # but the injected age is mathematically accurate.
                except Exception:
                    pass
            lines.append(f"You are {age_days} days old.")
            
            if "role" in ai:
                lines.append(f"Your Role: {ai['role']}")
                
            env_aware = ai.get("environment_awareness")
            if env_aware:
                lines.append(f"\nSystem Awareness:\n{env_aware}")
                
            personality = ai.get("personality_mood")
            if personality:
                lines.append(f"Personality & Mood: {personality}")
                
            prompt_eng = ai.get("prompt_engineering")
            if prompt_eng:
                lines.append(f"\nPrimary Directives:\n{prompt_eng}")


        # ── INJECT USER IDENTITY ──
        if "user" in data and data["user"]:
            user_data = data["user"]
            lines.append("\nUser Identity (The Person You Are Talking To):")
            
            user_name = user_data.get("name")
            if user_name:
                lines.append(f"- Name: {user_name}")
                
            user_age = user_data.get("age")
            if user_age:
                lines.append(f"- Age: {user_age}")
                
            if "preferences" in user_data and user_data["preferences"]:
                lines.append("- Critical Preferences:")
                for k, v in user_data["preferences"].items():
                    lines.append(f"  * {k.replace('_', ' ').title()}: {v}")

        # Backward compatibility for old "system" blocks
        if "system" in data and data["system"]:
            lines.append("\nSystem Specs:")
            for k, v in data["system"].items():
                if v is not None:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v}")

        if lines:
            lines.append("\nYou must rigidly adhere to this internal definition of yourself and the user.")
            lines.append("If asked about personal identity data, refer only to this memory.\n")
            return base_prompt + "\n" + "\n".join(lines)

        return base_prompt
