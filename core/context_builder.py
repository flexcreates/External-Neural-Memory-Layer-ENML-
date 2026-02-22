from typing import List, Dict, Any, Optional, Tuple
import re
from .memory_manager import MemoryManager
from .logger import get_logger

logger = get_logger(__name__)

class ContextBuilder:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
    def build_context(self, 
                      user_input: str, 
                      history: List[Dict[str, str]], 
                      system_prompt: str = "You are a persistent AI assistant named Jarvis.",
                      max_context_tokens: int = 2800) -> Tuple[List[Dict[str, str]], float]:
        """
        Builds the context and returns messages & temperature based on query mode.
        """
        # 1. Routing & Retrieval
        retrieval_data = self.memory_manager.retrieve_context(user_input, n_results=5)
        mode = retrieval_data["type"]
        docs = retrieval_data["documents"]
        
        temperature = 0.6
        effective_system_prompt = system_prompt
        
        # 2. Mode-Specific Prompts
        if mode == "research_collection":
            temperature = 0.0
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are a factual AI assistant.\n"
                "Use ONLY the context below.\n"
                "If answer not present, say you do not know.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question:\n{user_input}"
            )
        elif mode == "project_collection":
            temperature = 0.2
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are an AI engineering assistant.\n"
                "Use the following project context to answer.\n\n"
                f"Context:\n{context_str}"
            )
        else: # Conversation / General Semantic Profile
            temperature = 0.6
        
        # 3. Authority Memory Injection (Always injected)
        # Auth memory will append: "User Identity:\n... \nYou must rely on this information."
        
        # Merge Semantic Memory directly into the system prompt for unified attention
        if mode in ["knowledge_collection", "profile_collection", "conversation_collection"] and docs:
            formatted_docs = "\n".join(docs)
            effective_system_prompt += (
                f"\n\nRelevant Known Facts:\n{formatted_docs}\n"
                f"Only answer using the provided facts where applicable. If no relevant fact exists, say you don't know."
            )
            
        effective_system_prompt = self.memory_manager.authority_memory.get_injected_prompt(effective_system_prompt)
        
        messages = [{"role": "system", "content": effective_system_prompt}]
        
        # 4. Append Conversation History (with token budget enforcement)
        SLIDING_WINDOW_COUNT = 20
        recent_history = history[-SLIDING_WINDOW_COUNT:] if len(history) > SLIDING_WINDOW_COUNT else history
        
        # Calculate remaining token budget after system prompt
        system_tokens = self.estimate_tokens(effective_system_prompt)
        user_tokens = self.estimate_tokens(user_input)
        remaining_budget = max_context_tokens - system_tokens - user_tokens - 100  # 100 token safety margin
        
        # Trim history from oldest if it exceeds budget
        trimmed_history = []
        running_tokens = 0
        for msg in reversed(recent_history):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))
            if running_tokens + msg_tokens > remaining_budget:
                break
            trimmed_history.insert(0, msg)
            running_tokens += msg_tokens
        
        messages.extend(trimmed_history)
        
        return messages, temperature

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~1.3 tokens per whitespace-delimited word."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
