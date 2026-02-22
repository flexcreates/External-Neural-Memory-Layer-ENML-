
from typing import List, Dict, Any, Generator
from datetime import datetime
from openai import OpenAI
from .logger import get_logger
from .config import LLAMA_SERVER_URL, EMBEDDING_MODEL
from .memory_manager import MemoryManager
from .context_builder import ContextBuilder

logger = get_logger(__name__)

class Orchestrator:
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.memory_manager = MemoryManager()
        self.context_builder = ContextBuilder(self.memory_manager)
        
    def process_message(self, 
                        user_input: str, 
                        session_id: str, 
                        history: List[Dict[str, str]],
                        system_prompt: str = "You are a helpful AI.") -> Generator[str, None, None]:
        """
        Processes a user message through the ENML pipeline.
        
        Flow:
        1. Validate Input
        2. Retrieve Context (Memory, Profile) -> via ContextBuilder
        3. Build Prompt
        4. Call LLM (Stream)
        5. Store Result
        """
        logger.info(f"Processing message for session {session_id}")
        
        # 1. Knowledge Graph Query (Future)
        # 2. Tool Validation (Future)
        
        # 3. Build Context
        # We don't verify token limits here yet (TODO in Phase 5), but ContextBuilder has the logic placeholders
        # Add user input to history temporarily for context building if needed, 
        # but usually we want to pass the *previous* history + current input.
        
        # 1. Update Profile Immediately (Real-time Learning)
        # Pass recent conversation context so the extractor can resolve pronouns
        # like "its", "that", "this" (e.g., "its David" refers to the pet turtle)
        self.memory_manager.update_profile(user_input, conversation_history=history)

        # 2. Build Context
        # For this implementation, we assume 'history' contains the conversation SO FAR.
        # We append the NEW user message to the context sent to LLM.
        
        full_context, temperature = self.context_builder.build_context(user_input, history, system_prompt=system_prompt)
        
        # Append current user message if not already in history
        # (It shouldn't be in history yet if we want to save it strictly after)
        full_context.append({"role": "user", "content": user_input})
        
        # 4. Call LLM
        try:
            stream = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=full_context,
                stream=True,
                temperature=temperature,
                top_p=0.9,
                max_tokens=1000
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
                    full_response += content
                    
            # 5. Post-Processing
            # Save interaction to memory
            # We return the full response for the caller to update their local state, 
            # but we also should save it to persistent storage here or let the caller do it?
            # Design: Orchestrator should probably handle persistence to ensure it happens.
            # But 'history' passed in is mutable list?
            # Let's verify 'save_session' usage. 
            # For now, we'll let the caller (chat.py) manage the high-level loop and saving,
            # as Orchestrator is the engine.
            
            # TODO: Async Profile Update (Phase 2)
            
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            yield f"Error: {str(e)}"

    def save_session(self, session_id: str, messages: List[Dict[str, str]]):
        return self.memory_manager.save_session(session_id, messages)
