
import os
import sys
import argparse
from datetime import datetime
from openai import OpenAI
from src.memory_manager import MemoryManager
from src.logger import get_logger

# Initialize Logger
logger = get_logger("ChatInterface")

# Configuration (Mirrors run_server.sh optimized defaults)
API_BASE_URL = "http://localhost:8080/v1"
API_KEY = "sk-no-key-required"
MODEL_NAME = "Meta-Llama-3-8B-Instruct"

# Optimization Limits
MAX_TOTAL_TOKENS = 3000     # Hard cap for prompt + memory
MAX_CONTEXT_TOKENS = 2500   # Limit for history + system prompt
SAFE_OUTPUT_TOKENS = 500    # Reserved for generation
SLIDING_WINDOW_TURNS = 10   # Keep last 10 exchanges (20 msgs)
MAX_MEMORY_HITS = 2         # Tier 3 limit

def estimate_tokens(text: str) -> int:
    """Approximate token count (1.3 chars/token is rough, user asked for words * 1.3)."""
    if not text: return 0
    return int(len(text.split()) * 1.3)

def truncate_history(messages: list, max_tokens: int) -> list:
    """
    Truncates message history to fit within max_tokens.
    Keeps system prompt (if present) and most recent messages.
    """
    if not messages:
        return []

    # Separate system prompt if it exists at index 0
    system_msg = None
    if messages[0]['role'] == 'system':
        system_msg = messages[0]
        history = messages[1:]
    else:
        history = messages[:]

    # Calculate token usage
    current_tokens = estimate_tokens(system_msg['content']) if system_msg else 0
    kept_history = []
    
    # Add messages from newest to oldest until limit reached
    for msg in reversed(history):
        msg_tokens = estimate_tokens(msg['content'])
        if current_tokens + msg_tokens > max_tokens:
            break
        kept_history.insert(0, msg)
        current_tokens += msg_tokens
    
    # Reattach system prompt
    final_messages = [system_msg] + kept_history if system_msg else kept_history
    return final_messages

def main():
    parser = argparse.ArgumentParser(description="Flex AI Chat Interface")
    parser.add_argument("--session", type=str, help="Resume specific session ID")
    args = parser.parse_args()

    print("Initializing Memory System (Optimized)...")
    try:
        memory = MemoryManager()
        logger.info("MemoryManager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MemoryManager: {e}")
        sys.exit(1)

    print("Connecting to Llama 3 Server...")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Session Management
    session_id = args.session if args.session else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Session ID: {session_id}")
    
    # Load history if resuming
    messages = []
    if args.session:
        loaded = memory.get_session(session_id)
        if loaded:
            messages = loaded.get("messages", [])
            print(f"Resumed session {session_id} with {len(messages)} messages.")
    
    print(f"\n--- Chat Started (Session: {session_id}) ---")
    print("Type 'exit' to quit, '/remember <text>' to save a fact.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("Saving session and exiting...")
                break

            # Handle commands
            if user_input.startswith("/remember "):
                fact = user_input.replace("/remember ", "", 1).strip()
                if fact:
                    memory.add_memory(fact, metadata={"source": session_id, "type": "user_fact"})
                    print(f"Memory saved: '{fact}'")
                    logger.info(f"User explicitly saved memory: {fact}")
                continue

            # --- TIER 3: Limited Semantic Memory Injection ---
            context_results = memory.retrieve_context(user_input, n_results=MAX_MEMORY_HITS)
            context_str = ""
            if context_results:
                # Limit total context tokens (e.g. max 400 chars roughly)
                context_str = "\n".join([f"- {c}" for c in context_results])
                logger.info(f"Retrieved {len(context_results)} context items.")

            # --- TIER 1 & 2: System Identity & Structured Facts ---
            # (Placeholder for structured facts, user can be added manually or retrieved via specific meta-tag query)
            system_prompt_content = (
                "You are Flex's AI assistant running on a local RTX 3050.\n"
                "Keep answers concise and efficient.\n"
            )
            
            if context_str:
                system_prompt_content += f"\nRelevant Context:\n{context_str}\n"

            # --- Sliding Window Enforcement ---
            # 1. Update local history first
            messages.append({"role": "user", "content": user_input})
            
            # 2. Keep last N turns (user said 10 exchanges = 20 messages)
            # We treat the list as [history], system prompt is dynamic per turn
            recent_history = messages[- (SLIDING_WINDOW_TURNS * 2):]

            # 3. Construct API Payload
            api_messages = [{"role": "system", "content": system_prompt_content}] + recent_history
            
            # 4. Hard Cap Token Check
            api_messages = truncate_history(api_messages, MAX_CONTEXT_TOKENS)
            
            # Log usage
            total_est_tokens = sum(estimate_tokens(m['content']) for m in api_messages)
            if total_est_tokens > MAX_CONTEXT_TOKENS:
                logger.warning(f"Prompt size {total_est_tokens} exceeds target {MAX_CONTEXT_TOKENS} even after truncation.")

            # --- Generation ---
            print("AI: ", end="", flush=True)
            full_response = ""
            
            try:
                stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    stream=True,
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=SAFE_OUTPUT_TOKENS # Safety limit
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print() 
                
                # Update local history
                messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(f"\nError communicating with server: {e}")
                logger.error(f"API Error: {e}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
    # Save conversation
    if messages:
        try:
            path = memory.save_session(session_id, messages)
            print(f"Session saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

if __name__ == "__main__":
    main()
