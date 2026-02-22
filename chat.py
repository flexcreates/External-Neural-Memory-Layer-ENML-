
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict

# Add project root to path if needed, though running from root should work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from core.logger import get_logger
from core.config import AI_NAME, AI_HINT

logger = get_logger("ChatInterface")

def main():
    parser = argparse.ArgumentParser(description="Flex AI Chat Interface (ENML Powered)")
    parser.add_argument("--session", type=str, help="Resume specific session ID")
    args = parser.parse_args()

    print("Initializing ENML Orchestrator...")
    try:
        orchestrator = Orchestrator()
        logger.info("Orchestrator initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Orchestrator: {e}")
        sys.exit(1)

    # Session Management
    session_id = args.session if args.session else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Session ID: {session_id}")
    
    # Load history if resuming
    # Orchestrator has memory_manager
    history: List[Dict[str, str]] = []
    if args.session:
        loaded = orchestrator.memory_manager.get_session(session_id)
        if loaded:
            history = loaded.get("messages", [])
            print(f"Resumed session {session_id} with {len(history)} messages.")

    # System Prompt Definition
    SYSTEM_PROMPT = (
        f"You are {AI_NAME} {AI_HINT}.\n"
        "Keep answers concise and efficient.\n"
    )

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
                    orchestrator.memory_manager.add_memory(fact, metadata={"source": session_id, "type": "user_fact"})
                    print(f"Memory saved: '{fact}'")
                continue
            
            # TODO: Handle other commands like /project, /research via Orchestrator (Phase 3/4 integration)

            # Process Message via Orchestrator
            print("AI: ", end="", flush=True)
            full_response = ""
            
            try:
                # Orchestrator returns a generator that streams the response
                # It handles context building, memory retrieval, profile injection, etc.
                response_stream = orchestrator.process_message(
                    user_input=user_input, 
                    session_id=session_id, 
                    history=history,
                    system_prompt=SYSTEM_PROMPT
                )
                
                for chunk in response_stream:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print() 
                
                # Update history
                # Note: Orchestrator used 'history' for context but didn't modify it in-place?
                # or did it? ContextBuilder creates a new list. 
                # We must update our local history to maintain state conformant with what Orchestrator expects next time.
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": full_response})
                
                # Profile is already updated by Orchestrator before generation

            except Exception as e:
                print(f"\nError processing message: {e}")
                logger.error(f"Orchestrator Error: {e}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
    # Save conversation
    if history:
        try:
            path = orchestrator.save_session(session_id, history)
            print(f"Session saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

if __name__ == "__main__":
    main()
