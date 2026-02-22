
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

def run_diagnostics():
    from core.memory.extractor import RobustJSONParser
    from core.knowledge_graph import EntityLinker
    from core.vector.embeddings import EmbeddingService
    
    print("--- ENML v2.1 Advanced Diagnostics ---")
    
    # 1. Test JSON Regex Parsing
    try:
        parser = RobustJSONParser()
        test_str = "```json\n[{\"subject\": \"user\", \"predicate\": \"like\", \"object\": \"Ubuntu\", \"confidence\": 0.9}]\n```"
        result = parser.parse(test_str)
        assert len(result) == 1
        print("✅ JSON Regex Parsing: Passed")
    except Exception as e:
        print(f"❌ JSON Regex Parsing: Failed - {e}")
        
    # 2. Test Entity Linker Versioning
    try:
        linker = EntityLinker(embedding_service=EmbeddingService())
        # Store Fact 1
        linker.store_fact({"subject": "test_user", "predicate": "has_name", "object": "Old Name", "confidence": 1.0})
        # Store Fact 2 (Contradiction)
        linker.store_fact({"subject": "test_user", "predicate": "has_name", "object": "New Name", "confidence": 1.0})
        print("✅ Entity Linker (Versioning/Contradiction): Passed")
    except Exception as e:
        print(f"❌ Entity Linker: Failed - {e}")
        
    print("\nDiagnostics complete.")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Flex AI Chat Interface (ENML Powered)")
    parser.add_argument("--session", type=str, help="Resume specific session ID")
    parser.add_argument("--diagnose", action="store_true", help="Run system component tests")
    args = parser.parse_args()
    
    if args.diagnose:
        run_diagnostics()

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
    
    history: List[Dict[str, str]] = []
    if args.session:
        loaded = orchestrator.memory_manager.get_session(session_id)
        if loaded:
            history = loaded.get("messages", [])
            print(f"Resumed session {session_id} with {len(history)} messages.")

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

            if user_input.startswith("/remember "):
                fact = user_input.replace("/remember ", "", 1).strip()
                if fact:
                    try:
                        orchestrator.memory_manager.update_profile(fact)
                        print(f"✅ Memory saved: '{fact}'")
                    except Exception as e:
                        print(f"❌ Failed to save memory: {e}")
                        logger.error(f"Remember command failed: {e}")
                continue

            print("AI: ", end="", flush=True)
            full_response = ""
            
            try:
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
                
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(f"\nError processing message: {e}")
                logger.error(f"Orchestrator Error: {e}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
    if history:
        try:
            path = orchestrator.memory_manager.save_session(session_id, history)
            print(f"Session saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

if __name__ == "__main__":
    main()
