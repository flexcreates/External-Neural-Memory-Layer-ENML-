
import pytest
import shutil
from pathlib import Path
from src.memory_manager import MemoryManager

# Use a temporary directory for tests
TEST_DIR = Path("test_data")

@pytest.fixture
def memory_manager():
    # Setup
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    
    # Mock config paths (monkeypatching would be better, but this is simpler for now)
    import src.config
    src.config.MEMORY_DIR = TEST_DIR
    src.config.SESSIONS_DIR = TEST_DIR / "sessions"
    src.config.VECTORS_DIR = TEST_DIR / "vectors"
    src.config.SESSIONS_DIR.mkdir(parents=True)
    src.config.VECTORS_DIR.mkdir(parents=True)
    
    mm = MemoryManager()
    yield mm
    
    # Teardown
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

def test_save_load_session(memory_manager):
    session_id = "test_session"
    messages = [{"role": "user", "content": "hello"}]
    
    memory_manager.save_session(session_id, messages)
    
    loaded = memory_manager.get_session(session_id)
    assert loaded is not None
    assert loaded["session_id"] == session_id
    assert loaded["messages"] == messages

def test_vector_storage(memory_manager):
    memory_manager.add_memory("The sky is blue.")
    memory_manager.add_memory("The grass is green.")
    
    results = memory_manager.retrieve_context("What color is the sky?")
    assert len(results) > 0
    assert "The sky is blue." in results
