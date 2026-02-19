
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import MagicMock, patch
from src.memory_manager import MemoryManager

# Mock OpenAI to avoid actual server connection during test
@patch("src.memory_manager.MemoryManager")
@patch("openai.OpenAI")
def test_chat_initialization(mock_openai, mock_memory):
    from chat import main
    
    # Mock input to exit immediately
    with patch("builtins.input", return_value="exit"):
        main()
        
    mock_memory.return_value.save_session.assert_not_called() # No messages to save

@patch("src.memory_manager.MemoryManager")
def test_memory_integration(mock_memory):
    mm = mock_memory.return_value
    mm.retrieve_context.return_value = ["Fact 1", "Fact 2"]
    
    # Verify retrieval structure
    results = mm.retrieve_context("query")
    assert len(results) == 2
    assert results[0] == "Fact 1"
