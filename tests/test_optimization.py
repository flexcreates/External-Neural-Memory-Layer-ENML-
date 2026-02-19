
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chat import estimate_tokens, truncate_history

def test_estimate_tokens():
    text = "hello world"
    # len("hello world".split()) * 1.3 = 2 * 1.3 = 2.6 -> 2
    assert estimate_tokens(text) == 2
    
    text = "a " * 10
    # 10 words * 1.3 = 13
    assert estimate_tokens(text) == 13

def test_truncate_history():
    # Mock messages
    system_msg = {'role': 'system', 'content': 'system prompt'} # 2 words -> ~2 tokens
    user_msg_1 = {'role': 'user', 'content': 'hello there'} # 2 words -> ~2 tokens
    asst_msg_1 = {'role': 'assistant', 'content': 'hi !'} # 2 words -> ~2 tokens
    
    messages = [system_msg, user_msg_1, asst_msg_1]
    
    # Total ~6 tokens
    
    # 1. Test no truncation needed
    result = truncate_history(messages, 100)
    assert len(result) == 3
    assert result[0] == system_msg
    
    # 2. Test truncation (limit to fit system + 1 msg)
    # limit 5 tokens. system (2) + latest (2) = 4. middle one should be dropped.
    result = truncate_history(messages, 5)
    assert len(result) == 2
    assert result[0] == system_msg
    assert result[1] == asst_msg_1 # Sould keep newest
    
    # 3. Test truncation strict (only system fits)
    # limit 3 tokens. system (2). latest (2). 2+2=4 > 3. 
    # Logic: system is always kept. loop reversed history.
    # current=2. add asst(2) -> 4 > 3. break.
    result = truncate_history(messages, 3)
    assert len(result) == 1
    assert result[0] == system_msg

if __name__ == "__main__":
    pass
