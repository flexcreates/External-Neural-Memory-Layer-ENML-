import json
import re
from typing import Dict, Any, List, Union
from openai import OpenAI
from core.config import LLAMA_SERVER_URL
from core.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are a fact extraction system forming a Knowledge Graph from the user message.

RULES:
1. Output ONLY a pure JSON array. No markdown (` ```json `), no explanations, no prefix text.
2. Format: [{{"subject": "user", "predicate": "snake_case_verb", "object": "clean_value", "fact_type": "factual_claim|preference|identity|general_knowledge", "confidence": 0.0}}]
3. If no facts found, output exactly: []
4. Subject should be the entity being described (usually "user" for personal info).
5. Use snake_case for all keys. Provide a realistic confidence score between 0.0 and 1.0.

User message: {message}

Output:"""

class RobustJSONParser:
    def __init__(self):
        self.extractors = [
            self._extract_direct_json,
            self._extract_code_block,
            self._extract_json_array,
            self._extract_json_object,
        ]
        
    def parse(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse LLM output into a list of fact dictionaries."""
        if not raw_output or not isinstance(raw_output, str):
            return []
            
        cleaned = raw_output.strip()
        if not cleaned:
            return []
        
        # Try each extractor in order
        for extractor in self.extractors:
            try:
                result = extractor(cleaned)
                if result is not None:
                    normalized = self._normalize_to_facts(result)
                    if normalized:  # Only return if we got valid facts
                        return normalized
            except Exception as e:
                logger.debug(f"Extractor {extractor.__name__} failed: {e}")
                continue
        
        logger.warning(f"All extractors failed for: {cleaned[:200]}...")
        return []

    def _extract_direct_json(self, text: str) -> Union[List, Dict, None]:
        """Try parsing the entire string as JSON first."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_code_block(self, text: str) -> Union[List, Dict, None]:
        """Extract JSON from markdown code blocks."""
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        return None

    def _extract_json_array(self, text: str) -> Union[List, None]:
        """Extract the first JSON array found in text."""
        # Find array boundaries, handling nested braces
        start = text.find('[')
        if start == -1:
            return None
            
        # Find matching closing bracket
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        return None

    def _extract_json_object(self, text: str) -> Union[Dict, None]:
        """Extract the first JSON object found in text."""
        start = text.find('{')
        if start == -1:
            return None
            
        # Find matching closing brace
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        return None

    def _normalize_to_facts(self, data: Any) -> List[Dict[str, Any]]:
        """
        Normalize various data structures into a list of fact dictionaries.
        Returns empty list if data is invalid.
        """
        # None check
        if data is None:
            return []
        
        # Already a list - validate each item
        if isinstance(data, list):
            valid_facts = []
            for item in data:
                if isinstance(item, dict) and self._is_valid_fact(item):
                    valid_facts.append(self._sanitize_fact(item))
                else:
                    logger.debug(f"Skipping invalid list item: {item}")
            return valid_facts
        
        # Single dictionary - wrap in list
        if isinstance(data, dict):
            # Check for {"facts": [...]} wrapper
            if "facts" in data and isinstance(data["facts"], list):
                return self._normalize_to_facts(data["facts"])
            
            # Single fact object
            if self._is_valid_fact(data):
                return [self._sanitize_fact(data)]
            
            # Unknown dict structure
            logger.debug(f"Unknown dict structure: {data}")
            return []
        
        # String or other primitive - log and return empty
        if isinstance(data, (str, int, float, bool)):
            logger.debug(f"Primitive value received instead of fact structure: {data}")
            return []
        
        return []

    def _is_valid_fact(self, item: Dict) -> bool:
        """Check if dictionary has required fact fields."""
        if not isinstance(item, dict):
            return False
        
        # Must have at least subject and predicate
        has_subject = "subject" in item and isinstance(item["subject"], str)
        has_predicate = "predicate" in item and isinstance(item["predicate"], str)
        
        # Must have object or object_id
        has_object = "object" in item or "object_id" in item or "object_literal" in item
        
        return has_subject and has_predicate and has_object

    def _sanitize_fact(self, fact: Dict) -> Dict[str, Any]:
        """Ensure fact has all required fields with proper types."""
        sanitized = {
            "subject": str(fact.get("subject", "user")),
            "predicate": str(fact.get("predicate", "has_property")),
            "object": str(fact.get("object", fact.get("object_literal", fact.get("object_id", "unknown")))),
            "confidence": float(fact.get("confidence", 0.8)),
            "fact_type": str(fact.get("fact_type", "factual_claim")),
        }
        
        # Optional fields
        if "subject_id" in fact:
            sanitized["subject_id"] = str(fact["subject_id"])
        if "object_id" in fact:
            sanitized["object_id"] = str(fact["object_id"])
            
        return sanitized


class MemoryExtractor:
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.parser = RobustJSONParser()
        
        self.thresholds = {
            'factual_claim': 0.85,    
            'preference': 0.75,         
            'general_knowledge': 0.9, 
            'identity': 0.95,          
        }
        
    def extract_facts(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Extract facts from user input with full error handling.
        Returns empty list if extraction fails.
        """
        if not user_input or not isinstance(user_input, str):
            return []
            
        logger.debug(f"MemoryExtractor: Extracting facts from '{user_input[:100]}...'")
        
        try:
            prompt = EXTRACTION_PROMPT.format(message=user_input)
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500  # Increased for complex facts
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning("MemoryExtractor: Empty LLM response")
                return []
                
            raw_content = raw_content.strip()
            logger.debug(f"MemoryExtractor: Raw LLM output: {raw_content[:200]}...")
            
            # Parse with robust parser
            facts = self.parser.parse(raw_content)
            
            if not facts:
                logger.info(f"MemoryExtractor: No facts extracted from: {raw_content[:100]}...")
                return []
            
            # Verify confidence thresholds
            verified_facts = []
            for fact in facts:
                try:
                    fact_type = fact.get("fact_type", "factual_claim")
                    confidence = float(fact.get("confidence", 0.8))
                    threshold = self.thresholds.get(fact_type, 0.85)

                    if confidence >= threshold:
                        verified_facts.append(fact)
                        logger.debug(f"MemoryExtractor: Accepted fact: {fact}")
                    else:
                        logger.info(f"MemoryExtractor: Rejected (confidence {confidence:.2f} < {threshold}): {fact['predicate']}")
                        
                except (TypeError, ValueError) as e:
                    logger.warning(f"MemoryExtractor: Invalid fact structure: {fact}, error: {e}")
                    continue
                    
            logger.info(f"MemoryExtractor: Extracted {len(verified_facts)}/{len(facts)} facts")
            return verified_facts
            
        except Exception as e:
            logger.error(f"MemoryExtractor: Extraction failed: {type(e).__name__}: {e}")
            return []
    
    def extract_with_context(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced extraction with conversation context.
        """
        # Build context-aware prompt
        context_str = ""
        if context.get("user_name"):
            context_str += f"\nUser's name is {context['user_name']}."
        if context.get("recent_topics"):
            context_str += f"\nRecent topics: {', '.join(context['recent_topics'][:3])}."
        
        enhanced_prompt = EXTRACTION_PROMPT.replace(
            "User message: {message}",
            f"{context_str}\nUser message: {{message}}"
        )
        
        try:
            prompt = enhanced_prompt.format(message=user_input)
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            raw_content = response.choices[0].message.content.strip()
            return self.parser.parse(raw_content)
            
        except Exception as e:
            logger.error(f"MemoryExtractor: Context extraction failed: {e}")
            return self.extract_facts(user_input)  # Fallback to basic


# Test function for validation
def test_extractor():
    """Run diagnostic tests on the extractor."""
    extractor = MemoryExtractor()
    
    test_cases = [
        ("Simple greeting", "hi how are you brother"),
        ("Identity statement", "my name is Flex"),
        ("Complex specs", "my pc specs are lenovo loq 12450HX i5 processor rtx 3050 6gb"),
        ("Empty input", ""),
        ("Question", "what is my name?"),
    ]
    
    print("=" * 60)
    print("MEMORY EXTRACTOR TESTS")
    print("=" * 60)
    
    for name, test_input in test_cases:
        print(f"\n🧪 Test: {name}")
        print(f"   Input: '{test_input[:50]}...' ")
        
        try:
            facts = extractor.extract_facts(test_input)
            print(f"   ✅ Extracted {len(facts)} facts")
            for i, fact in enumerate(facts[:2]):  # Show first 2
                print(f"      {i+1}. {fact.get('subject')} {fact.get('predicate')} {fact.get('object')} (conf: {fact.get('confidence')})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("PARSER UNIT TESTS")
    print("=" * 60)
    
    parser = RobustJSONParser()
    parser_tests = [
        ('Direct array', '[{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 1.0}]'),
        ('Markdown code block', '```json\n[{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 1.0}]\n```'),
        ('Plain code block', '```\n[{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 1.0}]\n```'),
        ('Empty array', '[]'),
        ('Single object', '{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 1.0}'),
        ('Wrapped object', '{"facts": [{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 1.0}]}'),
        ('Invalid string', 'not json'),
        ('Empty string', ''),
    ]
    
    for name, test_input in parser_tests:
        result = parser.parse(test_input)
        status = "✅" if result is not None else "⚠️"
        print(f"{status} {name}: {len(result) if isinstance(result, list) else 'N/A'} facts")

if __name__ == "__main__":
    test_extractor()