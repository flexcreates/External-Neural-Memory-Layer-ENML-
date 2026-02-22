import json
import re
from typing import Dict, Any, List, Union
from openai import OpenAI
from core.config import LLAMA_SERVER_URL
from core.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """You are a fact extraction system forming a Knowledge Graph from the user message.

CRITICAL RULES:
1. Output ONLY a pure JSON array. No markdown, no explanations.
2. Format: [{{"subject": "user", "predicate": "snake_case_verb", "object": "clean_value", "fact_type": "identity|preference|fact|interest|property", "confidence": 0.0-1.0}}]
3. Use confidence scores realistically:
   - 0.95-1.0: Direct explicit statements ("my name is Flex")
   - 0.85-0.94: Clear but slightly indirect statements
   - 0.70-0.84: Implied or contextual information
   - <0.70: Uncertain or inferred
4. If no facts found, output: []
5. Subject is usually "user" for personal info. Use specific entities (e.g., "bruno", "lenovo_laptop") when appropriate.
6. For multiple values (hobbies, interests), create separate facts with SAME subject and predicate.

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
        if not raw_output or not isinstance(raw_output, str):
            return []
            
        cleaned = raw_output.strip()
        if not cleaned:
            return []
        
        for extractor in self.extractors:
            try:
                result = extractor(cleaned)
                if result is not None:
                    normalized = self._normalize_to_facts(result)
                    if normalized:
                        return normalized
            except Exception as e:
                logger.debug(f"Extractor {extractor.__name__} failed: {e}")
                continue
        
        logger.warning(f"All extractors failed for: {cleaned[:200]}...")
        return []

    def _extract_direct_json(self, text: str) -> Union[List, Dict, None]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _extract_code_block(self, text: str) -> Union[List, Dict, None]:
        patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        return None

    def _extract_json_array(self, text: str) -> Union[List, None]:
        start = text.find('[')
        if start == -1:
            return None
            
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
        start = text.find('{')
        if start == -1:
            return None
            
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
        if data is None:
            return []
        
        if isinstance(data, list):
            valid_facts = []
            for item in data:
                if isinstance(item, dict) and self._is_valid_fact(item):
                    valid_facts.append(self._sanitize_fact(item))
            return valid_facts
        
        if isinstance(data, dict):
            if "facts" in data and isinstance(data["facts"], list):
                return self._normalize_to_facts(data["facts"])
            if self._is_valid_fact(data):
                return [self._sanitize_fact(data)]
            return []
        
        if isinstance(data, (str, int, float, bool)):
            logger.debug(f"Primitive value received: {data}")
            return []
        
        return []

    def _is_valid_fact(self, item: Dict) -> bool:
        if not isinstance(item, dict):
            return False
        has_subject = "subject" in item and isinstance(item["subject"], str)
        has_predicate = "predicate" in item and isinstance(item["predicate"], str)
        has_object = "object" in item or "object_id" in item or "object_literal" in item
        return has_subject and has_predicate and has_object

    def _sanitize_fact(self, fact: Dict) -> Dict[str, Any]:
        sanitized = {
            "subject": str(fact.get("subject", "user")),
            "predicate": str(fact.get("predicate", "has_property")),
            "object": str(fact.get("object", fact.get("object_literal", fact.get("object_id", "unknown")))),
            "confidence": float(fact.get("confidence", 0.8)),
            "fact_type": str(fact.get("fact_type", "fact")),
        }
        if "subject_id" in fact:
            sanitized["subject_id"] = str(fact["subject_id"])
        if "object_id" in fact:
            sanitized["object_id"] = str(fact["object_id"])
        return sanitized


class MemoryExtractor:
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.parser = RobustJSONParser()
        
        # FIXED: Lowered thresholds to realistic values
        # The LLM rarely gives 0.95+ confidence, so 0.80 is more practical for identity
        self.thresholds = {
            'identity': 0.80,      # Was 0.95 - too high, caused name rejection
            'preference': 0.70,    # Was 0.75 - OK but lowered for flexibility  
            'fact': 0.75,          # Was 0.85 - too high for general facts
            'interest': 0.70,      # New category for hobbies/interests
            'property': 0.75,      # For PC specs, attributes
            'general_knowledge': 0.80,  # Was 0.90
        }
        
    def extract_facts(self, user_input: str) -> List[Dict[str, Any]]:
        if not user_input or not isinstance(user_input, str):
            return []
            
        logger.debug(f"Extracting facts from: '{user_input[:100]}...'")
        
        try:
            prompt = EXTRACTION_PROMPT.format(message=user_input)
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.warning("Empty LLM response")
                return []
                
            raw_content = raw_content.strip()
            logger.debug(f"Raw LLM output: {raw_content[:200]}...")
            
            facts = self.parser.parse(raw_content)
            
            if not facts:
                logger.info(f"No facts extracted from: {raw_content[:100]}...")
                return []
            
            verified_facts = []
            for fact in facts:
                try:
                    fact_type = fact.get("fact_type", "fact").lower()
                    confidence = float(fact.get("confidence", 0.8))
                    
                    # Map generic types to specific thresholds
                    threshold = self._get_threshold(fact_type, fact)
                    
                    if confidence >= threshold:
                        verified_facts.append(fact)
                        logger.debug(f"✅ Accepted: {fact['subject']} {fact['predicate']} {fact['object']} (conf: {confidence:.2f} >= {threshold})")
                    else:
                        logger.info(f"❌ Rejected: {fact['predicate']} (conf: {confidence:.2f} < {threshold})")
                        
                except (TypeError, ValueError) as e:
                    logger.warning(f"Invalid fact structure: {fact}, error: {e}")
                    continue
                    
            logger.info(f"Extracted {len(verified_facts)}/{len(facts)} facts")
            return verified_facts
            
        except Exception as e:
            logger.error(f"Extraction failed: {type(e).__name__}: {e}")
            return []
    
    def _get_threshold(self, fact_type: str, fact: Dict) -> float:
        """Dynamic threshold based on fact type and content."""
        # Check for identity-related predicates regardless of declared type
        identity_predicates = ['has_name', 'is_named', 'preferred_name', 'legal_name', 'has_pet', 'pet_name']
        if fact.get('predicate', '') in identity_predicates:
            return self.thresholds['identity']
        
        # Check for interests/hobbies
        interest_predicates = ['has_interest', 'has_hobby', 'likes', 'enjoys']
        if fact.get('predicate', '') in interest_predicates:
            return self.thresholds['interest']
            
        return self.thresholds.get(fact_type, 0.75)


# Test function
def test_extractor():
    extractor = MemoryExtractor()
    
    test_cases = [
        ("Identity statement", "my name is Flex"),
        ("Pet info", "my pet dog name is bruno"),
        ("PC specs", "my pc specs are lenovo loq 12450HX i5 processor rtx 3050"),
        ("Multiple interests", "i like vibe coding and creating art"),
        ("Question", "what is my name?"),
    ]
    
    print("=" * 70)
    print("MEMORY EXTRACTOR TESTS")
    print("=" * 70)
    
    for name, test_input in test_cases:
        print(f"\n🧪 Test: {name}")
        print(f"   Input: '{test_input}'")
        
        try:
            facts = extractor.extract_facts(test_input)
            print(f"   ✅ Extracted {len(facts)} facts")
            for i, fact in enumerate(facts):
                print(f"      {i+1}. [{fact.get('fact_type')}] {fact.get('subject')} {fact.get('predicate')} {fact.get('object')} (conf: {fact.get('confidence')})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_extractor()