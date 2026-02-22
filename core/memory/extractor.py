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
2. Format: [{{"subject": "user|assistant|entity_name", "predicate": "snake_case_verb", "object": "clean_value", "fact_type": "identity|preference|fact|interest|property", "confidence": 0.0-1.0}}]
3. Use confidence scores realistically:
   - 0.95-1.0: Direct explicit statements ("my name is Flex")
   - 0.85-0.94: Clear but slightly indirect statements
   - 0.70-0.84: Implied or contextual information
   - <0.70: Uncertain or inferred
4. If no facts found, output: []
5. SUBJECT RULES (VERY IMPORTANT):
   - Use "user" when the person is talking about THEMSELVES: "my name is Flex", "I have a dog", "I like coding"
   - Use "assistant" ONLY when the person is giving the AI a NAME or ROLE: "you are Jarvis", "your name is Jarvis", "call yourself X"
   - HARDWARE/SPECS: Even if phrased as "you are running on my laptop", the hardware belongs to the USER. Use "user" as subject for has_device, has_processor, has_graphics_card, uses, etc.
   - Use specific entity names (e.g., "colu", "lenovo_laptop") for third-party entities when appropriate.
6. For multiple values (hobbies, interests, pets), create separate facts with SAME subject and predicate.
7. DO NOT extract facts from questions, commands, or conversational filler (e.g., "what is my name?", "hello", "give me specs"). Return [] for questions and commands.
8. DO NOT extract facts from the AI's own previous responses in the context. Only extract from the USER's current message.
9. CORRECTION HANDLING: When the user corrects an earlier statement (e.g., "no its not X, its Y"), use the CONVERSATION CONTEXT to determine WHAT is being corrected. Extract ONLY the corrected value with the correct predicate.
10. PRONOUN RESOLUTION: Use the conversation context to resolve "it", "its", "that", "this". If the user says "its David" after talking about a pet, that is the pet's name.
11. NEVER use predicate "has_name" for devices, laptops, brands or models. Use "has_device", "has_model", etc.

EXAMPLES:
- "my name is Flex" -> [{{"subject": "user", "predicate": "has_name", "object": "Flex", "fact_type": "identity", "confidence": 0.95}}]
- "from today you are known as Jarvis" -> [{{"subject": "assistant", "predicate": "has_name", "object": "Jarvis", "fact_type": "identity", "confidence": 0.95}}]
- "you are running on my laptop Lenovo Loq with i5 and RTX 3050" -> [{{"subject": "user", "predicate": "has_device", "object": "Lenovo Loq", "fact_type": "property", "confidence": 0.95}}, {{"subject": "user", "predicate": "has_processor", "object": "i5", "fact_type": "property", "confidence": 0.90}}, {{"subject": "user", "predicate": "has_graphics_card", "object": "RTX 3050", "fact_type": "property", "confidence": 0.90}}]
- "no its not ThinkPad its Lenovo Loq" -> [{{"subject": "user", "predicate": "has_device", "object": "Lenovo Loq", "fact_type": "property", "confidence": 0.95}}]
- Context: User has pet lizard. User says: "its name is Colu" -> [{{"subject": "user", "predicate": "has_pet_name", "object": "Colu", "fact_type": "identity", "confidence": 0.90}}]
- "give me laptop specs" -> [] (this is a COMMAND, not a fact)
- "what is my graphics card?" -> [] (this is a QUESTION, not a fact)

{context}User message: {message}

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
        
    def extract_facts(self, user_input: str, conversation_context: str = "") -> List[Dict[str, Any]]:
        """Extract semantic triples from user input.
        
        Args:
            user_input: The current user message.
            conversation_context: Recent conversation history for pronoun resolution.
        """
        if not user_input or not isinstance(user_input, str):
            return []
        
        # Pre-check: immediately skip questions and commands
        if self._is_question_or_command(user_input):
            logger.info(f"Skipping extraction (question/command): {user_input[:60]}...")
            return []
            
        logger.debug(f"Extracting facts from: '{user_input[:100]}...'")
        
        # Build context string for the prompt
        context_block = ""
        if conversation_context:
            context_block = f"Recent conversation context (for resolving pronouns like 'it', 'its', 'that'):\n{conversation_context}\n\n"
        
        try:
            prompt = EXTRACTION_PROMPT.format(message=user_input, context=context_block)
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
                    # Post-process: fix subject misclassification
                    fact = self._fix_subject_misclassification(fact, user_input)
                    
                    # Guard: block device/brand names from being stored as user names
                    if self._guard_name_override(fact):
                        continue
                    
                    # Filter: reject noise/filler facts
                    if self._is_noise_fact(fact):
                        continue
                    
                    # Normalize: map generic predicates to specific ones
                    fact = self._normalize_predicate(fact)
                    
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
    
    # ── Pre-check: skip questions and commands before LLM call ──────────
    
    _QUESTION_PATTERNS = [
        re.compile(r'^\s*(what|who|where|when|why|how|which|can you|do you|does|did|is|are|was|were|will|would|could|should|tell me|explain|describe|give|show|list)\b', re.IGNORECASE),
        re.compile(r'\?\s*$'),  # Ends with question mark
    ]
    
    _COMMAND_PATTERNS = [
        re.compile(r'^\s*(give|show|list|tell|explain|describe|help|find|search|get|fetch|display|print|run|execute)\s+(me\s+)?(my\s+|the\s+|a\s+|all\s+|complete\s+|full\s+)?\w', re.IGNORECASE),
    ]
    
    def _is_question_or_command(self, text: str) -> bool:
        """Pre-check: detect questions and commands to skip LLM extraction entirely."""
        text_stripped = text.strip()
        
        # Short messages are likely greetings or filler
        if len(text_stripped.split()) <= 2 and not any(c.isupper() for c in text_stripped[1:]):
            # Allow "my name is X" but skip "hi", "bye", "hello"
            greetings = {'hi', 'hello', 'hey', 'bye', 'goodbye', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'sure', 'yep', 'nope'}
            if text_stripped.lower().strip('!., ') in greetings:
                return True
        
        for pattern in self._QUESTION_PATTERNS:
            if pattern.search(text_stripped):
                return True
        
        for pattern in self._COMMAND_PATTERNS:
            if pattern.search(text_stripped):
                return True
        
        return False
    
    # ── Noise filter: reject filler facts ───────────────────────────────
    
    _NOISE_PREDICATES = {
        'has_greeting', 'has_introduction', 'says', 'asks', 'wants', 'requests',
        'greets', 'introduces', 'mentions', 'states', 'inquires', 'responds',
        'is_about', 'has_question', 'has_command', 'has_response', 'has_message',
        'is_greeting', 'is_question', 'is_command',
    }
    
    def _is_noise_fact(self, fact: Dict) -> bool:
        """Reject filler facts that don't represent real knowledge."""
        predicate = fact.get("predicate", "").lower()
        obj = fact.get("object", "").lower().strip()
        
        # Block noise predicates
        if predicate in self._NOISE_PREDICATES:
            logger.info(f"🗑 Noise filter: rejected {predicate} = {obj}")
            return True
        
        # Block if object is just a common greeting word
        greeting_objects = {'hi', 'hello', 'hey', 'bye', 'goodbye', 'thanks', 'introduce yourself',
                          'how are you', 'good morning', 'good evening', 'good night'}
        if obj in greeting_objects:
            logger.info(f"🗑 Noise filter: rejected greeting object '{obj}'")
            return True
        
        return False
    
    # ── Predicate normalization: disambiguate generic predicates ────────
    
    _OS_KEYWORDS = {'ubuntu', 'linux', 'windows', 'macos', 'fedora', 'debian', 'arch', 'mint',
                    'centos', 'manjaro', 'pop_os', 'kali', 'android', 'ios', 'chrome os'}
    _LANGUAGE_KEYWORDS = {'python', 'javascript', 'java', 'rust', 'go', 'c++', 'typescript',
                          'ruby', 'php', 'swift', 'kotlin', 'c#', 'scala', 'lua'}
    _TOOL_KEYWORDS = {'docker', 'git', 'vim', 'vscode', 'neovim', 'emacs', 'tmux',
                      'kubernetes', 'terraform', 'ansible', 'jenkins'}
    
    def _normalize_predicate(self, fact: Dict) -> Dict:
        """Map generic predicates to specific ones based on object content.
        
        Prevents collisions like 'user uses ubuntu' vs 'user uses ENML'
        by normalizing 'uses' + OS → 'uses_os', 'uses' + lang → 'has_skill', etc.
        """
        predicate = fact.get("predicate", "").lower()
        obj_lower = fact.get("object", "").lower()
        
        if predicate == "uses":
            obj_words = set(obj_lower.split())
            if obj_words & self._OS_KEYWORDS:
                fact["predicate"] = "uses_os"
            elif obj_words & self._LANGUAGE_KEYWORDS:
                fact["predicate"] = "has_skill"
            elif obj_words & self._TOOL_KEYWORDS:
                fact["predicate"] = "uses_tool"
            # else: leave as generic 'uses' (will be multi-value)
        
        elif predicate == "is_working_on" or predicate == "works_on":
            fact["predicate"] = "working_on"
        
        elif predicate == "has_favourite_os" or predicate == "has_favorite_os":
            fact["predicate"] = "uses_os"
        
        return fact
    
    # Regex patterns that indicate the user is naming/describing the AI, not themselves
    _AI_NAMING_PATTERNS = [
        re.compile(r'\byou\s+are\s+(?:now\s+)?(?:known\s+as\s+)?(?:called\s+)?(\w+)', re.IGNORECASE),
        re.compile(r'\byour\s+name\s+is\s+(\w+)', re.IGNORECASE),
        re.compile(r'\bcall\s+yourself\s+(\w+)', re.IGNORECASE),
        re.compile(r'\bfrom\s+(?:now|today)\s+.*\byou\b.*\b(?:are|known|called)\b', re.IGNORECASE),
        re.compile(r'\bi(?:\'ll| will)?\s+call\s+you\s+(\w+)', re.IGNORECASE),
        re.compile(r'\bname\s+(?:you|yourself)\s+(\w+)', re.IGNORECASE),
    ]
    
    def _fix_subject_misclassification(self, fact: Dict, original_input: str) -> Dict:
        """Regex safety net: corrects subject when the LLM misclassifies AI-naming as user-naming.
        
        If the original user message contains patterns like "you are Jarvis" or
        "your name is X" but the LLM output used subject="user", this corrects
        it to subject="assistant".
        """
        # Only apply to naming predicates with subject "user"
        naming_predicates = {'has_name', 'is_named', 'preferred_name', 'name', 'called', 'known_as'}
        
        if fact.get("subject", "").lower() != "user":
            return fact  # Already correct or third-party entity
        
        if fact.get("predicate", "").lower() not in naming_predicates:
            return fact  # Not a naming fact
        
        # Check if the original input was targeting the AI
        input_lower = original_input.lower()
        for pattern in self._AI_NAMING_PATTERNS:
            if pattern.search(input_lower):
                logger.info(f"Subject correction: '{fact['object']}' is AI name, not user name (matched: {pattern.pattern[:40]}...)")
                fact["subject"] = "assistant"
                return fact
        
        return fact

    # Known device/brand names that should NEVER be stored as a user's name
    _DEVICE_BRAND_KEYWORDS = {
        'lenovo', 'loq', 'thinkpad', 'ideapad', 'legion',
        'dell', 'xps', 'inspiron', 'latitude',
        'hp', 'elitebook', 'pavilion', 'omen',
        'asus', 'rog', 'vivobook', 'zenbook',
        'acer', 'predator', 'aspire', 'nitro',
        'apple', 'macbook', 'imac', 'mac',
        'samsung', 'galaxy', 'surface', 'microsoft',
        'nvidia', 'rtx', 'gtx', 'geforce',
        'intel', 'amd', 'ryzen', 'radeon',
        'ubuntu', 'windows', 'linux', 'macos',
    }
    
    def _guard_name_override(self, fact: Dict) -> bool:
        """Block device/brand names from being stored as `user has_name`.
        
        Returns True if the fact should be SKIPPED (blocked).
        Also attempts to reclassify obvious device-name facts.
        """
        subject = fact.get("subject", "").lower()
        predicate = fact.get("predicate", "").lower()
        obj = fact.get("object", "").strip()
        
        # Only guard against user has_name
        naming_predicates = {'has_name', 'is_named', 'preferred_name', 'name'}
        if subject != "user" or predicate not in naming_predicates:
            return False
        
        # Check if the object looks like a device/brand name
        obj_lower = obj.lower()
        obj_words = set(obj_lower.split())
        
        if obj_words & self._DEVICE_BRAND_KEYWORDS:
            logger.warning(f"🛡 Name guard BLOCKED: '{obj}' looks like a device/brand, not a person name")
            return True  # Skip this fact
        
        # Block very short values (single char or empty) as likely extraction errors
        if len(obj.strip()) <= 1:
            logger.warning(f"🛡 Name guard BLOCKED: '{obj}' is too short to be a name")
            return True
        
        return False


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