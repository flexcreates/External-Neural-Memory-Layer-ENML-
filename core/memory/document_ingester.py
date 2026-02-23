"""
Document Ingestion Pipeline for ENML — LLM-Summarized Memory.

Handles bulk content (README, code docs, pasted documentation) by:
1. Splitting into logical sections (by markdown headings / paragraphs)
2. Filtering out noise (code blocks, ASCII art, pure syntax)
3. Using the LLM to generate STRUCTURED SUMMARIES per section → stored in document_collection
4. Extracting facts per-section and storing in knowledge_collection
5. Capping summaries and facts to prevent memory explosion
"""

import re
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI
from core.config import (
    MAX_DOCUMENT_FACTS, MAX_FACTS_PER_EXTRACTION,
    QDRANT_KNOWLEDGE_COLLECTION, QDRANT_DOCUMENT_COLLECTION,
    QDRANT_PROJECT_COLLECTION, QDRANT_RESEARCH_COLLECTION,
    MAX_DOCUMENT_SUMMARIES, LLAMA_SERVER_URL
)
from core.logger import get_logger

logger = get_logger("DocumentIngester")

# ── Min section size to summarize (skip tiny fragments) ──
MIN_SECTION_CHARS = 60

# Patterns for content that should NOT be sent to fact extraction
_CODE_FENCE_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
_INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
_URL_LINE_PATTERN = re.compile(r'^\s*https?://\S+\s*$', re.MULTILINE)
_ASCII_ART_PATTERN = re.compile(r'[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬]{3,}')
_FILE_PATH_PATTERN = re.compile(r'^\s*[\w./\\]+\.\w{1,5}\s*$', re.MULTILINE)
_TABLE_SEP_PATTERN = re.compile(r'^\s*\|[-:| ]+\|\s*$', re.MULTILINE)
_HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# ── Summarization prompt template ──
_SUMMARIZE_PROMPT = """You are a document analysis engine. Summarize the following section from a document.

RULES:
- Include ALL specific names, numbers, file paths, URLs, version numbers, and technical details
- Preserve the original structure (lists, hierarchies, relationships)
- Be detailed and complete — don't generalize or lose information
- If the section contains a file/folder tree, reproduce it exactly
- If the section has a table, reproduce the key rows
- Keep the summary focused and factual — no opinions or commentary
- Output ONLY the summary text, nothing else

Section heading: "{heading}"
Section content:
{content}

Detailed summary:"""


# ── Classification prompt template ──
_CLASSIFY_PROMPT = """Analyze the following document snippet and classify its primary nature.

Categories:
- "project": Codebase documentation, software architecture, technical READMEs, API docs, system specifications. 
- "research": Academic papers, theoretical articles, study findings, conceptual explainers, historical essays.
- "document": General text, chat logs, unstructured notes, or anything that doesn't strongly fit the above.

Output ONLY the exact category name ("project", "research", or "document") in lowercase. No other text.

Snippet:
{content}

Category:"""


class DocumentIngester:
    """Batch ingestion pipeline for large text inputs (documents, READMEs, etc.).
    
    Dual-layer storage:
    - Layer 1 (Summaries): LLM generates detailed summaries per section, stored
      in document_collection for rich RAG retrieval. This enables accurate answers
      about document content.
    - Layer 2 (Facts): Extract semantic triples and store in knowledge_collection
      for identity/attribute/relationship recall.
    """
    
    def __init__(self, memory_manager, llm_client: Optional[OpenAI] = None):
        """
        Args:
            memory_manager: The MemoryManager instance (for extractor + retriever access).
            llm_client: OpenAI-compatible client for LLM summarization calls.
                        If None, creates one using LLAMA_SERVER_URL.
        """
        self.memory_manager = memory_manager
        self.extractor = memory_manager.extractor
        self.retriever = memory_manager.retriever
        self.llm_client = llm_client or OpenAI(
            base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key"
        )
    
    def _summarize_section(self, heading: str, content: str) -> Optional[str]:
        """Use the LLM to generate a structured summary of a document section.
        
        Returns:
            Summary string, or None if summarization failed.
        """
        prompt = _SUMMARIZE_PROMPT.format(heading=heading, content=content[:2000])
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a precise document summarizer. Output only the summary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )
            summary = response.choices[0].message.content.strip()
            
            # Validate: reject empty or too-short summaries
            if not summary or len(summary) < 20:
                logger.warning(f"[STORE] Summary too short for '{heading[:40]}', skipping")
                return None
            
            logger.debug(f"[STORE] Summarized '{heading[:40]}': {len(summary)} chars")
            return summary
            
        except Exception as e:
            logger.error(f"[STORE] LLM summarization failed for '{heading[:40]}': {e}")
            return None
            
    def _classify_document(self, text: str) -> str:
        """Classify the entire document to determine which Qdrant collection to store its summaries in."""
        snippet = text[:2000] # Use first 2000 chars for classification
        prompt = _CLASSIFY_PROMPT.format(content=snippet)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a precise document classifier. Output only one word."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10,
                stream=False
            )
            category = response.choices[0].message.content.strip().lower()
            
            # Clean LLM output just in case
            category = re.sub(r'[^a-z]', '', category)
            
            if category == "project":
                return QDRANT_PROJECT_COLLECTION
            elif category == "research":
                return QDRANT_RESEARCH_COLLECTION
            else:
                return QDRANT_DOCUMENT_COLLECTION
                
        except Exception as e:
            logger.error(f"[CLASSIFY] LLM classification failed, defaulting to document: {e}")
            return QDRANT_DOCUMENT_COLLECTION
    
    def ingest(self, text: str, source_label: str = "pasted_document") -> Dict[str, Any]:
        """Ingest a large document through the summarization pipeline.
        
        Args:
            text: The full document text.
            source_label: Label for the source (e.g., "README.md", "pasted_document").
            
        Returns:
            Summary dict: {sections, summaries_stored, facts_extracted, skipped_noise, source}
        """
        logger.info(f"Starting document ingestion: {len(text)} chars, source={source_label}")
        
        # 0. Classify document to determine storage destination
        target_collection = self._classify_document(text)
        logger.info(f"Document classified for storage in: {target_collection}")
        
        # 1. Split into sections
        sections = self._split_into_sections(text)
        logger.info(f"Split into {len(sections)} sections")
        
        # 2. Process each section — summarize AND extract facts
        total_summaries = 0
        total_facts = 0
        skipped_sections = 0
        
        for i, section in enumerate(sections):
            heading = section.get("heading", f"section_{i}")
            
            # Clean the section
            cleaned = self._clean_section(section["content"])
            
            # Skip sections that are mostly noise after cleaning
            if not cleaned or len(cleaned.strip()) < 20:
                skipped_sections += 1
                continue
            
            # ── Layer 1: LLM-Summarized sections → document_collection ──
            if total_summaries < MAX_DOCUMENT_SUMMARIES and len(cleaned.strip()) >= MIN_SECTION_CHARS:
                summary = self._summarize_section(heading, cleaned)
                
                if summary:
                    # Prepend heading for better semantic search matching
                    if heading and heading != "(preamble)" and not heading.startswith("paragraph_"):
                        searchable_text = f"{heading}: {summary}"
                    else:
                        searchable_text = summary
                    
                    summary_payload = {
                        "text": summary,
                        "heading": heading,
                        "type": "document_summary",
                        "source": "document",
                        "source_label": source_label,
                        "chunk_index": i,
                        "char_count": len(summary),
                        "confidence": 0.85,
                        "timestamp": datetime.now().isoformat(),
                        "status": "active",
                    }
                    
                    self.retriever.add_memory(
                        collection=target_collection,
                        text=searchable_text,
                        payload=summary_payload,
                        memory_id=str(uuid.uuid4()),
                    )
                    total_summaries += 1
                    logger.info(f"[STORE] Summary {total_summaries} stored: '{heading[:40]}' ({len(summary)} chars)")
            
            # ── Layer 2: Extract facts (existing behavior) ──
            if total_facts >= MAX_DOCUMENT_FACTS:
                continue
            
            remaining_budget = MAX_DOCUMENT_FACTS - total_facts
            per_section_cap = min(5, remaining_budget)
            
            try:
                facts = self.extractor.extract_facts(
                    user_input=cleaned,
                    conversation_context="",
                    max_facts=per_section_cap
                )
            except Exception as e:
                logger.error(f"Extraction failed for section {i}: {e}")
                facts = []
            
            # Store each extracted fact
            for fact in facts:
                subject = fact.get("subject", "user").lower()
                predicate = fact.get("predicate", "").lower().replace(' ', '_')
                obj = fact.get("object", "")
                confidence = float(fact.get("confidence", 0.0))
                
                if not predicate or not obj:
                    continue
                
                payload = {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                    "source": "document",
                    "source_label": source_label,
                    "section_index": i,
                    "section_heading": heading,
                }
                payload["text"] = f"{subject} {predicate} {obj}."
                
                self.retriever.add_memory(
                    collection=QDRANT_KNOWLEDGE_COLLECTION,
                    text=payload["text"],
                    payload=payload
                )
                total_facts += 1
            
            if facts:
                logger.info(f"Section {i} ({heading[:40]}): {len(facts)} facts")
            
            if total_facts >= MAX_DOCUMENT_FACTS:
                logger.warning(f"Reached document fact cap ({MAX_DOCUMENT_FACTS}), stopping extraction")
        
        result = {
            "sections": len(sections),
            "summaries_stored": total_summaries,
            "facts_extracted": total_facts,
            "skipped_noise": skipped_sections,
            "source": source_label,
        }
        
        logger.info(f"Document ingestion complete: {result}")
        return result
    
    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into sections by markdown headings or double-newlines."""
        sections = []
        
        # Try splitting by markdown headings first
        heading_matches = list(_HEADING_PATTERN.finditer(text))
        
        if heading_matches:
            # Add content before first heading (if any)
            if heading_matches[0].start() > 0:
                pre_content = text[:heading_matches[0].start()].strip()
                if pre_content:
                    sections.append({"heading": "(preamble)", "content": pre_content})
            
            # Add each heading's section
            for i, match in enumerate(heading_matches):
                heading = match.group(2).strip()
                start = match.end()
                end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
                content = text[start:end].strip()
                if content:
                    sections.append({"heading": heading, "content": content})
        else:
            # No headings — split by double newlines
            paragraphs = re.split(r'\n\s*\n', text)
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para and len(para) > 15:
                    sections.append({"heading": f"paragraph_{i}", "content": para})
        
        return sections
    
    def _clean_section(self, text: str) -> str:
        """Remove noise from a section: code blocks, ASCII art, URLs, file paths, table separators."""
        cleaned = text
        
        # Remove code blocks (triple backtick)
        cleaned = _CODE_FENCE_PATTERN.sub(' ', cleaned)
        
        # Remove inline code
        cleaned = _INLINE_CODE_PATTERN.sub(lambda m: m.group(0).strip('`'), cleaned)
        
        # Remove URL-only lines
        cleaned = _URL_LINE_PATTERN.sub('', cleaned)
        
        # Remove ASCII art lines
        cleaned = _ASCII_ART_PATTERN.sub('', cleaned)
        
        # Remove file path lines
        cleaned = _FILE_PATH_PATTERN.sub('', cleaned)
        
        # Remove table separator lines
        cleaned = _TABLE_SEP_PATTERN.sub('', cleaned)
        
        # Remove markdown image/link syntax but keep alt text
        cleaned = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', cleaned)
        cleaned = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', cleaned)
        
        # Collapse multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
