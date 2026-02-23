"""
Document Ingestion Pipeline for ENML.

Handles bulk content (README, code docs, pasted documentation) by:
1. Splitting into logical sections (by markdown headings / paragraphs)
2. Filtering out noise (code blocks, ASCII art, pure syntax)
3. Extracting facts per-section with caps to prevent memory explosion
4. Storing in Qdrant with 'document' source metadata
"""

import re
from typing import Dict, Any, List
from datetime import datetime
from core.config import MAX_DOCUMENT_FACTS, MAX_FACTS_PER_EXTRACTION, QDRANT_KNOWLEDGE_COLLECTION
from core.logger import get_logger

logger = get_logger("DocumentIngester")

# Patterns for content that should NOT be sent to fact extraction
_CODE_FENCE_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
_INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
_URL_LINE_PATTERN = re.compile(r'^\s*https?://\S+\s*$', re.MULTILINE)
_ASCII_ART_PATTERN = re.compile(r'[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬]{3,}')
_FILE_PATH_PATTERN = re.compile(r'^\s*[\w./\\]+\.\w{1,5}\s*$', re.MULTILINE)
_TABLE_SEP_PATTERN = re.compile(r'^\s*\|[-:| ]+\|\s*$', re.MULTILINE)
_HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


class DocumentIngester:
    """Batch ingestion pipeline for large text inputs (documents, READMEs, etc.).
    
    Instead of sending raw text through real-time LLM extraction (which causes
    memory explosion), this class:
    - Chunks by markdown sections
    - Cleans noise from each section
    - Extracts limited facts per section
    - Stores with document-source metadata
    """
    
    def __init__(self, memory_manager):
        """
        Args:
            memory_manager: The MemoryManager instance (for extractor + retriever access).
        """
        self.memory_manager = memory_manager
        self.extractor = memory_manager.extractor
        self.retriever = memory_manager.retriever
    
    def ingest(self, text: str, source_label: str = "pasted_document") -> Dict[str, Any]:
        """Ingest a large document through the batch pipeline.
        
        Args:
            text: The full document text.
            source_label: Label for the source (e.g., "README.md", "pasted_document").
            
        Returns:
            Summary dict: {sections, facts_extracted, skipped_noise, source}
        """
        logger.info(f"Starting document ingestion: {len(text)} chars, source={source_label}")
        
        # 1. Split into sections
        sections = self._split_into_sections(text)
        logger.info(f"Split into {len(sections)} sections")
        
        # 2. Process each section
        total_facts = 0
        skipped_sections = 0
        
        for i, section in enumerate(sections):
            if total_facts >= MAX_DOCUMENT_FACTS:
                logger.warning(f"Reached document fact cap ({MAX_DOCUMENT_FACTS}), stopping extraction")
                break
            
            # Clean the section
            cleaned = self._clean_section(section["content"])
            
            # Skip sections that are mostly noise after cleaning
            if not cleaned or len(cleaned.strip()) < 20:
                skipped_sections += 1
                continue
            
            # Extract facts from this section (with per-section cap)
            remaining_budget = MAX_DOCUMENT_FACTS - total_facts
            per_section_cap = min(5, remaining_budget)
            
            try:
                facts = self.extractor.extract_facts(
                    user_input=cleaned,
                    conversation_context="",  # No conversation context for documents
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
                    "section_heading": section.get("heading", ""),
                }
                payload["text"] = f"{subject} {predicate} {obj}."
                
                self.retriever.add_memory(
                    collection=QDRANT_KNOWLEDGE_COLLECTION,
                    text=payload["text"],
                    payload=payload
                )
                total_facts += 1
            
            if facts:
                logger.info(f"Section {i} ({section.get('heading', 'untitled')[:40]}): {len(facts)} facts")
        
        result = {
            "sections": len(sections),
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
