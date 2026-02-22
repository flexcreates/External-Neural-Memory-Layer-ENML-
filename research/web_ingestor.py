
import requests
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict
from core.logger import get_logger

logger = get_logger(__name__)

class WebIngestor:
    def __init__(self):
        pass
        
    def fetch_page(self, url: str) -> Optional[str]:
        try:
            headers = {"User-Agent": "ENML-Research-Bot/1.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract_content(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            
            # Clean whitespace
            # Replace multiple newlines with single newline
            text = re.sub(r'\n+', '\n', text)
            # Replace multiple spaces with single space within lines
            lines = []
            for line in text.splitlines():
                clean_line = re.sub(r'\s+', ' ', line).strip()
                if clean_line:
                    lines.append(clean_line)
            
            return '\n'.join(lines)
        except Exception as e:
            logger.error(f"Failed to extract content: {e}")
            return ""

    def ingest_url(self, url: str):
        """
        Fetches, cleans, and (TODO) chunks/embeds content.
        Currently returns cleaned text.
        """
        html = self.fetch_page(url)
        if not html:
            return None
            
        text = self.extract_content(html)
        logger.info(f"Ingested {len(text)} chars from {url}")
        return text
