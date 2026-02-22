
import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# Define log directory
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

class MemoryLogger:
    _instance = None

    def __new__(cls, name="MemorySystem"):
        if cls._instance is None:
            cls._instance = super(MemoryLogger, cls).__new__(cls)
            cls._instance._initialize_logger(name)
        return cls._instance.logger

    def _initialize_logger(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. File Handler (Rotating)
        # Log up to 10MB per file, keep 5 backups
        log_file = LOG_DIR / "memory_system.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 3. JSON Lines Handler for structured auditing (Optional but good for parsing)
        json_log_file = LOG_DIR / "audit.jsonl"
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(json_handler)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        return json.dumps(log_record)

def get_logger(name=None):
    """Factory function to get a logger with the standard configuration."""
    # Use the root MemoryLogger configuration but return a child logger if name is provided
    root_logger = MemoryLogger() # Ensure root is configured
    if name:
        return root_logger.getChild(name)
    return root_logger
