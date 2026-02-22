
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# Define log directory
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JsonFormatter(logging.Formatter):
    """Outputs structured JSON lines for machine-parseable audit logs."""
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


class _LoggerConfigurator:
    """Singleton that configures the root 'MemorySystem' logger once."""
    _initialized = False

    @classmethod
    def configure(cls):
        if cls._initialized:
            return
        cls._initialized = True

        root_logger = logging.getLogger("MemorySystem")
        root_logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers on reload
        if root_logger.handlers:
            return

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. File Handler (Rotating) — 10MB per file, 5 backups
        log_file = LOG_DIR / "memory_system.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 3. JSON Lines Handler for structured auditing
        json_log_file = LOG_DIR / "audit.jsonl"
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)


def get_logger(name: str = None) -> logging.Logger:
    """Factory function to get a properly-named child logger.

    All loggers are children of the 'MemorySystem' root logger, which
    is configured once with console, file, and JSON-lines handlers.
    Calling ``get_logger("core.extractor")`` returns
    ``logging.getLogger("MemorySystem.core.extractor")``.
    """
    _LoggerConfigurator.configure()
    root = logging.getLogger("MemorySystem")
    if name:
        return root.getChild(name)
    return root
