"""Logging configuration and utilities."""
import logging
import sys
from typing import Optional
from config import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """Setup logging configuration."""
    settings = get_settings()
    log_level = level or settings.log_level
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)
