"""Whisper model loading and management."""
import torch
from faster_whisper import WhisperModel
from typing import Optional

from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

# Global model instance
_whisper_model: Optional[WhisperModel] = None


class WhisperModelLoader:
    """Manages Whisper model loading and initialization."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[WhisperModel] = None
    
    def load(self) -> WhisperModel:
        """
        Load and return the Whisper model.
        
        Returns:
            Loaded WhisperModel instance
        """
        if self.model is not None:
            return self.model
        
        logger.info("Loading Whisper model...")
        
        # Determine device
        if self.settings.whisper_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.settings.whisper_device
        
        # Determine compute type
        if self.settings.whisper_compute_type == "auto":
            if device == "cuda":
                compute_type = "int8_float16"
            else:
                compute_type = "int8"
        else:
            compute_type = self.settings.whisper_compute_type
        
        logger.info(
            f"Loading model '{self.settings.whisper_model_size}' "
            f"on {device} with compute_type={compute_type}"
        )
        
        try:
            self.model = WhisperModel(
                self.settings.whisper_model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded")


def get_whisper_model() -> WhisperModel:
    """
    Get or load the global Whisper model instance.
    
    Returns:
        WhisperModel instance
    """
    global _whisper_model
    
    if _whisper_model is None:
        loader = WhisperModelLoader()
        _whisper_model = loader.load()
    
    return _whisper_model


def preload_whisper_model() -> None:
    """
    Preload the Whisper model (e.g. at startup to avoid cold start delays).
    Safe to call from a thread (e.g. via run_in_threadpool).
    """
    get_whisper_model()
