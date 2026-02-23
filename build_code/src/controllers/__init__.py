"""Controllers (route handlers) for the STT service."""
from .health import router as health_router
from .transcription import router as transcription_router
from .languages import router as languages_router

__all__ = ["health_router", "transcription_router", "languages_router"]
