"""Main FastAPI application for the STT service."""
import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from utils.logger import setup_logging, get_logger
from utils.languages import SUPPORTED_LANGUAGES
from schemas.requests import TranscriptionRequest
from schemas.responses import TranscriptionResponse, HealthResponse, ErrorResponse
from services.transcription_service import TranscriptionService
from models import get_whisper_model
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress libmpg123 stderr warnings (non-fatal MP3 decoding errors)
# These warnings occur with some MP3 files but don't prevent processing
os.environ["MPG123_QUIET"] = "1"

# Redirect stderr at process level to suppress libmpg123 warnings globally
# This is a fallback if the context manager doesn't catch everything
import sys
if hasattr(sys.stderr, 'fileno'):
    try:
        # Open /dev/null for stderr redirection (will be restored by context managers)
        devnull = open(os.devnull, 'w')
        # Note: We don't redirect here permanently, just set up for context managers
        devnull.close()
    except Exception:
        pass  # Ignore if /dev/null is not available

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Global service instance
transcription_service: TranscriptionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global transcription_service
    
    # Startup
    logger.info("Starting STT service...")
    try:
        # Pre-load models
        logger.info("Pre-loading Whisper model...")
        get_whisper_model()
        logger.info("Models loaded successfully")
        
        # Initialize transcription service
        transcription_service = TranscriptionService()
        logger.info("STT service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down STT service...")
    transcription_service = None


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-ready Speech-to-Text service with multi-language support and speaker diarization",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - returns health status."""
    return await health()


@app.get("/health", response_model=HealthResponse)
@app.get("/ping")
async def health():
    """Health check endpoint."""
    try:
        models_loaded = transcription_service is not None
        gpu_available = torch.cuda.is_available()
        
        return HealthResponse(
            status="healthy" if models_loaded else "initializing",
            version=settings.api_version,
            models_loaded=models_loaded,
            gpu_available=gpu_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.api_version,
            models_loaded=False,
            gpu_available=torch.cuda.is_available()
        )


@app.get("/api/v1/languages")
async def get_languages():
    """Get list of supported languages."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "count": len(SUPPORTED_LANGUAGES)
    }


@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    """
    Transcribe audio file with optional speaker diarization.
    
    Supports:
    - Multi-language transcription (auto-detect or specify)
    - Speaker diarization
    - Translation to English
    - Custom number of speakers
    """
    if transcription_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready. Please try again in a moment."
        )
    
    # Validate request
    if not request.audio_url and not request.audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'audio_url' or 'audio_file' must be provided"
        )
    
    try:
        response = transcription_service.process(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@app.post("/invocations", response_model=TranscriptionResponse)
async def invocations(request: TranscriptionRequest):
    """
    Legacy endpoint for backward compatibility.
    """
    return await transcribe(request)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
