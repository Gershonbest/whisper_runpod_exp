"""Main FastAPI application for the STT service."""
import asyncio
import os
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from utils.logger import setup_logging, get_logger
from schemas.responses import ErrorResponse
from services.transcription_service import TranscriptionService
from models import get_whisper_model, preload_whisper_model
from controllers import health_router, transcription_router, languages_router, queue_router
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress libmpg123 stderr warnings (non-fatal MP3 decoding errors)
os.environ["MPG123_QUIET"] = "1"

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Max concurrent GPU requests (queuing to avoid server overload)
MAX_CONCURRENCY = settings.max_concurrency

# Global service instance (used by controllers)
transcription_service: TranscriptionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global transcription_service

    # ---- startup ----
    logger.info("Starting application...")
    logger.info("Preloading Whisper model (this may take a moment)...")
    try:
        await run_in_threadpool(preload_whisper_model)
        logger.info("Whisper model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {e}", exc_info=True)
        raise

    # Initialize transcription service
    transcription_service = TranscriptionService()
    app.state.transcription_service = transcription_service

    # Queue state: semaphore limits concurrent GPU work; pending_requests counts queued items
    app.state.gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    app.state.pending_lock = asyncio.Lock()
    app.state.pending_requests = 0
    app.state.max_concurrency = MAX_CONCURRENCY

    logger.info("Application ready to accept requests")

    yield

    # ---- shutdown ----
    logger.info("Shutting down application...")
    transcription_service = None
    app.state.transcription_service = None
    app.state.gpu_semaphore = None
    app.state.pending_lock = None
    app.state.pending_requests = 0


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-ready Speech-to-Text service with multi-language support, speaker diarization, and request queuing to avoid server overload",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (controllers)
app.include_router(health_router)
app.include_router(transcription_router)
app.include_router(languages_router)
app.include_router(queue_router)


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
