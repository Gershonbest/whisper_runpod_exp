"""Main FastAPI application for the STT service."""
import asyncio
import json
import os
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import redis.asyncio as redis
from redis.exceptions import AuthenticationError, RedisError
from fastapi import FastAPI, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from config import get_settings
from utils.logger import setup_logging, get_logger
from schemas.requests import TranscriptionRequest
from schemas.responses import ErrorResponse
from services.transcription_service import TranscriptionService
from models import get_whisper_model, preload_whisper_model
from controllers import health_router, transcription_router, languages_router
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress libmpg123 stderr warnings (non-fatal MP3 decoding errors)
os.environ["MPG123_QUIET"] = "1"

# Setup logging
setup_logging()
logger = get_logger(__name__)


settings = get_settings()

MAX_CONCURRENCY = settings.max_concurrency
QUEUE_WORKER_ENABLED = settings.queue_worker_enabled
REDIS_URL = settings.redis_url
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
QUEUE_NAME = settings.queue_name
QUEUE_BRPOP_TIMEOUT = settings.queue_brpop_timeout

transcription_service: TranscriptionService = None
redis_client: Optional[redis.Redis] = None
queue_worker_task: Optional[asyncio.Task] = None


def _redacted_redis_target(redis_url: str) -> str:
    parsed = urlsplit(redis_url)
    host = parsed.hostname or "localhost"
    port = f":{parsed.port}" if parsed.port else ""
    db_path = parsed.path or ""
    return f"{parsed.scheme}://{host}{port}{db_path}"


def _build_redis_client() -> redis.Redis:
    connection_kwargs: Dict[str, str] = {}
    if REDIS_USERNAME:
        connection_kwargs["username"] = REDIS_USERNAME
    if REDIS_PASSWORD:
        connection_kwargs["password"] = REDIS_PASSWORD
    return redis.from_url(REDIS_URL, **connection_kwargs)


async def _run_job_with_slot(app: FastAPI, job_id: str, request_model: TranscriptionRequest) -> None:
    """Run a queued transcription job while holding a GPU slot."""
    semaphore: asyncio.Semaphore = app.state.gpu_semaphore
    service: TranscriptionService = app.state.transcription_service

    try:
        logger.info("Processing queued job %s", job_id)
        await run_in_threadpool(service.process, request_model)
        logger.info("Finished queued job %s", job_id)
    except Exception as exc:
        logger.error("Queued job %s failed: %s", job_id, exc, exc_info=True)
    finally:
        semaphore.release()


async def queue_worker(app: FastAPI) -> None:
    """Continuously consume jobs from Redis and dispatch to the transcription service."""
    if redis_client is None:
        raise RuntimeError("Redis client is not initialized")

    semaphore: asyncio.Semaphore = app.state.gpu_semaphore

    try:
        while True:
            await semaphore.acquire()
            job_data = await redis_client.brpop(QUEUE_NAME, timeout=QUEUE_BRPOP_TIMEOUT)

            if job_data:
                _, payload = job_data
                try:
                    job = json.loads(payload.decode())
                    job_id = str(job.get("job_id", "<unknown>"))
                    request_payload: Dict[str, Any] = job.get("request", job)
                    request_model = TranscriptionRequest(**request_payload)
                except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
                    logger.error("Invalid job payload received from queue: %s", exc, exc_info=True)
                    semaphore.release()
                    continue

                asyncio.create_task(_run_job_with_slot(app, job_id, request_model))
            else:
                semaphore.release()
    except asyncio.CancelledError:
        logger.info("Queue worker cancelled; shutting down consumer loop")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global transcription_service, redis_client, queue_worker_task

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

    # GPU concurrency: semaphore limits concurrent GPU work
    app.state.gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    app.state.max_concurrency = MAX_CONCURRENCY

    if QUEUE_WORKER_ENABLED:
        logger.info("Queue worker enabled; connecting to Redis at %s", _redacted_redis_target(REDIS_URL))
        redis_client = _build_redis_client()
        try:
            await redis_client.ping()
        except AuthenticationError as exc:
            redis_client = None
            logger.error(
                "Redis authentication failed. Set REDIS_USERNAME/REDIS_PASSWORD (or USERNAME/PASSWORD) and verify credentials."
            )
            raise RuntimeError("Redis authentication failed") from exc
        except RedisError as exc:
            redis_client = None
            logger.error("Unable to reach Redis at %s: %s", _redacted_redis_target(REDIS_URL), exc)
            raise RuntimeError("Redis connection failed") from exc

        queue_worker_task = asyncio.create_task(queue_worker(app))
        logger.info("Redis queue worker started (queue=%s)", QUEUE_NAME)

    logger.info("Application ready to accept requests")

    yield

    # ---- shutdown ----
    logger.info("Shutting down application...")
    if queue_worker_task is not None:
        queue_worker_task.cancel()
        try:
            await queue_worker_task
        except asyncio.CancelledError:
            pass
    if redis_client is not None:
        await redis_client.aclose()
        redis_client = None
    queue_worker_task = None
    transcription_service = None
    app.state.transcription_service = None
    app.state.gpu_semaphore = None


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-ready Speech-to-Text service with multi-language support, speaker diarization, and GPU concurrency control",
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
