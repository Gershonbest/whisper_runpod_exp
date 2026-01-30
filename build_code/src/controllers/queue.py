"""Queue status routes."""
import asyncio
from fastapi import APIRouter, Request

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["queue"])


@router.get("/queue_status")
async def queue_status(request: Request):
    """Get the current queue status including pending requests and GPU slots."""
    gpu_semaphore = getattr(request.app.state, "gpu_semaphore", None)
    pending_lock = getattr(request.app.state, "pending_lock", None)
    max_concurrency = getattr(request.app.state, "max_concurrency", settings.max_concurrency)

    queue_size = 0
    if pending_lock is not None:
        async with pending_lock:
            queue_size = getattr(request.app.state, "pending_requests", 0)
    else:
        queue_size = getattr(request.app.state, "pending_requests", 0)

    # Semaphore._value is the number of available permits
    available_slots = gpu_semaphore._value if gpu_semaphore is not None else max_concurrency
    active_requests = max_concurrency - available_slots if gpu_semaphore is not None else 0

    return {
        "max_concurrency": max_concurrency,
        "available_slots": available_slots,
        "requests_in_queue": queue_size,
        "active_requests": active_requests,
    }


@router.get("/queue_size")
async def queue_size(request: Request):
    """Get the number of requests currently waiting in the queue."""
    pending_lock = getattr(request.app.state, "pending_lock", None)

    if pending_lock is not None:
        async with pending_lock:
            queue_size_value = getattr(request.app.state, "pending_requests", 0)
    else:
        queue_size_value = getattr(request.app.state, "pending_requests", 0)

    return {
        "requests_in_queue": queue_size_value,
    }
