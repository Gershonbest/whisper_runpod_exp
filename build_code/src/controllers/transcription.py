"""Transcription routes."""
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from schemas.requests import TranscriptionRequest
from schemas.responses import TranscriptionResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["transcription"])


async def process_request_background(app, body: TranscriptionRequest):
    """
    Background task to process a transcription request.
    Runs with GPU semaphore to limit concurrency; does not block the HTTP response.
    """
    pending_lock = getattr(app.state, "pending_lock", None)
    gpu_semaphore = getattr(app.state, "gpu_semaphore", None)
    transcription_service = getattr(app.state, "transcription_service", None)

    if pending_lock is None or gpu_semaphore is None or transcription_service is None:
        logging.error("Queue or service not initialized")
        return

    # Increment pending counter before waiting for semaphore
    async with pending_lock:
        app.state.pending_requests += 1

    try:
        async with gpu_semaphore:
            # Decrement pending when we acquire a slot
            async with pending_lock:
                app.state.pending_requests -= 1

            await run_in_threadpool(
                transcription_service.process,
                body
            )
    except Exception as e:
        async with pending_lock:
            app.state.pending_requests -= 1
        logger.error(f"Error in background transcription task: {e}", exc_info=True)


@router.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: Request, body: TranscriptionRequest):
    """
    Transcribe audio (synchronous). Waits for a GPU slot then returns the result.
    Limited by max_concurrency to avoid overloading the server.
    """
    transcription_service = getattr(request.app.state, "transcription_service", None)
    gpu_semaphore = getattr(request.app.state, "gpu_semaphore", None)

    if transcription_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready. Please try again in a moment."
        )

    if not body.audio_url and not body.audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'audio_url' or 'audio_file' must be provided"
        )

    if gpu_semaphore is not None:
        async with gpu_semaphore:
            try:
                response = await run_in_threadpool(
                    transcription_service.process,
                    body
                )
                return response
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Transcription failed: {str(e)}"
                )

    try:
        response = await run_in_threadpool(transcription_service.process, body)
        return response
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/invocations")
async def invocations(request: Request, body: TranscriptionRequest):
    """
    Submit a transcription request. Returns immediately with status "accepted".
    Request is processed in the background; all GPU work goes through the semaphore.
    Results are sent to dispatcher_endpoint if provided in the body.
    """
    transcription_service = getattr(request.app.state, "transcription_service", None)

    if transcription_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready. Please try again in a moment."
        )

    if not body.audio_url and not body.audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'audio_url' or 'audio_file' must be provided"
        )

    pending_lock = getattr(request.app.state, "pending_lock", None)
    current_queue_size = 0
    if pending_lock is not None:
        async with pending_lock:
            current_queue_size = request.app.state.pending_requests

    asyncio.create_task(process_request_background(request.app, body))

    return {
        "status": "accepted",
        "message": "Request queued for processing",
        "queue_position": current_queue_size + 1,
    }
