"""Transcription routes."""
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from schemas.requests import TranscriptionRequest
from schemas.responses import TranscriptionResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["transcription"])


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
