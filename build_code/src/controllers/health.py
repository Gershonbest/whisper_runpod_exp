"""Health check routes."""
import torch
from fastapi import APIRouter, Request

from config import get_settings
from schemas.responses import HealthResponse
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
async def root(request: Request):
    """Root endpoint - returns health status."""
    return await health(request)


@router.get("/health", response_model=HealthResponse)
@router.get("/ping")
async def health(request: Request):
    """Health check endpoint."""
    try:
        transcription_service = getattr(request.app.state, "transcription_service", None)
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
