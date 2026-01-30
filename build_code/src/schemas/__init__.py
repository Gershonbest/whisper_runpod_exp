"""Request and response schemas for the STT service."""
from .requests import TranscriptionRequest, RunPodRequest
from .responses import (
    TranscriptionResponse,
    TranscriptionSegment,
    DiarizedSegment,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    "TranscriptionRequest",
    "RunPodRequest",
    "TranscriptionResponse",
    "TranscriptionSegment",
    "DiarizedSegment",
    "HealthResponse",
    "ErrorResponse",
]
