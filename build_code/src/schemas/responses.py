"""Response schemas for the STT service."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A single transcription segment with timestamps."""
    
    id: int = Field(description="Segment ID")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")


class DiarizedSegment(BaseModel):
    """A diarized segment with speaker information."""
    
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    speaker: str = Field(description="Speaker identifier (e.g., 'SPEAKER_1')")
    text: str = Field(description="Transcribed text for this segment")


class TranscriptionResponse(BaseModel):
    """Response schema for transcription endpoint."""
    
    text: str = Field(
        default="",
        description="Full transcribed text without speaker labels"
    )
    diarized_text: str = Field(
        default="",
        description="Diarized transcript with speaker labels and timestamps"
    )
    translation: Optional[str] = Field(
        None,
        description="English translation (if translate_to_english is True)"
    )
    diarized_translation: Optional[str] = Field(
        None,
        description="Diarized English translation with speaker labels"
    )
    language: Optional[str] = Field(
        None,
        description="Detected or specified language code"
    )
    language_probability: Optional[float] = Field(
        None,
        description="Confidence score for language detection"
    )
    duration: Optional[float] = Field(
        None,
        description="Audio duration in seconds"
    )
    segments: Optional[List[TranscriptionSegment]] = Field(
        None,
        description="List of transcription segments with timestamps"
    )
    diarized_segments: Optional[List[DiarizedSegment]] = Field(
        None,
        description="List of diarized segments with speaker information"
    )
    num_speakers: Optional[int] = Field(
        None,
        description="Number of speakers detected"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Processing time in seconds"
    )
    cost: Optional[float] = Field(
        None,
        description="Estimated processing cost"
    )
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test transcription.",
                "diarized_text": "SPEAKER_1: [00:00:00 - 00:00:02] Hello, this is a test\nSPEAKER_2: [00:00:02 - 00:00:05] Yes, it is working.",
                "language": "en",
                "duration": 5.0,
                "num_speakers": 2,
                "processing_time": 2.5,
                "cost": 0.00175
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    models_loaded: bool = Field(description="Whether models are loaded")
    gpu_available: bool = Field(description="Whether GPU is available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "gpu_available": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(description="Error message")
    error_type: str = Field(description="Error type/class name")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Failed to download audio file",
                "error_type": "ValueError",
                "details": {"audio_url": "https://example.com/audio.mp3"}
            }
        }
