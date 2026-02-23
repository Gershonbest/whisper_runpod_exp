"""Request schemas for the STT service."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator


class TranscriptionRequest(BaseModel):
    """Request schema for transcription endpoint."""
    
    audio_url: Optional[HttpUrl] = Field(
        None,
        description="URL of the audio file to transcribe"
    )
    audio_file: Optional[str] = Field(
        None,
        description="Base64 encoded audio file (alternative to audio_url)"
    )
    language: Optional[str] = Field(
        None,
        description="Language code (ISO 639-1). If not provided, will be auto-detected."
    )
    task: str = Field(
        "transcribe",
        description="Task type: 'transcribe' or 'translate'"
    )
    enable_diarization: bool = Field(
        True,
        description="Whether to perform speaker diarization"
    )
    num_speakers: Optional[int] = Field(
        None,
        description="Number of speakers (for diarization). Auto-detected if not provided."
    )
    translate_to_english: bool = Field(
        False,
        description="Whether to translate non-English transcripts to English"
    )
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to include in the response"
    )
    dispatcher_endpoint: Optional[str] = Field(
        None,
        description="Optional endpoint to send results to after processing"
    )
    
    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        """Validate task type."""
        if v not in ["transcribe", "translate"]:
            raise ValueError("task must be either 'transcribe' or 'translate'")
        return v
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language code."""
        if v is not None:
            v = v.lower()
            # Basic validation - can be extended with full language list
            if len(v) not in [2, 3]:
                raise ValueError("Language code must be 2 or 3 characters (ISO 639-1 or ISO 639-2)")
        return v
    
    @field_validator("num_speakers")
    @classmethod
    def validate_num_speakers(cls, v: Optional[int]) -> Optional[int]:
        """Validate number of speakers."""
        if v is not None and v < 1:
            raise ValueError("num_speakers must be at least 1")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "https://2e97e207-backend.dataconect.com/api/v1/call-record-ext/documents/download/54e3ddb5-ad35-415a-8171-717420456940/CallRecord_1754597505874.mp3",
                "language": "en",
                "task": "transcribe",
                "enable_diarization": True,
                "num_speakers": 2,
                "translate_to_english": False,
                "extra_data": {}
            }
        }


class RunPodRequest(BaseModel):
    """Request schema for RunPod serverless handler."""
    
    input: TranscriptionRequest
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": {
                    "audio_url": "https://example.com/audio.mp3",
                    "language": "en",
                    "enable_diarization": True
                }
            }
        }
