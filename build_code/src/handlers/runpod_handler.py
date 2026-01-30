"""
RunPod Serverless Handler for STT Service

This handler is designed to work with RunPod's serverless infrastructure.
It processes audio transcription requests with support for multi-language
transcription and speaker diarization.
"""
import json
import warnings
from typing import Dict, Any

from utils.logger import setup_logging, get_logger
from schemas.requests import TranscriptionRequest, RunPodRequest
from schemas.responses import TranscriptionResponse, ErrorResponse
from services.transcription_service import TranscriptionService

warnings.filterwarnings("ignore")

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global service instance (initialized on first use)
_transcription_service: TranscriptionService = None


def _get_service() -> TranscriptionService:
    """Get or create the transcription service instance."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.mp3",  # Required
            "language": "en",  # Optional, auto-detected if not provided
            "task": "transcribe",  # Optional, "transcribe" or "translate"
            "enable_diarization": true,  # Optional, defaults to True
            "num_speakers": 2,  # Optional, auto-detected if not provided
            "translate_to_english": false,  # Optional
            "extra_data": {},  # Optional
            "dispatcher_endpoint": "https://example.com/api"  # Optional
        }
    }
    
    Returns:
    {
        "text": "...",
        "diarized_text": "...",
        "translation": "...",
        "diarized_translation": "...",
        "language": "en",
        "duration": 123.45,
        "segments": [...],
        "diarized_segments": [...],
        "num_speakers": 2,
        "processing_time": 2.5,
        "cost": 0.00175,
        "extra_data": {}
    }
    """
    try:
        # Extract input data
        input_data = event.get("input", {})
        
        if not input_data:
            return {
                "error": "Missing 'input' field in request",
                "error_type": "ValueError"
            }
        
        # Validate required fields
        if not input_data.get("audio_url") and not input_data.get("audio_file"):
            return {
                "error": "Either 'audio_url' or 'audio_file' must be provided in the input",
                "error_type": "ValueError"
            }
        
        # Create request object
        try:
            request = TranscriptionRequest(**input_data)
        except Exception as e:
            return {
                "error": f"Invalid request parameters: {str(e)}",
                "error_type": type(e).__name__
            }
        
        # Process transcription
        logger.info(f"Processing transcription request: audio_url={request.audio_url}")
        service = _get_service()
        response = service.process(request)
        
        # Convert to dict for JSON serialization
        result = response.model_dump()
        
        logger.info(
            f"Transcription completed: language={result.get('language')}, "
            f"duration={result.get('duration')}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }


# For local testing
if __name__ == "__main__":
    # Example test event
    test_event = {
        "input": {
            "audio_url": "https://example.com/audio.mp3",
            "language": "en",
            "enable_diarization": True,
            "num_speakers": 2,
            "extra_data": {}
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
