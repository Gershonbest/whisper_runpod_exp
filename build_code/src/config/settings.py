"""Application settings and configuration."""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
dir_path = (Path(__file__) / ".." / ".." / ".." / "..").resolve()
env_path = os.path.join(dir_path, ".env")
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # HuggingFace token for model access
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    
    def __post_init__(self):
        """Set HF_TOKEN environment variable if provided."""
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
    
    # Compute rate for billing (per second)
    compute_rate_per_second: float = float(os.getenv("COMPUTE_RATE_PER_SECOND", "0.0007"))
    
    # Whisper model configuration
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "distil-large-v3")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")  # auto, cuda, cpu
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto, int8, float16, etc.
    
    # Diarization configuration
    diarization_model: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    default_num_speakers: int = int(os.getenv("DEFAULT_NUM_SPEAKERS", "2"))
    diarization_segmentation_model: str = os.getenv(
        "DIARIZATION_SEGMENTATION_MODEL",
        "diarizers-community/speaker-segmentation-fine-tuned-callhome-eng"
    )
    
    # VAD (Voice Activity Detection) options
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.25"))
    vad_min_speech_duration_ms: int = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "50"))
    vad_min_silence_duration_ms: int = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "500"))
    vad_speech_pad_ms: int = int(os.getenv("VAD_SPEECH_PAD_MS", "1000"))
    
    # Transcription options
    beam_size: int = int(os.getenv("BEAM_SIZE", "1"))
    compression_ratio_threshold: float = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "3.0"))
    language_detection_threshold: float = float(os.getenv("LANGUAGE_DETECTION_THRESHOLD", "0.5"))
    language_detection_segments: int = int(os.getenv("LANGUAGE_DETECTION_SEGMENTS", "5"))
    
    # Audio processing
    target_sample_rate: int = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
    target_dbfs: float = float(os.getenv("TARGET_DBFS", "-15.0"))
    
    # API configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_title: str = os.getenv("API_TITLE", "Speech-to-Text Service")
    api_version: str = os.getenv("API_VERSION", "1.0.0")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Batch processing (if needed)
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "6"))
    batch_timeout: float = float(os.getenv("BATCH_TIMEOUT", "0.07"))
    
    # Queue / concurrency: max concurrent GPU requests (prevents server overload)
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "5"))


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
