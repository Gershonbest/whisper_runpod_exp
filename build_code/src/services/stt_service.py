"""Speech-to-Text service using Whisper."""
import torch
from typing import Optional, Tuple, List, Dict
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions

from utils.logger import get_logger
from utils.transcription_utils import TranscriptionUtils
from config import get_settings
from models import get_whisper_model

logger = get_logger(__name__)


class STTService:
    """Handles speech-to-text transcription using Whisper."""
    
    def __init__(self):
        self.settings = get_settings()
        self.transcription_utils = TranscriptionUtils()
        self._vad_options = None
    
    @property
    def vad_options(self) -> VadOptions:
        """Get VAD options."""
        if self._vad_options is None:
            self._vad_options = VadOptions(
                threshold=self.settings.vad_threshold,
                min_speech_duration_ms=self.settings.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.settings.vad_min_silence_duration_ms,
                speech_pad_ms=self.settings.vad_speech_pad_ms,
            )
        return self._vad_options
    
    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[str, List, str, float]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_file: Path to audio file
            language: Language code (ISO 639-1). Auto-detected if None.
            task: 'transcribe' or 'translate'
            
        Returns:
            Tuple of (text, segments, language, duration)
        """
        logger.info(f"Starting transcription (task={task}, language={language or 'auto'})...")
        
        model = get_whisper_model()
        
        # Set transcription parameters
        options_dict = {
            "task": task,
            "word_timestamps": True,
            "beam_size": self.settings.beam_size,
            "vad_filter": True,
            "vad_parameters": self.vad_options,
            "compression_ratio_threshold": self.settings.compression_ratio_threshold,
            "language_detection_threshold": self.settings.language_detection_threshold,
            "language_detection_segments": self.settings.language_detection_segments,
        }
        
        if language is not None:
            options_dict["language"] = language.lower()
        
        try:
            with torch.inference_mode():
                segment_generator, info = model.transcribe(audio_file, **options_dict)
                
                segments = []
                text = ""
                for segment in segment_generator:
                    segments.append(segment)
                    text += segment.text + " "
                
                text = text.strip()
            
            logger.info(
                f"Transcription completed: language={info.language}, "
                f"duration={info.duration:.2f}s, segments={len(segments)}"
            )
            
            return text, segments, info.language, info.duration
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_with_diarization(
        self,
        audio_file: str,
        diarization_df,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[str, str, str, float]:
        """
        Transcribe audio and combine with diarization results.
        
        Args:
            audio_file: Path to audio file
            diarization_df: DataFrame with diarization results
            language: Language code (optional)
            task: 'transcribe' or 'translate'
            
        Returns:
            Tuple of (diarized_text, text, language, duration)
        """
        # Get transcription
        text, segments, language, duration = self.transcribe(
            audio_file=audio_file,
            language=language,
            task=task
        )
        
        # Convert segments to dict format
        transcripts = self.transcription_utils.convert_faster_whisper_segments_to_dict(
            segments
        )
        
        # Convert to DataFrame
        whisper_df = self.transcription_utils.segment_to_dataframe(transcripts)
        
        # Combine with diarization
        if len(diarization_df) > 0:
            full_df = self.transcription_utils.combine_whisper_and_pyannote(
                whisper_df, diarization_df
            )
            
            if len(full_df) > 0:
                # Combine consecutive speakers
                combine_text = self.transcription_utils.combine_consecutive_speakers(
                    full_df
                )
                diarized_output = self.transcription_utils.format_diarized_text(
                    combine_text
                )
            else:
                logger.warning("No overlapping segments found, returning plain text")
                diarized_output = text
        else:
            logger.warning("No diarization data, returning plain text")
            diarized_output = text
        
        return diarized_output, text, language, duration
    
    def get_segments_dict(self, segments) -> List[Dict]:
        """
        Convert Whisper segments to dictionary format.
        
        Args:
            segments: Whisper segment generator or list
            
        Returns:
            List of segment dictionaries
        """
        return self.transcription_utils.convert_faster_whisper_segments_to_dict(
            segments
        )
