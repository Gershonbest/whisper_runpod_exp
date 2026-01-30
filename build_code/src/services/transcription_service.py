"""Main transcription service orchestrator."""
import time
import math
from typing import Optional, Dict, Any
import requests

from utils.logger import get_logger
from utils.audio_processing import AudioProcessor
from services.stt_service import STTService
from services.diarization_service import DiarizationService
from schemas.requests import TranscriptionRequest
from schemas.responses import TranscriptionResponse, TranscriptionSegment, DiarizedSegment
from config import get_settings

logger = get_logger(__name__)


class TranscriptionService:
    """Orchestrates the complete transcription pipeline."""
    
    def __init__(self):
        self.settings = get_settings()
        self.audio_processor = AudioProcessor()
        self.stt_service = STTService()
        self.diarization_service = DiarizationService()
    
    def process(
        self,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Process a transcription request.
        
        Args:
            request: Transcription request
            
        Returns:
            Transcription response with results
        """
        start_time = time.perf_counter()
        audio_file_path = None
        
        try:
            # Process audio
            logger.info("Processing audio...")
            if not request.audio_url:
                raise HTTPException(
                    status_code=400,
                    detail="audio_url is required"
                )
            audio_file_path, waveform_dict = self.audio_processor.process_audio(
                audio_url=str(request.audio_url)
            )
            
            # Determine task
            task = request.task
            if request.translate_to_english:
                task = "translate"
            
            # Perform diarization if requested
            diarization_df = None
            num_speakers = None
            if request.enable_diarization:
                logger.info("Performing speaker diarization...")
                num_speakers = request.num_speakers or self.settings.default_num_speakers
                diarization_df = self.diarization_service.diarize(
                    waveform_dict,
                    audio_file_path=audio_file_path,
                    num_speakers=num_speakers
                )
            
            # Transcribe
            segments = None
            if request.enable_diarization and diarization_df is not None and len(diarization_df) > 0:
                diarized_text, text, language, duration = self.stt_service.process_with_diarization(
                    audio_file=audio_file_path,
                    diarization_df=diarization_df,
                    language=request.language,
                    task=task
                )
                # Get segments for response
                _, segments, _, _ = self.stt_service.transcribe(
                    audio_file=audio_file_path,
                    language=request.language,
                    task=task
                )
            else:
                text, segments, language, duration = self.stt_service.transcribe(
                    audio_file=audio_file_path,
                    language=request.language,
                    task=task
                )
                diarized_text = text
            
            # Convert segments to response format
            segments_list = None
            if segments:
                segments_list = [
                    TranscriptionSegment(
                        id=seg[0] - 1,
                        start=seg[2],
                        end=seg[3],
                        text=seg[4]
                    )
                    for seg in segments
                ]
            
            # Handle translation if needed
            translation = None
            diarized_translation = None
            if request.translate_to_english and language and language.lower() != "en":
                logger.info("Translating to English...")
                if request.enable_diarization and diarization_df is not None and len(diarization_df) > 0:
                    diarized_translation, translation, _, _ = self.stt_service.process_with_diarization(
                        audio_file=audio_file_path,
                        diarization_df=diarization_df,
                        language=language,
                        task="translate"
                    )
                else:
                    translation, _, _, _ = self.stt_service.transcribe(
                        audio_file=audio_file_path,
                        language=language,
                        task="translate"
                    )
                    diarized_translation = None
            
            # Calculate processing metrics
            processing_time = time.perf_counter() - start_time
            cost = math.ceil(processing_time) * self.settings.compute_rate_per_second
            
            # Build response
            response = TranscriptionResponse(
                text=text,
                diarized_text=diarized_text,
                translation=translation,
                diarized_translation=diarized_translation,
                language=language.upper() if language else None,
                duration=round(duration, 2) if duration else None,
                segments=segments_list,
                num_speakers=num_speakers,
                processing_time=round(processing_time, 2),
                cost=round(cost, 6),
                extra_data={
                    **request.extra_data,
                    "billing": {
                        "taskDuration": math.ceil(processing_time),
                        "taskCost": round(cost, 6)
                    }
                }
            )
            
            logger.info(
                f"Transcription completed: language={language}, "
                f"duration={duration:.2f}s, processing_time={processing_time:.2f}s"
            )
            
            # Send to dispatcher if provided
            if request.dispatcher_endpoint:
                self._send_to_dispatcher(request.dispatcher_endpoint, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            if audio_file_path:
                self.audio_processor.cleanup_temp_file(audio_file_path)
    
    def _send_to_dispatcher(
        self,
        dispatcher_endpoint: str,
        response: TranscriptionResponse
    ) -> None:
        """Send results to dispatcher endpoint."""
        try:
            dispatcher_url = f"{dispatcher_endpoint}/transcribtion/data"
            payload = {"data": response.model_dump()}
            
            logger.info(f"Sending results to dispatcher: {dispatcher_url}")
            requests.post(url=dispatcher_url, json=payload, timeout=30)
            logger.info("Results sent to dispatcher successfully")
            
        except Exception as e:
            logger.error(f"Failed to send results to dispatcher: {e}")
