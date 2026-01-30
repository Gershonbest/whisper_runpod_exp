"""Speaker diarization service."""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pyannote.audio import Pipeline

from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

# Try to import diarizers, fallback if not available
try:
    from diarizers import SegmentationModel
    DIARIZERS_AVAILABLE = True
except ImportError:
    logger.warning("diarizers package not available, using default segmentation")
    DIARIZERS_AVAILABLE = False


class DiarizationService:
    """Handles speaker diarization using Pyannote."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pipeline: Optional[Pipeline] = None
        self._initialized = False
    
    def _initialize_pipeline(self) -> None:
        """Initialize the diarization pipeline."""
        if self._initialized:
            return
        
        logger.info("Initializing diarization pipeline...")
        
        try:
            # Load pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.settings.diarization_model,
                use_auth_token=self.settings.hf_token
            )
            
            # Use fine-tuned segmentation model if available
            if DIARIZERS_AVAILABLE:
                try:
                    segmentation_model = SegmentationModel().from_pretrained(
                        self.settings.diarization_segmentation_model
                    )
                    model = segmentation_model.to_pyannote_model()
                    self.pipeline._segmentation.model = model
                    logger.info("Using fine-tuned segmentation model")
                except Exception as e:
                    logger.warning(f"Could not load fine-tuned segmentation model: {e}")
            
            # Configure pipeline hyperparameters
            self.pipeline.segmentation.threshold = 0.7
            self.pipeline.segmentation.min_duration_off = 0.2
            self.pipeline.segmentation.min_duration_on = 0.1
            self.pipeline.segmentation.offset = 0.5
            self.pipeline.segmentation.onset = 0.75
            self.pipeline.embedding_exclusive_overlap = True
            self.pipeline.clustering.threshold = 0.7
            self.pipeline.clustering.method = "centroid"
            self.pipeline.segmentation_step = 0.05
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline moved to GPU")
            
            self._initialized = True
            logger.info("Diarization pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            raise
    
    def diarize(
        self,
        waveform_dict: Dict,
        audio_file_path: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Perform speaker diarization on audio.
        
        Args:
            waveform_dict: Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
            audio_file_path: Optional path to audio file (preferred for Pyannote)
            num_speakers: Number of speakers (auto-detected if None)
            
        Returns:
            DataFrame with columns: index, start, end, speaker
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        num_speakers = num_speakers or self.settings.default_num_speakers
        
        logger.info(f"Starting diarization with {num_speakers} speakers...")
        
        try:
            torch.cuda.empty_cache()
            
            # Pyannote prefers file path, but can also work with waveform dict
            # Use file path if available, otherwise use waveform dict
            if audio_file_path:
                diarization_input = audio_file_path
            else:
                diarization_input = waveform_dict
            
            # Run diarization
            diarization_result = self.pipeline(
                diarization_input,
                num_speakers=num_speakers
            )
            
            torch.cuda.empty_cache()
            
            # Convert to DataFrame
            seg_info_list = []
            for speech_turn, track, speaker in diarization_result.itertracks(yield_label=True):
                # Format speaker label
                if speaker.startswith("SPEAKER_00"):
                    formatted_speaker = "SPEAKER_1"
                elif speaker.startswith("SPEAKER_0"):
                    speaker_num = int(speaker.split("_")[1])
                    formatted_speaker = f"SPEAKER_{speaker_num + 1}"
                else:
                    formatted_speaker = speaker
                
                segment_info = {
                    "start": np.round(speech_turn.start, 2),
                    "end": np.round(speech_turn.end, 2),
                    "speaker": formatted_speaker,
                }
                segment_info_df = pd.DataFrame.from_dict(
                    {track: segment_info},
                    orient="index"
                )
                seg_info_list.append(segment_info_df)
            
            if not seg_info_list:
                logger.warning("No diarization segments found")
                return pd.DataFrame(columns=["index", "start", "end", "speaker"])
            
            seg_info_df = pd.concat(seg_info_list, axis=0)
            seg_info_df = seg_info_df.reset_index()
            
            logger.info(f"Diarization completed: {len(seg_info_df)} segments found")
            
            return seg_info_df
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            torch.cuda.empty_cache()
            raise
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._initialized = False
            logger.info("Diarization pipeline cleaned up")
