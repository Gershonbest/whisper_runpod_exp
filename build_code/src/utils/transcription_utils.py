"""Transcription utility functions."""
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# Fix for numpy compatibility with pyannote-audio
if not hasattr(np, 'NAN'):
    np.NAN = np.nan


class TranscriptionUtils:
    """Utility functions for processing transcription results."""
    
    @staticmethod
    def convert_faster_whisper_segments_to_dict(segments) -> List[Dict]:
        """
        Convert faster-whisper segments to dictionary format.
        
        Args:
            segments: Faster-whisper segment generator or list
            
        Returns:
            List of segment dictionaries with id, start, end, text
        """
        openai_segments = []
        for segment in segments:
            id, _, start, end, text, _, _, _, _, _, words = segment
            openai_segments.append({
                "id": id - 1,
                "start": start,
                "end": end,
                "text": text
            })
        return openai_segments
    
    @staticmethod
    def segment_to_dataframe(transcription_result: List[Dict]) -> pd.DataFrame:
        """
        Convert segments to pandas DataFrame.
        
        Args:
            transcription_result: List of segment dictionaries
            
        Returns:
            DataFrame with columns: id, start, end, text
        """
        df = pd.DataFrame(transcription_result, columns=["id", "start", "end", "text"])
        df["start"] = df["start"].apply(lambda x: round(x, 2))
        df["end"] = df["end"].apply(lambda x: round(x, 2))
        return df
    
    @staticmethod
    def combine_whisper_and_pyannote(
        text_df: pd.DataFrame,
        speaker_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine Whisper transcription segments with Pyannote speaker diarization.
        
        Args:
            text_df: DataFrame with transcription segments (id, start, end, text)
            speaker_df: DataFrame with speaker segments (index, start, end, speaker)
            
        Returns:
            Combined DataFrame with speaker assignments
        """
        text_df = text_df.loc[:, ["id", "start", "end", "text"]]
        speaker_df = speaker_df.loc[:, ["index", "start", "end", "speaker"]]
        
        overlap_list = []
        for idx, pyannote_row in speaker_df.iterrows():
            pyannote_start = pyannote_row["start"]
            pyannote_end = pyannote_row["end"]
            pyannote_speaker = pyannote_row["speaker"]
            
            # Find overlapping segments
            overlap_mask = ~(
                (text_df["end"] < pyannote_start) | (text_df["start"] > pyannote_end)
            )
            this_overlap_texts = text_df.loc[overlap_mask, :].copy()
            
            if len(this_overlap_texts) == 0:
                continue
            
            this_overlap_texts["speaker_start"] = pyannote_start
            this_overlap_texts["speaker_end"] = pyannote_end
            this_overlap_texts["speaker"] = pyannote_speaker
            
            overlap_list.append(this_overlap_texts)
        
        if not overlap_list:
            logger.warning("No overlapping segments found between transcription and diarization")
            return pd.DataFrame()
        
        all_overlaps = pd.concat(overlap_list)
        all_overlaps = all_overlaps.reset_index(drop=True)
        
        # Calculate overlap duration
        all_overlaps["max_start"] = np.maximum(
            all_overlaps["start"], all_overlaps["speaker_start"]
        )
        all_overlaps["min_end"] = np.minimum(
            all_overlaps["end"], all_overlaps["speaker_end"]
        )
        all_overlaps["overlap_duration"] = (
            all_overlaps["min_end"] - all_overlaps["max_start"]
        )
        
        # Select segment with maximum overlap for each transcription segment
        max_overlap_indices = all_overlaps.groupby("id")["overlap_duration"].idxmax()
        text_speaker_df = all_overlaps.loc[max_overlap_indices, :]
        
        return text_speaker_df
    
    @staticmethod
    def combine_consecutive_speakers(text_speaker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine consecutive segments from the same speaker.
        
        Args:
            text_speaker_df: DataFrame with speaker assignments
            
        Returns:
            DataFrame with consecutive segments combined
        """
        text_speaker_df = text_speaker_df.copy()
        n_iter = text_speaker_df.shape[0]
        
        for counter in range(1, n_iter):
            is_same_speaker = (
                text_speaker_df["speaker"].iloc[counter]
                == text_speaker_df["speaker"].iloc[counter - 1]
            )
            
            if is_same_speaker:
                new_start = text_speaker_df["start"].iloc[counter - 1]
                previous_text = text_speaker_df["text"].iloc[counter - 1]
                new_text = previous_text + " " + text_speaker_df["text"].iloc[counter]
                
                text_speaker_df["start"].iloc[counter] = new_start
                text_speaker_df["text"].iloc[counter] = new_text
                text_speaker_df["start"].iloc[counter - 1] = np.nan
                text_speaker_df["end"].iloc[counter - 1] = np.nan
        
        text_speaker_df = text_speaker_df.dropna().loc[
            :, ["start", "end", "text", "speaker"]
        ]
        text_speaker_df = text_speaker_df.reset_index(drop=True)
        text_speaker_df = text_speaker_df.sort_values("start")
        
        return text_speaker_df
    
    @staticmethod
    def format_diarized_text(text_speaker_df: pd.DataFrame) -> str:
        """
        Format diarized segments as text with timestamps.
        
        Args:
            text_speaker_df: DataFrame with diarized segments
            
        Returns:
            Formatted string with speaker labels and timestamps
        """
        output_str = ""
        
        for idx, row in text_speaker_df.iterrows():
            start_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(np.round(row['start'], 2))
            )
            end_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(np.round(row['end'], 2))
            )
            speaker = row["speaker"]
            text = row["text"]
            
            output_str += f'{speaker}: [{start_time} - {end_time}]--{text}\n'
        
        return output_str
    
    @staticmethod
    def format_segments_with_timestamps(segments: List[Dict]) -> str:
        """
        Format segments with timestamps.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Formatted string with timestamps
        """
        transcript = ""
        for segment in segments:
            start_time = time.strftime(
                "%H:%M:%S",
                time.gmtime(segment["start"])
            )
            transcript += f"\n[{start_time}] {segment['text']} "
        
        return transcript.strip()
