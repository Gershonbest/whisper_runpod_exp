"""Audio processing utilities."""
import io
import os
import sys
import subprocess
import tempfile
import requests
import torchaudio
from typing import Tuple, Dict, Optional
from contextlib import contextmanager
from pydub import AudioSegment
from fastapi import HTTPException

from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

# Keep libmpg123 quiet globally, regardless of entrypoint
os.environ.setdefault("MPG123_QUIET", "1")
os.environ.setdefault("MPG123_VERBOSE", "0")
os.environ.setdefault("MPG123_IGNORE_STREAMERROR", "1")


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for libmpg123 warnings).
    
    Uses file descriptor redirection at the OS level to catch C library stderr output.
    """
    try:
        # Save original stderr file descriptor
        original_stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        # If stderr doesn't have a file descriptor (e.g., in some environments), skip suppression
        yield
        return
    
    # Create a duplicate of the original stderr to restore later
    saved_stderr_fd = os.dup(original_stderr_fd)
    
    # Open /dev/null for writing
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        # If /dev/null can't be opened, skip suppression
        os.close(saved_stderr_fd)
        yield
        return
    
    try:
        # Redirect stderr at the file descriptor level (catches C library output)
        os.dup2(devnull_fd, original_stderr_fd)
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


class AudioProcessor:
    """Handles audio file downloading, processing, and conversion."""
    
    def __init__(self):
        self.settings = get_settings()
        self.target_sample_rate = self.settings.target_sample_rate
        self.target_dbfs = self.settings.target_dbfs
    
    def reencode_audio(self, input_path: str, output_path: str) -> None:
        """
        Re-encode audio file using ffmpeg to fix corruption and suppress warnings.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            
        Raises:
            HTTPException: If re-encoding fails
        """
        try:
            logger.info(f"Re-encoding audio: {input_path} -> {output_path}")
            
            # Use ffmpeg to re-encode the audio
            # This fixes corrupted MP3 files and suppresses libmpg123 warnings
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', input_path,
                    '-acodec', 'libmp3lame',
                    '-b:a', '192k',
                    '-y',  # Overwrite output file if it exists
                    output_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL  # Suppress ffmpeg output
            )
            
            logger.info("Audio re-encoding completed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg re-encoding failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to re-encode audio: {str(e)}"
            )
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install ffmpeg.")
            raise HTTPException(
                status_code=500,
                detail="FFmpeg is required but not found. Please install ffmpeg."
            )
    
    def get_audio_data(self, audio_url: str) -> str:
        """
        Download audio file from URL using custom endpoint.
        This is the exact logic from the original implementation.
        
        Args:
            audio_url: URL of the audio file
            
        Returns:
            Path to temporary audio file
            
        Raises:
            HTTPException: If download fails
        """
        try:
            logger.info("Downloading Audio...")
            # Use custom endpoint first (original logic)
            response = requests.post(
                url="http://18.194.85.55:3001/voip-call-record/bite-data",
                json={"fileUrl": audio_url},
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            
            try:
                audio_data = response_data["data"]
                audio_bytes = bytes(audio_data)
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail="'data' key not found in response"
                )
                
        except requests.exceptions.ConnectionError:
            logger.error("connection error in downloading audio, check network connection of audio download server.")
            raise HTTPException(
                status_code=503,
                detail="connection error in downloading audio, check network connection of audio download server."
            )
        except requests.exceptions.Timeout:
            logger.error("request timeout in downloading audio, check network connection of audio download server.")
            raise HTTPException(
                status_code=504,
                detail="request timeout in downloading audio, check network connection of audio download server."
            )
        except requests.exceptions.HTTPError:
            logger.error("HTTP Error from audio_url")
            raise HTTPException(
                status_code=500,
                detail="HTTP Error from audio_url"
            )
        except requests.exceptions.RequestException:
            logger.error("An error occurred, check audio url")
            raise HTTPException(
                status_code=500,
                detail="An error occurred, check audio url"
            )

        # Save to temporary file (exact original logic)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            with open(temp_file.name, "wb") as f:
                f.write(audio_bytes)
                if os.path.exists(temp_file.name):
                    with open(temp_file.name, "rb") as f:
                        file_reader = io.BufferedReader(f)
                        audio_data = temp_file.name
                        temp_file.close()

        return audio_data
    
    def process_audio(
        self,
        audio_url: Optional[str] = None,
        target_dBFS: Optional[float] = None
    ) -> Tuple[str, Dict]:
        """
        Process audio file: download, normalize, and convert to tensor.
        This is the exact logic from the original implementation.
        
        Parameters:
        - audio_url (str): The URL of the audio file to be processed.
        - target_dBFS (float, optional): The target decibel-relative to 
                full scale (dBFS) for the audio volume. Default is -15.0.
        
        Returns:
        - tuple: A tuple containing the audio data file path and a 
                 dictionary with the processed waveform and sample rate.
        
        Raises:
            ValueError: If audio_url is not provided
            HTTPException: If audio processing fails
        """
        if audio_url is None:
            raise ValueError("audio_url must be provided")
        
        # Use default target_dBFS if not provided
        if target_dBFS is None:
            target_dBFS = self.target_dbfs
        
        # Get audio data using original logic
        audio_data = self.get_audio_data(audio_url=audio_url)
        if isinstance(audio_data, dict):
            error_message = audio_data.get("error", "Unknown error")
            raise ValueError(error_message)
        
        # Re-encode audio using ffmpeg to fix corruption and suppress libmpg123 warnings
        reencoded_file_path = None
        audio_file_path = audio_data
        try:
            reencoded_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            reencoded_file_path = reencoded_file.name
            reencoded_file.close()
            # Suppress stderr during re-encoding
            with suppress_stderr():
                self.reencode_audio(audio_data, reencoded_file_path)
            # Use re-encoded file for processing
            audio_file_path = reencoded_file_path
        except Exception as e:
            logger.warning(f"Audio re-encoding failed: {e}, using original file")
            # Fallback to original file if re-encoding fails
            audio_file_path = audio_data
        
        try:
            # Load audio using AudioSegment (exact original logic)
            # Suppress libmpg123 stderr warnings during loading
            # This is critical - libmpg123 writes directly to stderr
            with suppress_stderr():
                audio = AudioSegment.from_file(audio_file_path)

            # Compute gain to normalize audio (exact original logic)
            change_in_dBFS = target_dBFS - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dBFS)

            # Convert to WAV format in memory (exact original logic)
            # Suppress stderr during export as well
            temp_wav = io.BytesIO()
            with suppress_stderr():
                normalized_audio.export(temp_wav, format="wav")
            temp_wav.seek(0)

            # Load into TorchAudio tensor (exact original logic)
            waveform, sample_rate = torchaudio.load(temp_wav)
            
            # Resample if needed (exact original logic)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
                sample_rate = target_sample_rate
                
            waveform_sample_rate = {"waveform": waveform, "sample_rate": sample_rate}

            logger.info(
                f"Audio processed: duration={len(normalized_audio)/1000:.2f}s, "
                f"sample_rate={sample_rate}Hz"
            )

            return audio_data, waveform_sample_rate
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process audio: {str(e)}"
            )
        finally:
            # Clean up re-encoded file if it exists
            if reencoded_file_path and os.path.exists(reencoded_file_path):
                try:
                    os.unlink(reencoded_file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup re-encoded file: {e}")
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """Remove temporary file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
