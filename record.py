import os
import time
from typing import Optional, Tuple, Literal
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from enum import Enum, auto
from time import sleep

class AudioRecorderError(Exception):
    """Base exception class for audio recorder related errors."""
    pass


class InvalidConfigurationError(AudioRecorderError):
    """Raised when invalid configuration parameters are provided."""
    pass


class RecordingType(Enum):
    """Enumeration of supported recording types."""
    CLAP = auto()
    BACKGROUND_NOISE = auto()


class AudioRecorder:
    """
    A class for recording audio chunks with configurable parameters.
    
    This class provides functionality to record audio in chunks with specified duration,
    sample rate, and number of samples. It handles the audio stream and saves the
    recorded chunks to WAV files in a specified directory.
    
    Attributes:
        chunk_duration (float): Duration of each audio chunk in seconds
        num_samples (int): Total number of samples to record
        sample_rate (int): Audio sample rate in Hz
        dtype (np.dtype): Data type for audio samples
        base_directory (str): Base directory for saving recordings
        recording_type (RecordingType): Type of recording (clap or background noise)
    """
    
    def __init__(
        self,
        recording_type: RecordingType = RecordingType.CLAP,
        chunk_duration: float = 1.0,
        num_samples: int = 300,
        sample_rate: int = 44100,
        dtype: np.dtype = np.int16,
        base_directory: str = "data"
    ):
        """
        Initialize the AudioRecorder with specified parameters.
        
        Args:
            recording_type (RecordingType): Type of recording (clap or background noise)
            chunk_duration (float): Duration of each audio chunk in seconds
            num_samples (int): Total number of samples to record
            sample_rate (int): Audio sample rate in Hz
            dtype (np.dtype): Data type for audio samples
            base_directory (str): Base directory for saving recordings
            
        Raises:
            InvalidConfigurationError: If any of the parameters are invalid
        """
        self._validate_parameters(chunk_duration, num_samples, sample_rate)
        
        self.recording_type = recording_type
        self.chunk_duration = chunk_duration
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.base_directory = base_directory
        self.directory = self._get_recording_directory()
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.stream: Optional[sd.InputStream] = None
        
        self._create_directory()
    
    def _get_recording_directory(self) -> str:
        """
        Get the appropriate directory based on recording type.
        
        Returns:
            str: Path to the recording directory
        """
        if self.recording_type == RecordingType.CLAP:
            return os.path.join(self.base_directory, "claps")
        else:
            return os.path.join(self.base_directory, "background_noise")
    
    def _validate_parameters(
        self,
        chunk_duration: float,
        num_samples: int,
        sample_rate: int
    ) -> None:
        """
        Validate the input parameters.
        
        Args:
            chunk_duration (float): Duration of each audio chunk
            num_samples (int): Total number of samples
            sample_rate (int): Audio sample rate
            
        Raises:
            InvalidConfigurationError: If any parameter is invalid
        """
        assert isinstance(chunk_duration, (int, float)), "chunk_duration must be numeric"
        assert isinstance(num_samples, int), "num_samples must be an integer"
        assert isinstance(sample_rate, int), "sample_rate must be an integer"
        
        if chunk_duration <= 0:
            raise InvalidConfigurationError("chunk_duration must be positive")
        if num_samples <= 0:
            raise InvalidConfigurationError("num_samples must be positive")
        if sample_rate <= 0:
            raise InvalidConfigurationError("sample_rate must be positive")
    
    def _create_directory(self) -> None:
        """Create the output directory if it doesn't exist."""
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError as e:
            raise AudioRecorderError(f"Failed to create directory: {e}")
    
    def _setup_stream(self) -> None:
        """Set up the audio input stream."""
        try:
            self.stream = sd.InputStream(
                callback=None,
                channels=1,
                samplerate=self.sample_rate,
                dtype=self.dtype
            )
        except sd.PortAudioError as e:
            raise AudioRecorderError(f"Failed to initialize audio stream: {e}")
    
    def _record_chunk(self) -> Tuple[np.ndarray, bool]:
        """
        Record a single audio chunk.
        
        Returns:
            Tuple[np.ndarray, bool]: Recorded audio data and overflow status
            
        Raises:
            AudioRecorderError: If recording fails
        """
        try:
            chunk, overflowed = self.stream.read(self.chunk_samples)
            return chunk, overflowed
        except sd.PortAudioError as e:
            raise AudioRecorderError(f"Failed to record audio chunk: {e}")
    
    def _save_chunk(self, chunk: np.ndarray, index: int) -> None:
        """
        Save an audio chunk to a WAV file.
        
        Args:
            chunk (np.ndarray): Audio data to save
            index (int): Index of the chunk
            
        Raises:
            AudioRecorderError: If saving fails
        """
        try:
            prefix = "clap" if self.recording_type == RecordingType.CLAP else "bg"
            filename = os.path.join(self.directory, f"{prefix}_{index}.wav")
            write(filename, self.sample_rate, chunk)
        except Exception as e:
            raise AudioRecorderError(f"Failed to save audio chunk: {e}")
    
    def record(self) -> None:
        """
        Record multiple audio chunks and save them to files.
        
        Raises:
            AudioRecorderError: If recording fails at any point
        """
        try:
            self._setup_stream()
            
            with self.stream:
                print(f"Recording {self.recording_type.name.lower()}...")
                for i in range(self.num_samples):
                    if self.recording_type == RecordingType.CLAP:
                        print("\nGet ready to clap...", flush=True)
                        for countdown in range(3, 0, -1):
                            print(f"{countdown}...", end="\r", flush=True)
                            time.sleep(0.5)
                        print("CLAP NOW!", flush=True)
                    else:
                        print("\nRecording background noise...", end="\r", flush=True)
                        print("Ready...", flush=True)
                        time.sleep(0.5)
                    
                    print(f"Chunk {i+1} of {self.num_samples}.", flush=True)
                    
                    chunk, overflowed = self._record_chunk()
                    if overflowed:
                        print("Warning: Audio buffer overflow detected")
                    
                    self._save_chunk(chunk, i)
                    time.sleep(self.chunk_duration)
            
            print("\nRecording finished.")
            
        except Exception as e:
            raise AudioRecorderError(f"Recording failed: {e}")
        finally:
            if self.stream is not None:
                self.stream.close()


def main():
    """Main function to demonstrate the AudioRecorder usage."""
    try:
        # Example usage for recording claps
        print("Recording claps...")
        clap_recorder = AudioRecorder(recording_type=RecordingType.CLAP, num_samples=10)
        clap_recorder.record()
        
        print("Ready to Record Background noise after 10 seconds.")
        sleep(10)
        # Example usage for recording background noise
        print("\nRecording background noise...")
        background_recorder = AudioRecorder(
            recording_type=RecordingType.BACKGROUND_NOISE,
            num_samples=10  # Fewer samples for background noise
        )
        background_recorder.record()
        
    except AudioRecorderError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")


if __name__ == "__main__":
    main()