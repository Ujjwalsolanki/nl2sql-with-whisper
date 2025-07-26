# speech_to_text.py

import logging
import io
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
import asyncio # Import asyncio for async functions

# Setup logging for this module
try:
    from logger_config import setup_logging
    logger = setup_logging(log_level="INFO")
except ImportError:
    # Fallback logger if logger_config.py is not yet available or configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("logger_config.py not found or not configured. Using basic logging.")

whisper_model = None

class ChunkedSpeechToTextManager:
    """
    Manages speech-to-text transcription using a locally downloaded Faster-Whisper model.
    The model is loaded upon instantiation of this class.
    """
    def __init__(self):
        """
        Initializes the SpeechToTextManager by loading the local Whisper model
        based on the hardcoded class-level configuration.
        Args:
            model_base_path (str): The base local path where Whisper models are stored.
                                   The model will be loaded from {model_base_path}/{MODEL_TYPE}.
                                   Defaults to "./models" as per your provided code's download_root.
        """
        self.model = self._get_whisper_model()

    def _get_whisper_model(self):
        """
        Loads the Faster-Whisper model based on the hardcoded configuration.
        Returns:
            WhisperModel: The loaded Whisper model instance.
        """
        MODEL_TYPE = "tiny"
        # MODEL_TYPE = "large-v2"
        RUN_TYPE = "cpu"  # "cpu" or "gpu"

        # For CPU usage (https://github.com/SYSTRAN/faster-whisper/issues/100#issuecomment-1492141352)
        NUM_WORKERS = 10
        CPU_THREADS = 4

        # For GPU usage
        GPU_DEVICE_INDICES = [0, 1, 2, 3]

        VAD_FILTER = True

        try:
            if RUN_TYPE.lower() == "gpu":
                whisper = WhisperModel(MODEL_TYPE,
                               device="cuda",
                               compute_type="float16",
                               device_index=GPU_DEVICE_INDICES,
                               download_root="./models")
            elif RUN_TYPE.lower() == "cpu":
                whisper = WhisperModel(MODEL_TYPE,
                               device="cpu",
                               compute_type="int8",
                               num_workers=NUM_WORKERS,
                               cpu_threads=CPU_THREADS,
                               download_root="./models")
            else:
                raise ValueError(f"Invalid model type: {RUN_TYPE}")

            print("Loaded model")

            return whisper

        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise

    async def transcribe_audio_async_chunked(self, audio_file_like: io.BytesIO, chunk_length_s: int = 1) -> str:
        """
        Asynchronously transcribes audio from a file-like object in specified 1-second chunks
        using Faster-Whisper. This processes a complete audio file by segmenting it internally.

        Args:
            audio_file_like (io.BytesIO): A file-like object containing the audio data
                                        (e.g., from st.audio_input, typically OGG/WebM).
            chunk_length_s (int): The length of each audio chunk in seconds for internal processing.

        Returns:
            str: The complete transcribed text. Returns an empty string if transcription fails.
        """
        if self.model is None:
            logger.error("Whisper model not loaded. Cannot transcribe audio.")
            return ""

        temp_input_audio_path = "temp_input_audio_full.ogg" # Use OGG/WEBM as initial temp file
        temp_wav_chunk_path = "temp_chunk.wav" # For individual WAV chunks

        full_transcription = []

        try:
            logger.info("Starting async chunked audio transcription process.")
            
            # Save the incoming BytesIO to a temporary OGG/WEBM file first
            audio_file_like.seek(0) # Ensure we're at the beginning of the stream
            with open(temp_input_audio_path, "wb") as f:
                f.write(audio_file_like.read())
            logger.info(f"Temporary full input audio saved to: {temp_input_audio_path}")

            # Read the full audio data from the temporary file
            audio_data, samplerate = sf.read(temp_input_audio_path)
            logger.info(f"Full audio loaded. Samplerate: {samplerate}, Duration: {len(audio_data) / samplerate:.2f}s")

            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.info("Converted stereo audio to mono.")

            # Resample to 16kHz if necessary
            if samplerate != 16000:
                logger.info(f"Resampling audio from {samplerate}Hz to 16000Hz.")
                audio_data = resample_poly(audio_data, 16000, samplerate)
                samplerate = 16000 # Update samplerate after resampling
            else:
                logger.info(f"Audio already at 16kHz sample rate.")

            # Calculate number of samples per chunk
            samples_per_chunk = int(chunk_length_s * samplerate)
            total_samples = len(audio_data)
            num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk

            logger.info(f"Processing audio in {chunk_length_s}-second chunks. Total chunks: {num_chunks}")

            for i in range(num_chunks):
                start_sample = i * samples_per_chunk
                end_sample = min((i + 1) * samples_per_chunk, total_samples)
                chunk = audio_data[start_sample:end_sample]

                if len(chunk) == 0:
                    continue # Skip empty chunks

                # Write the current chunk to a temporary WAV file
                sf.write(temp_wav_chunk_path, chunk, samplerate, format='WAV')
                logger.info(f"Processing chunk {i+1}/{num_chunks} (samples {start_sample}-{end_sample}).")

                # Transcribe the chunk using Faster-Whisper
                # Note: _whisper_model_instance.transcribe is a blocking call.
                # For true non-blocking, this would need to be run in an executor.
                # For Streamlit's context, st.spinner provides the visual async feel.
                segments, info = self.model.transcribe(temp_wav_chunk_path, beam_size=5)
                chunk_transcription = " ".join([segment.text for segment in segments])
                full_transcription.append(chunk_transcription)
                logger.info(f"Chunk {i+1} transcribed: '{chunk_transcription}'")

                # Clean up temporary chunk file after each use
                if os.path.exists(temp_wav_chunk_path):
                    os.remove(temp_wav_chunk_path)
                
                # Yield control to the event loop if this were truly async,
                # but for blocking calls, this is mostly for signature.
                await asyncio.sleep(0.01) # Small sleep to allow other tasks if any, though transcribe is blocking

            final_transcription = " ".join(full_transcription).strip()
            logger.info(f"Full audio transcribed successfully: '{final_transcription}'")
            return final_transcription

        except Exception as e:
            logger.error(f"Error during chunked audio transcription: {e}", exc_info=True)
            return ""
        finally:
            # Clean up temporary input audio file
            if os.path.exists(temp_input_audio_path):
                os.remove(temp_input_audio_path)
                logger.info(f"Temporary full input audio file removed: {temp_input_audio_path}")
            # Ensure temporary chunk file is also removed if loop was interrupted
            if os.path.exists(temp_wav_chunk_path):
                os.remove(temp_wav_chunk_path)
                logger.info(f"Temporary chunk audio file removed: {temp_wav_chunk_path}")
