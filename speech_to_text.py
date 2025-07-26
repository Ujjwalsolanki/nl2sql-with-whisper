# speech_to_text.py

import logging
import io
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# Setup logging for this module
try:
    from logger_config import setup_logging
    logger = setup_logging(log_level="INFO")
except ImportError:
    # Fallback logger if logger_config.py is not yet available or configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("logger_config.py not found or not configured. Using basic logging.")


class SpeechToTextManager:
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

    def transcribe_audio(self, audio_file_like: io.BytesIO) -> str:
        """
        Transcribes audio from a file-like object into text using Faster-Whisper.

        Args:
            audio_file_like (io.BytesIO): A file-like object containing the audio data.
                                         (e.g., from st.audio_input, typically OGG/WebM)

        Returns:
            str: The transcribed text. Returns an empty string if transcription fails.
        """
        if not self.model:
            logger.error("Whisper model not loaded. Cannot transcribe audio.")
            return ""

        temp_audio_path = "temp_audio_for_whisper.wav" # Define outside try for finally block

        try:
            logger.info("Starting audio transcription process.")
            
            # Read audio data from the BytesIO object.
            # Streamlit's st.audio_input typically outputs in WebM/OGG format.
            # soundfile can read from file-like objects directly.
            # Try reading without explicit format first, then with common formats if needed.
            try:
                # Attempt to read with auto-detection
                audio_data, samplerate = sf.read(audio_file_like)
                logger.info(f"Audio read successfully with auto-detection. Samplerate: {samplerate}")
            except Exception as read_error_auto:
                logger.warning(f"Auto-detection failed: {read_error_auto}. Trying 'OGG' format.")
                try:
                    # Try explicitly with OGG
                    audio_file_like.seek(0) # Reset stream position
                    audio_data, samplerate = sf.read(audio_file_like, format='OGG')
                    logger.info(f"Audio read successfully with 'OGG' format. Samplerate: {samplerate}")
                except Exception as read_error_ogg:
                    logger.warning(f"'OGG' format failed: {read_error_ogg}. Trying 'WEBM' format.")
                    try:
                        # Try explicitly with WEBM
                        audio_file_like.seek(0) # Reset stream position
                        audio_data, samplerate = sf.read(audio_file_like, format='WEBM')
                        logger.info(f"Audio read successfully with 'WEBM' format. Samplerate: {samplerate}")
                    except Exception as read_error_webm:
                        logger.error(f"Failed to read audio data from BytesIO with auto-detection, OGG, or WEBM formats: {read_error_webm}", exc_info=True)
                        raise ValueError("Could not read audio data. Ensure it's a supported format.")

            # Convert to mono if stereo (Faster-Whisper prefers mono)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.info("Converted stereo audio to mono.")

            # Resample to 16kHz if necessary (Faster-Whisper's preferred sample rate)
            if samplerate != 16000:
                logger.info(f"Resampling audio from {samplerate}Hz to 16000Hz.")
                audio_data = resample_poly(audio_data, 16000, samplerate)
                samplerate = 16000 # Update samplerate after resampling
            else:
                logger.info(f"Audio already at 16kHz sample rate.")

            # Write the processed audio to a temporary WAV file
            sf.write(temp_audio_path, audio_data, samplerate, format='WAV')
            logger.info(f"Audio converted to WAV and saved temporarily: {temp_audio_path}")

            # Transcribe using the loaded Faster-Whisper model
            logger.info(f"Calling Faster-Whisper transcribe with {temp_audio_path}...")
            segments, info = self.model.transcribe(temp_audio_path, beam_size=5)
            transcribed_text = " ".join([segment.text for segment in segments])
            logger.info(f"Audio transcribed successfully. Text: '{transcribed_text}'")

            return transcribed_text.strip()
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            return ""
        finally: # Ensure temp file is cleaned up even if errors occur
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"Temporary audio file removed: {temp_audio_path}")