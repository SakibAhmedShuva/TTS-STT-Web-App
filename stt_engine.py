import os
import librosa
from transformers import pipeline as hf_pipeline

class AudioTranscriber:
    """A simple wrapper for the Whisper STT model."""
    def __init__(self, model_name="openai/whisper-tiny"):
        print(f"Loading STT model: {model_name}...")
        self.transcriber = hf_pipeline("automatic-speech-recognition", model=model_name)
        print("STT model loaded.")

    def transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        y, sr = librosa.load(audio_path, sr=16000)
        result = self.transcriber({"array": y, "sampling_rate": sr})
        return result["text"]

# Global STT instance
stt_model = None

def initialize_stt():
    """Initialize the STT model."""
    global stt_model
    if stt_model is None:
        stt_model = AudioTranscriber()
    return stt_model

def get_stt_model():
    """Get the initialized STT model."""
    return stt_model

# File validation
ALLOWED_STT_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_stt_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_STT_EXTENSIONS