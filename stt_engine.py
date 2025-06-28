from transformers import pipeline
import librosa
import os
from werkzeug.utils import secure_filename

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class AudioTranscriber:
    def __init__(self):
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", return_timestamps=True)

    def transcribe(self, audio_path):
        # Load audio with standard sample rate
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Get transcription
        result = self.transcriber({"array": y, "sampling_rate": sr})
        
        return result["text"]

# Initialize transcriber
transcriber = AudioTranscriber()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
