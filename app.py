import os
import gc
import io
import json
import re
import base64
import time
import traceback
import tempfile
import logging
from collections import OrderedDict

# --- Flask and Web ---
from flask import Flask, request, jsonify, send_from_directory, render_template, after_this_request
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Machine Learning and Audio Libraries ---
import torch
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

# --- LQ STT/TTS Imports ---
from transformers import pipeline as hf_pipeline
from TTS.api import TTS

# --- HQ (Kokoro) TTS Imports ---
try:
    from kokoro import KModel, KPipeline
    import kokoro
    KOKORO_AVAILABLE = True
    kokoro_version = getattr(kokoro, '__version__', 'N/A')
    print(f'DEBUG: Kokoro version {kokoro_version} found.')
except ImportError:
    KOKORO_AVAILABLE = False
    print("WARNING: Kokoro library not found. HQ TTS features will be disabled.")
    KModel, KPipeline = None, None

# --- Global Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- Global Model Storage ---
models = {
    # STT
    "stt_transcriber": None,
    # LQ TTS
    "lq_tts": None,
    # HQ TTS
    "kokoro_gpu": None,
    "kokoro_cpu": None,
    "kokoro_pipelines": {},
    "kokoro_voices": {},
}

# --- STT (LQ) Engine ---
ALLOWED_STT_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_stt_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_STT_EXTENSIONS

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

def initialize_stt_model():
    """Initializes the STT model."""
    if models["stt_transcriber"] is None:
        models["stt_transcriber"] = AudioTranscriber()

# --- TTS (LQ) Engine ---
class LQTTS:
    """A simplified TTS engine for the Tacotron2 model."""
    def __init__(self):
        print("Loading LQ TTS model (tacotron2-DDC)... This may take a moment.")
        self.model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
        self.output_dir = OUTPUT_FOLDER
        self.supported_formats = ['wav', 'flac', 'ogg', 'mp3']
        self.voice_info = {
            "id": "female_tacotron2",
            "language": "en-US",
            "gender": "female",
            "description": "Standard Female Voice (Tacotron2)"
        }
        print("LQ TTS model loaded.")

    def get_available_voices(self):
        return {"voices": [self.voice_info]}

    def save_speech_file(self, text, format="mp3"):
        format = format.lower()
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")

        temp_wav_path = os.path.join(self.output_dir, f"temp_{time.time_ns()}.wav")
        final_path = temp_wav_path.replace(".wav", f".{format}")

        self.model.tts_to_file(text=text, file_path=temp_wav_path)

        if format != 'wav':
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(final_path, format=format)
            os.remove(temp_wav_path) # Clean up intermediate wav

        return {"status": "success", "file_path": final_path}


def initialize_lq_tts_model():
    """Initializes the LQ TTS model."""
    if models["lq_tts"] is None:
        models["lq_tts"] = LQTTS()

# --- TTS (HQ - Kokoro) Engine ---
def initialize_hq_tts_model():
    """Load Kokoro models on startup."""
    if not KOKORO_AVAILABLE:
        logging.warning("Skipping HQ TTS initialization because Kokoro library is not available.")
        return

    logging.info("Initializing HQ TTS (Kokoro) model...")
    try:
        if CUDA_AVAILABLE:
            logging.info("Loading Kokoro GPU model...")
            models["kokoro_gpu"] = KModel().to('cuda').eval()
        logging.info("Loading Kokoro CPU model...")
        models["kokoro_cpu"] = KModel().to('cpu').eval()

        logging.info("Initializing Kokoro pipelines...")
        models["kokoro_pipelines"] = {
            lang_code: KPipeline(lang_code=lang_code, model=False)
            for lang_code in ['a', 'b']
        }
        kokoro_voices = {
                '🇺🇸 🚺 Heart ❤️': 'af_heart', '🇺🇸 🚺 Bella 🔥': 'af_bella', '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
                '🇺🇸 🚺 Aoede': 'af_aoede', '🇺🇸 🚺 Kore': 'af_kore', '🇺🇸 🚺 Sarah': 'af_sarah',
                '🇺🇸 🚺 Nova': 'af_nova', '🇺🇸 🚺 Sky': 'af_sky', '🇺🇸 🚺 Alloy': 'af_alloy',
                '🇺🇸 🚺 Jessica': 'af_jessica', '🇺🇸 🚺 River': 'af_river', '🇺🇸 🚹 Michael': 'am_michael',
                '🇺🇸 🚹 Fenrir': 'am_fenrir', '🇺🇸 🚹 Puck': 'am_puck', '🇺🇸 🚹 Echo': 'am_echo',
                '🇺🇸 🚹 Eric': 'am_eric', '🇺🇸 🚹 Liam': 'am_liam', '🇺🇸 🚹 Onyx': 'am_onyx',
                '🇺🇸 🚹 Santa': 'am_santa', '🇺🇸 🚹 Adam': 'am_adam', '🇬🇧 🚺 Emma': 'bf_emma',
                '🇬🇧 🚺 Isabella': 'bf_isabella', '🇬🇧 🚺 Alice': 'bf_alice', '🇬🇧 🚺 Lily': 'bf_lily',
                '🇬🇧 🚹 George': 'bm_george', '🇬🇧 🚹 Fable': 'bm_fable', '🇬🇧 🚹 Lewis': 'bm_lewis',
                '🇬🇧 🚹 Daniel': 'bm_daniel',
        }
        models["kokoro_voices"] = kokoro_voices

        logging.info(f"Loading {len(models['kokoro_voices'])} Kokoro voice packs...")
        for voice_id in models["kokoro_voices"].values():
            models["kokoro_pipelines"][voice_id[0]].load_voice(voice_id)
        logging.info("Kokoro initialization complete.")
    except Exception as e:
        logging.error(f"ERROR initializing Kokoro: {e}\n{traceback.format_exc()}")


# --- Web Interface & General Endpoints ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# --- Speech-to-Text (STT) Endpoint ---
@app.route('/transcribe_lq', methods=['POST'])
def transcribe_audio_lq():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_stt_file(file.filename):
        return jsonify({"error": "No selected file or file type not allowed"}), 400

    filename = secure_filename(file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(audio_path)

    try:
        transcript_text = models["stt_transcriber"].transcribe(audio_path)
        return jsonify({"transcription": transcript_text})
    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# --- Text-to-Speech (TTS) Endpoints ---

# --- LQ TTS Endpoints ---
@app.route('/tts/voices_lq', methods=['GET'])
def get_lq_voices():
    if not models["lq_tts"]:
        return jsonify({"error": "LQ TTS service not available"}), 503
    return jsonify(models["lq_tts"].get_available_voices())

@app.route('/tts/synthesize_lq', methods=['POST'])
def synthesize_speech_lq():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text not provided"}), 400

    text = data.get('text')
    output_format = data.get('format', 'mp3').lower()

    if not models["lq_tts"]:
        return jsonify({"error": "LQ TTS service not available"}), 503

    try:
        result = models["lq_tts"].save_speech_file(text=text, format=output_format)
        if result.get("status") == "success":
            file_path = result.get("file_path")
            directory, filename = os.path.split(file_path)

            @after_this_request
            def cleanup(response):
                try:
                    os.remove(file_path)
                except Exception as e:
                    app.logger.error(f"Error cleaning up file {file_path}: {e}")
                return response

            return send_from_directory(directory, filename, as_attachment=True)
        else:
            return jsonify({"error": "Failed to generate LQ speech"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- HQ TTS (Kokoro) Endpoints ---
@app.route('/tts/voices_hq', methods=['GET'])
def get_hq_voices():
    if not KOKORO_AVAILABLE:
        return jsonify({"error": "HQ TTS (Kokoro) is not available"}), 503
    return jsonify({"voices": models.get("kokoro_voices", {})})

@app.route('/tts/synthesize_hq', methods=['POST'])
def synthesize_speech_hq():
    if not KOKORO_AVAILABLE:
        return jsonify({"error": "HQ TTS (Kokoro) is not available"}), 503

    data = request.json
    text = data.get('text')
    voice_id = data.get('voice') # e.g., 'af_heart'
    speed = float(data.get('speed', 1.0))
    use_gpu = data.get('use_gpu', True) and CUDA_AVAILABLE

    if not text or not voice_id:
        return jsonify({"error": "Missing text or voice ID"}), 400

    kokoro_model = models["kokoro_gpu"] if use_gpu else models["kokoro_cpu"]
    if kokoro_model is None:
        return jsonify({"error": f"Required Kokoro model ({'GPU' if use_gpu else 'CPU'}) not loaded"}), 503

    try:
        lang_code = voice_id[0]
        pipeline = models["kokoro_pipelines"][lang_code]
        pack = pipeline.load_voice(voice_id)
        ps = next(pipeline(text, voice_id, speed))[1]
        ref_s = pack[len(ps)-1].to(kokoro_model.device)

        with torch.no_grad():
            audio_tensor = kokoro_model(ps, ref_s, speed).squeeze().cpu()

        output_filename = f"hq_output_{time.time_ns()}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        sf.write(output_path, audio_tensor, 24000)

        @after_this_request
        def cleanup(response):
            try:
                os.remove(output_path)
                gc.collect()
                if CUDA_AVAILABLE: torch.cuda.empty_cache()
            except Exception as e:
                app.logger.error(f"Error cleaning up file {output_path}: {e}")
            return response

        return send_from_directory(OUTPUT_FOLDER, output_filename, as_attachment=True)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"HQ TTS generation failed: {str(e)}"}), 500

# --- Main Runner ---
if __name__ == '__main__':
    print("--- Initializing All Models ---")
    initialize_stt_model()
    initialize_lq_tts_model()
    initialize_hq_tts_model()
    print("--- Model Initialization Complete ---")
    # Note: Use a proper WSGI server like Gunicorn or Waitress for production.
    app.run(host='0.0.0.0', port=5000, debug=True)