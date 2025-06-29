import os
import logging
from flask import Flask, request, jsonify, send_from_directory, render_template, after_this_request
from werkzeug.utils import secure_filename
from flask_cors import CORS

from stt_engine import initialize_stt, get_stt_model, allowed_stt_file
from tts_engine import (
    initialize_lq_tts, initialize_hq_tts, 
    get_lq_tts_model, get_hq_tts_model,
    is_kokoro_available, cleanup_gpu
)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
CORS(app)
logging.basicConfig(level=logging.INFO)

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
        stt_model = get_stt_model()
        if not stt_model:
            return jsonify({"error": "STT service not available"}), 503
        
        transcript_text = stt_model.transcribe(audio_path)
        return jsonify({"transcription": transcript_text})
    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# --- LQ TTS Endpoints ---
@app.route('/tts/voices_lq', methods=['GET'])
def get_lq_voices():
    lq_tts = get_lq_tts_model()
    if not lq_tts:
        return jsonify({"error": "LQ TTS service not available"}), 503
    return jsonify(lq_tts.get_available_voices())

@app.route('/tts/synthesize_lq', methods=['POST'])
def synthesize_speech_lq():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text not provided"}), 400

    text = data.get('text')
    output_format = data.get('format', 'mp3').lower()

    lq_tts = get_lq_tts_model()
    if not lq_tts:
        return jsonify({"error": "LQ TTS service not available"}), 503

    try:
        result = lq_tts.save_speech_file(text=text, format=output_format)
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
    if not is_kokoro_available():
        return jsonify({"error": "HQ TTS (Kokoro) is not available"}), 503
    
    hq_tts = get_hq_tts_model()
    if not hq_tts:
        return jsonify({"error": "HQ TTS service not available"}), 503
    
    return jsonify(hq_tts.get_available_voices())

@app.route('/tts/synthesize_hq', methods=['POST'])
def synthesize_speech_hq():
    if not is_kokoro_available():
        return jsonify({"error": "HQ TTS (Kokoro) is not available"}), 503

    data = request.json
    text = data.get('text')
    voice_id = data.get('voice')  # e.g., 'af_heart'
    speed = float(data.get('speed', 1.0))
    use_gpu = data.get('use_gpu', True)

    if not text or not voice_id:
        return jsonify({"error": "Missing text or voice ID"}), 400

    hq_tts = get_hq_tts_model()
    if not hq_tts:
        return jsonify({"error": "HQ TTS service not available"}), 503

    try:
        result = hq_tts.synthesize(text, voice_id, speed, use_gpu)
        if result.get("status") == "success":
            file_path = result.get("file_path")
            directory, filename = os.path.split(file_path)

            @after_this_request
            def cleanup(response):
                try:
                    os.remove(file_path)
                    cleanup_gpu()
                except Exception as e:
                    app.logger.error(f"Error cleaning up file {file_path}: {e}")
                return response

            return send_from_directory(directory, filename, as_attachment=True)
        else:
            return jsonify({"error": "Failed to generate HQ speech"}), 500

    except Exception as e:
        return jsonify({"error": f"HQ TTS generation failed: {str(e)}"}), 500

# --- Main Runner ---
if __name__ == '__main__':
    print("--- Initializing All Models ---")
    initialize_stt()
    initialize_lq_tts()
    if is_kokoro_available():
        initialize_hq_tts()
    print("--- Model Initialization Complete ---")
    
    # Note: Use a proper WSGI server like Gunicorn or Waitress for production.
    app.run(host='0.0.0.0', port=5000, debug=True)