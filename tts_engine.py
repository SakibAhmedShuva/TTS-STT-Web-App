import os
import gc
import time
import logging
import traceback
import torch
import soundfile as sf
import numpy as np
from pydub import AudioSegment
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

# --- Configuration ---
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
CUDA_AVAILABLE = torch.cuda.is_available()

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
            os.remove(temp_wav_path)  # Clean up intermediate wav

        return {"status": "success", "file_path": final_path}

class HQTTS:
    """High Quality TTS using Kokoro."""
    def __init__(self):
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro library not available")
        
        self.kokoro_gpu = None
        self.kokoro_cpu = None
        self.kokoro_pipelines = {}
        self.kokoro_voices = {
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
        self._initialize_kokoro()

    def _initialize_kokoro(self):
        """Initialize Kokoro models and pipelines."""
        logging.info("Initializing HQ TTS (Kokoro) model...")
        try:
            if CUDA_AVAILABLE:
                logging.info("Loading Kokoro GPU model...")
                self.kokoro_gpu = KModel().to('cuda').eval()
            logging.info("Loading Kokoro CPU model...")
            self.kokoro_cpu = KModel().to('cpu').eval()

            logging.info("Initializing Kokoro pipelines...")
            self.kokoro_pipelines = {
                lang_code: KPipeline(lang_code=lang_code, model=False)
                for lang_code in ['a', 'b']
            }

            logging.info(f"Loading {len(self.kokoro_voices)} Kokoro voice packs...")
            for voice_id in self.kokoro_voices.values():
                self.kokoro_pipelines[voice_id[0]].load_voice(voice_id)
            logging.info("Kokoro initialization complete.")
        except Exception as e:
            logging.error(f"ERROR initializing Kokoro: {e}\n{traceback.format_exc()}")
            raise

    def get_available_voices(self):
        """Get available HQ voices."""
        return {"voices": self.kokoro_voices}

    def synthesize(self, text, voice_id, speed=1.0, use_gpu=True):
        """Synthesize speech using Kokoro."""
        use_gpu = use_gpu and CUDA_AVAILABLE
        kokoro_model = self.kokoro_gpu if use_gpu else self.kokoro_cpu
        
        if kokoro_model is None:
            raise RuntimeError(f"Required Kokoro model ({'GPU' if use_gpu else 'CPU'}) not loaded")

        lang_code = voice_id[0]
        pipeline = self.kokoro_pipelines[lang_code]
        pack = pipeline.load_voice(voice_id)
        ps = next(pipeline(text, voice_id, speed))[1]
        ref_s = pack[len(ps)-1].to(kokoro_model.device)

        with torch.no_grad():
            audio_tensor = kokoro_model(ps, ref_s, speed).squeeze().cpu()

        output_filename = f"hq_output_{time.time_ns()}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        sf.write(output_path, audio_tensor, 24000)

        return {"status": "success", "file_path": output_path}

# Global TTS instances
lq_tts_model = None
hq_tts_model = None

def initialize_lq_tts():
    """Initialize the LQ TTS model."""
    global lq_tts_model
    if lq_tts_model is None:
        lq_tts_model = LQTTS()
    return lq_tts_model

def initialize_hq_tts():
    """Initialize the HQ TTS model."""
    global hq_tts_model
    if hq_tts_model is None and KOKORO_AVAILABLE:
        hq_tts_model = HQTTS()
    return hq_tts_model

def get_lq_tts_model():
    """Get the initialized LQ TTS model."""
    return lq_tts_model

def get_hq_tts_model():
    """Get the initialized HQ TTS model."""
    return hq_tts_model

def is_kokoro_available():
    """Check if Kokoro is available."""
    return KOKORO_AVAILABLE

def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()