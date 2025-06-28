import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from datetime import datetime
import threading
import queue
import os
from pydub import AudioSegment
from TTS.api import TTS
import time

class PublicDynamicTTS:
    def __init__(self):
        """Initialize the TTS service with public models"""
        print("Loading TTS models... This may take a few moments.")
        # Direct initialization of TTS models without any modifications
        self.female_1 = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)
        self.female_2 = TTS(model_name="tts_models/en/ljspeech/fast_pitch", gpu=False)
        
        # Keep VITS for male voices
        self.male_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        
        # Basic initialization
        self.task_queue = queue.Queue()
        self.task_statuses = {}
        self.output_dir = "tts_outputs"
        self.supported_formats = ['wav', 'flac', 'ogg', 'mp3']
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.start_task_processor()

    def convert_text_to_speech(self, text, voice="female_1", language="en-US", speed=1.0):
        """Convert text to speech"""
        try:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.task_statuses[task_id] = "processing"
            
            output_path = os.path.join(self.output_dir, f"{task_id}.wav")
            
            if voice == "female_1":
                self.female_1.tts_to_file(text=text, file_path=output_path)
                self.task_statuses[task_id] = "completed"
            elif voice == "female_2":
                self.female_2.tts_to_file(text=text, file_path=output_path)
                self.task_statuses[task_id] = "completed"
            else:
                # Male voice processing
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    output = self.male_model(**inputs).waveform
                    if voice == "male_2":
                        output = output * 0.8  # Simple pitch adjustment for male_2
                    
                sf.write(output_path, output.numpy().T, 16000)
                self.task_statuses[task_id] = "completed"
            
            return task_id
            
        except Exception as e:
            print(f"Conversion error: {str(e)}")
            self.task_statuses[task_id] = "failed"
            return None

    def save_speech_file(self, text, voice="female_1", language="en-US", format="wav", destination=None):
        """Save speech to file"""
        try:
            format = format.lower()
            if format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}. Use: {self.supported_formats}")
            
            if destination is None:
                destination = os.path.join(
                    self.output_dir, 
                    f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                )
            
            # Direct generation for female voices
            if voice == "female_1":
                self.female_1.tts_to_file(text=text, file_path=destination)
            elif voice == "female_2":
                self.female_2.tts_to_file(text=text, file_path=destination)
            else:
                # Male voice processing
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    output = self.male_model(**inputs).waveform
                    if voice == "male_2":
                        output = output * 0.8
                
                if format == 'mp3':
                    audio = AudioSegment(
                        output.numpy().tobytes(),
                        frame_rate=16000,
                        sample_width=2,
                        channels=1
                    )
                    audio.export(destination, format='mp3', bitrate='128k')
                else:
                    sf.write(destination, output.numpy().T, 16000, format=format)
            
            return {"status": "success", "file_path": destination}
            
        except Exception as e:
            print(f"Save error: {str(e)}")
            return None

    def stream_speech(self, text, voice="female_1", language="en-US"):
        """Stream speech output"""
        try:
            temp_path = os.path.join(self.output_dir, f"stream_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            
            # Generate audio
            if voice == "female_1":
                self.female_1.tts_to_file(text=text, file_path=temp_path)
            elif voice == "female_2":
                self.female_2.tts_to_file(text=text, file_path=temp_path)
            else:
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    output = self.male_model(**inputs).waveform
                    if voice == "male_2":
                        output = output * 0.8
                sf.write(temp_path, output.numpy().T, 16000)
            
            # Read and stream
            audio_data, _ = sf.read(temp_path)
            chunk_size = 16000  # 1 second chunks
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield None

    def start_task_processor(self):
        """Simple task status monitoring"""
        def process_tasks():
            while True:
                try:
                    time.sleep(1)  # Check every second
                except Exception as e:
                    print(f"Task processing error: {str(e)}")

        threading.Thread(target=process_tasks, daemon=True).start()

    def check_status(self, task_id):
        """Check task status"""
        return {
            "status": self.task_statuses.get(task_id, "not_found"),
            "audio_path": os.path.join(self.output_dir, f"{task_id}.wav") 
            if self.task_statuses.get(task_id) == "completed" 
            else None
        }

    def get_available_voices(self):
        """List available voices"""
        return {
            "voices": [
                {
                    "id": "female_1",
                    "language": "en-US",
                    "gender": "female",
                    "description": "LJSpeech Tacotron2-DDC voice"
                },
                {
                    "id": "female_2",
                    "language": "en-US",
                    "gender": "female",
                    "description": "LJSpeech FastPitch voice"
                },
                {
                    "id": "male_1",
                    "language": "en-US",
                    "gender": "male",
                    "description": "Default male voice"
                },
                {
                    "id": "male_2",
                    "language": "en-US",
                    "gender": "male",
                    "description": "Lower pitched male voice"
                }
            ]
        }

# Initialize TTS service
tts_service = PublicDynamicTTS()