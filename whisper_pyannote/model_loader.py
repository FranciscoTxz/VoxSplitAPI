# myapp/model_loader.py
import whisper
from VoxSplit.settings import URL_WHISPER_MODEL, HF_TOKEN
from pyannote.audio import Pipeline, Model

# Load the Whisper model on application startup
whisper_model = whisper.load_model("turbo", download_root = f"{URL_WHISPER_MODEL}")
print("---Whisper---")

# Load Pyannote on application startup
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  cache_dir=r"/Users/samaelxz/Documents/Materias/Semestre 10/Cosas-Tesis/djangoAPI/pyannote_model",
  use_auth_token=HF_TOKEN)
print("---Pyannote---")

# Load voice overlay model on application start
overley_model = Model.from_pretrained(
  "pyannote/segmentation-3.0",
  cache_dir=r"/Users/samaelxz/Documents/Materias/Semestre 10/Cosas-Tesis/djangoAPI/pyannote_model_segmentation", 
  use_auth_token=HF_TOKEN)
print("---Overlay---")


def get_model_whisper():
    return whisper_model

def get_model_pyannote():
    return pipeline

def get_model_segmentation():
    return overley_model

