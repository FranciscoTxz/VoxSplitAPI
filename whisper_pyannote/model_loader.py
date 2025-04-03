# myapp/model_loader.py
import whisper
from VoxSplit.settings import URL_WHISPER_MODEL, HF_TOKEN
from pyannote.audio import Pipeline, Model

# Cargar el modelo de Whisper al iniciar la aplicación
whisper_model = whisper.load_model("turbo", download_root = f"{URL_WHISPER_MODEL}")
print("Modelo cargado Whisper")

#Cargar Pyannote al iniciar la aplicación
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  cache_dir=r"/Users/samaelxz/Documents/Materias/Semestre 10/Cosas-Tesis/djangoAPI/pyannote_model",
  use_auth_token=HF_TOKEN)
print("Modelo cargado Pyannote")

# Cargar modelo de Segmentacion
modelSegmentation = Model.from_pretrained(
  "pyannote/segmentation-3.0",
  cache_dir=r"/Users/samaelxz/Documents/Materias/Semestre 10/Cosas-Tesis/djangoAPI/pyannote_model_segmentation", 
  use_auth_token=HF_TOKEN)
print("Modelo cargado Segmentation")


def get_model_whisper():
    return whisper_model

def get_model_pyannote():
    return pipeline

def get_model_segmentation():
    return modelSegmentation

