import re
from VoxSplit.settings import URL_WHISPER_MODEL
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

# Ubuntu 24

from whisper_pyannote.model_loader import get_model_whisper, get_model_pyannote
from .models import AudioInfo
from .serializers import AudioFileSerializer

class TranscribeAudioView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = AudioFileSerializer(data=request.data)

        if file_serializer.is_valid():
            audio_instance = file_serializer.save()  # Guarda el archivo

            #   Leer valores de los headers
            language = request.headers.get("language", "en")  # Inglés por defecto
            num_speakers = request.headers.get("number_speakers", None)  # Puede ser None


            # TRANSCRIPCIÓN DE AUDIO
            options = {"language": language}
            if num_speakers:
                options["beam_size"] = int(num_speakers)  # Ajustar el beam search
            model = get_model_whisper() # Cargar el modelo de Whisper
            resultWhisper = model.transcribe(audio_instance.file.path, word_timestamps=True,  **options)  # Transcribir audio

            #DIARIZACIÓN DE AUDIO
            pyannote_model = get_model_pyannote()  # Cargar el modelo de Pyannote
            if num_speakers:
                resultPyannote = pyannote_model(audio_instance.file.path, num_speakers = num_speakers)
            else:
                resultPyannote = pyannote_model(audio_instance.file.path)

            # Formatear la salida de Pyannote
            formatted_data = []
            for turn, _, speaker in resultPyannote.itertracks(yield_label=True):
                formatted_data.append({
                                "inicio": f"{turn.start:.4f}",
                                "fin": f"{turn.end:.4f}",
                                "hablante": speaker
                            })

            response = {
                "transcription": resultWhisper,
                "diarization": formatted_data
            }
            
            AudioInfo.objects.create(
                transcription=resultWhisper["text"],  # Guarda la transcripción
                diarization=formatted_data  # Guarda toda la estructura JSON
            )
            
            return Response(response, status=status.HTTP_200_OK)

        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)