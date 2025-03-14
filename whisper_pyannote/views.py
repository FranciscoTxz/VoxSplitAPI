from VoxSplit.settings import URL_WHISPER_MODEL
import whisper
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .models import AudioFile
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

            # Configurar opciones de transcripción
            options = {"language": language}
            if num_speakers:
                options["beam_size"] = int(num_speakers)  # Ajustar el beam search


            model = whisper.load_model("small", download_root = f"{URL_WHISPER_MODEL}")  # Cargar el modelo de Whisper
            result = model.transcribe(audio_instance.file.path, word_timestamps=True,  **options)  # Transcribir audio
            
            return Response({"transcription": result["text"]}, status=status.HTTP_200_OK)

        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)