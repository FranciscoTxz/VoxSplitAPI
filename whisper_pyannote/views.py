import re
import logging
from VoxSplit.settings import URL_WHISPER_MODEL
from django.http import HttpRequest, JsonResponse
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from drfasyncview import AsyncRequest, AsyncAPIView
from asgiref.sync import sync_to_async

# Ubuntu 24
from whisper_pyannote.model_loader import get_model_whisper, get_model_pyannote
from .models import AudioInfo
from .serializers import AudioFileSerializer

logger = logging.getLogger(__name__)

class TranscribeAudioView(AsyncAPIView):  # 🔹 Ahora es una vista asíncrona
    parser_classes = (MultiPartParser, FormParser)

    async def post(self, request: AsyncRequest, *args, **kwargs):  # 🔹 Método asincrónico
        file_serializer = AudioFileSerializer(data=request.data)

        if file_serializer.is_valid():
            audio_instance = await self.save_file(file_serializer)  # 🔹 Guardar archivo de forma asíncrona

            language = request.headers.get("language", "en")  
            num_speakers = request.headers.get("number_speakers", None)

            try:
                transcription_result = await self.transcribe_audio(audio_instance.file.path, language, num_speakers)
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                return JsonResponse(
                            data={"error": f"Error en la transcripción del audio. {e}"},
                            status=500,
                        )

            try:
                diarization_result = await self.diarize_audio(audio_instance.file.path, num_speakers)
            except Exception as e:
                logger.error(f"Error en diarización: {e}")
                return JsonResponse(
                            data={"error": f"Error en la diarización del audio. {e}"},
                            status=500,
                        )

            response = {
                "transcription": transcription_result,
                "diarization": diarization_result
            }

            # 🔹 Guardar en la base de datos
            await self.save_audio_info(audio_instance.file.name, transcription_result["text"], diarization_result)

            return JsonResponse(
                            data=response,
                            status=200,
                           )

        return JsonResponse(
                            data={"error": "Error en una parte del proceso"},
                            status=400,
                           )              
    @sync_to_async
    def save_file(self, file_serializer):
        """Guarda el archivo en la base de datos de forma asíncrona"""
        return file_serializer.save()

    async def transcribe_audio(self, file_path, language, num_speakers):
        """Ejecuta la transcripción del audio de forma asíncrona"""
        options = {"language": language}
        if num_speakers:
            try:
                options["beam_size"] = int(num_speakers)
            except ValueError:
                raise ValueError("number_speakers debe ser un número válido.")

        model = get_model_whisper()  # 🔹 Se llama de forma asíncrona
        res =  model.transcribe(file_path, word_timestamps=True, **options)
        return res

    async def diarize_audio(self, file_path, num_speakers):
        """Ejecuta la diarización del audio de forma asíncrona"""
        pyannote_model = get_model_pyannote()  # 🔹 Se llama de forma asíncrona
        result = pyannote_model(file_path, num_speakers=int(num_speakers)) if num_speakers else pyannote_model(file_path)

        return [
            {
                "inicio": f"{turn.start:.4f}",
                "fin": f"{turn.end:.4f}",
                "hablante": speaker
            }
            for turn, _, speaker in result.itertracks(yield_label=True)
        ]

    async def save_audio_info(self, file_name, transcription, diarization):
        """Guarda la transcripción y la diarización en la base de datos"""
        await AudioInfo.objects.acreate(  # 🔹 `acreate()` es la versión asincrónica de `create()`
            file_name=file_name,
            transcription=transcription,
            diarization=diarization
        )
