import re
import logging
from VoxSplit.settings import URL_WHISPER_MODEL
from django.http import HttpRequest, JsonResponse
from pyannote.audio.pipelines import OverlappedSpeechDetection
from rest_framework.parsers import MultiPartParser, FormParser
from drfasyncview import AsyncRequest, AsyncAPIView
from asgiref.sync import sync_to_async

# Ubuntu 24
from whisper_pyannote.model_loader import get_model_segmentation, get_model_whisper, get_model_pyannote
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
            speakers = request.headers.get("speakers", 2)

            try:
                transcription_result = await self.transcribe_audio(audio_instance.file.path, language, speakers)
            except Exception as e:
                logger.error(f"Error en transcripción: {e}")
                return JsonResponse(
                            data={"error": f"Error en la transcripción del audio. {e}"},
                            status=500,
                        )

            try:
                diarization_result = await self.diarize_audio(audio_instance.file.path, speakers)
            except Exception as e:
                logger.error(f"Error en diarización: {e}")
                return JsonResponse(
                            data={"error": f"Error en la diarización del audio. {e}"},
                            status=500,
                        )
            
            try:
                osd_dict = await self.segmentation_audio(audio_instance.file.path)
            except Exception as e:
                logger.error(f"Error en segmentacion: {e}")
                return JsonResponse(
                            data={"error": f"Error en la segmentacion del audio. {e}"},
                            status=500,
                        )
            
            ### JUNTAR LOS RESULTADOS
            response = await self.mix_all(transcription_result, diarization_result, osd_dict)

            # 🔹 Guardar en la base de datos
            await self.save_audio_info(audio_instance.file.name, transcription_result["text"], diarization_result, osd_dict, response)

            response = {}
            return JsonResponse(
                            data=response,
                            status=201,
                           )

        return JsonResponse(
                            data={"error": "Error en una parte del proceso"},
                            status=400,
                           )              
    @sync_to_async
    def save_file(self, file_serializer):
        """Guarda el archivo en la base de datos de forma asíncrona"""
        return file_serializer.save()
    
    async def mix_all(self, transcription_result, diarization_result, osd_dict):
        """Mezcla los resultados de transcripción y diarización"""
        # CLEAN DATA
        for segment in transcription_result["segments"]:
            del segment["seek"]
            del segment["tokens"]
            del segment["temperature"]
            del segment["avg_logprob"]
            del segment["compression_ratio"]
            del segment["no_speech_prob"]
            for word in segment["words"]:
                del word["probability"]
                if 'start' in word:
                    word['start'] = float(word['start'])
                if 'end' in word:
                    word["end"] = float(word["end"])
                if 'probability' in word:
                    word["probability"] = float(word["probability"])

        # Speaker to text
        for segment in transcription_result["segments"]:
            for word in segment["words"]:
                word["speaker"] = "Unknown"
                for diarize in diarization_result:
                    if "start" in word and word["start"] >= float(diarize["inicio"]) and "end" in word and word["end"] <= float(diarize["fin"]):
                        word["speaker"] = diarize["hablante"]
                        break
                if word["speaker"] == "Unknown" and "start" in word:
                    promedio = (word["start"] + word["end"]) / 2
                    for diarize in diarization_result:
                        if promedio >= float(diarize["inicio"]):
                            word["speaker"] = diarize["hablante"]

        # Overlapped
        for segment in transcription_result["segments"]:
            for word in segment["words"]:
                for overlapp in osd_dict:
                    if "start" in word and word["start"] >= float(overlapp["start"]) and "end" in word and word["end"] <= float(overlapp["end"]):
                        word["overlapp"] = True
                        break 

        # Analyze the code
        for segment in transcription_result["segments"]:
            prob = {}
            probEnd = []
            cnt = 0
            cntOverlapp = 0
            for word in segment["words"]:
                if "speaker" in word:
                    cnt += 1
                    prob[word["speaker"]] = prob.get(word["speaker"], 0) + 1
                if "overlapp" in word and word["overlapp"] == True:
                    cntOverlapp += 1
            for key, value in prob.items():
                probEnd.append(f"{key}: {int((value / cnt) * 100)}%")
            segment["prob"] = probEnd
            segment["overlapp"] = f"{int((cntOverlapp / cnt) * 100)}%"

        return transcription_result

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
        result = pyannote_model(file_path, num_speakers=int(num_speakers))

        return [
            {
                "inicio": f"{turn.start:.4f}",
                "fin": f"{turn.end:.4f}",
                "hablante": speaker
            }
            for turn, _, speaker in result.itertracks(yield_label=True)
        ]

    async def segmentation_audio(self, file_path):
        """Ejecuta la segmentación del audio de forma asíncrona"""
        segmentation_model = get_model_segmentation()
        pipelineO = OverlappedSpeechDetection(segmentation=segmentation_model)
        HYPER_PARAMETERS = {
            "min_duration_on": 0.182,
            "min_duration_off": 0.501
        }

        pipelineO.instantiate(HYPER_PARAMETERS)
        osd = pipelineO(file_path)
        osd_dict = [
            {
                "start": f"{segment.start:.4f}",
                "end": f"{segment.end:.4f}",
                "label": label
            }
            for segment, label in osd.itertracks()
        ]
        return osd_dict

    async def save_audio_info(self, file_name, transcription, diarization, overlappR, resultX):
        """Guarda la transcripción y la diarización en la base de datos"""
        await AudioInfo.objects.acreate(  # 🔹 `acreate()` es la versión asincrónica de `create()`
            file_name=file_name,
            transcription=transcription,
            diarization=diarization,
            overlapp = overlappR,
            result = resultX
        )
