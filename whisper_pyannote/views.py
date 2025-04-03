import re
import logging
from VoxSplit.settings import URL_WHISPER_MODEL
from django.http import HttpRequest, JsonResponse
from pyannote.audio.pipelines import OverlappedSpeechDetection
from rest_framework.parsers import MultiPartParser, FormParser
from drfasyncview import AsyncRequest, AsyncAPIView
from asgiref.sync import sync_to_async
from whisper_pyannote.model_loader import get_model_segmentation, get_model_whisper, get_model_pyannote
from .models import AudioInfo
from .serializers import AudioFileSerializer

logger = logging.getLogger(__name__)

class TranscribeAudioView(AsyncAPIView): 
    parser_classes = (MultiPartParser, FormParser)

    async def post(self, request: AsyncRequest, *args, **kwargs):
        file_serializer = AudioFileSerializer(data=request.data)

        if file_serializer.is_valid():
            audio_instance = await self.save_file(file_serializer) # Save the file in the database

            # Save headers
            language = request.headers.get("language", "en")  
            speakers = request.headers.get("speakers", 2)

            # Transcribe the audio
            try:
                transcription_result = await self.transcribe_audio(audio_instance.file.path, language, speakers)
            except Exception as e:
                logger.error(f"Transcribe Error: {e}")
                return JsonResponse(
                            data={"error": f"Transcribe Error: {e}"},
                            status=500,
                        )

            # Diarize the audio
            try:
                diarization_result = await self.diarize_audio(audio_instance.file.path, speakers)
            except Exception as e:
                logger.error(f"Diarization Error: {e}")
                return JsonResponse(
                            data={"error": f"Diarization Error: {e}"},
                            status=500,
                        )
            
            # See Overlapped Speech Detection
            try:
                osd_dict = await self.overley_audio(audio_instance.file.path)
            except Exception as e:
                logger.error(f"Overlay Error: {e}")
                return JsonResponse(
                            data={"error": f"Overlay Error {e}"},
                            status=500,
                        )
            
            # Put together the results
            response = await self.mix_all(transcription_result, diarization_result, osd_dict)

            # Save the results in the database
            await self.save_audio_info(audio_instance.file.name, transcription_result["text"], diarization_result, osd_dict, response)

            # Send Response
            return JsonResponse(
                            data=response,
                            status=201,
                            json_dumps_params={'ensure_ascii': False},
                           )
        # Error
        return JsonResponse(
                            data={"error": "Invalid file"},
                            status=400,
                           )              
    @sync_to_async
    def save_file(self, file_serializer):
        """Saves the file to the database asynchronously"""
        return file_serializer.save()
    
    async def mix_all(self, transcription_result, diarization_result, osd_dict):
        """Mix the results of transcription and diarization"""
        # Clean the transcription result
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
                    if "start" in word and word["start"] >= float(diarize["start"]) and "end" in word and word["end"] <= float(diarize["end"]):
                        word["speaker"] = diarize["speaker"]
                        break
                if word["speaker"] == "Unknown" and "start" in word:
                    promedio = (word["start"] + word["end"]) / 2
                    for diarize in diarization_result:
                        if promedio >= float(diarize["start"]):
                            word["speaker"] = diarize["speaker"]

        # Voice overlay detection
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
        """Runs audio transcription asynchronously"""
        options = {"language": language}
        if num_speakers:
            try:
                options["beam_size"] = int(num_speakers)
            except ValueError:
                raise ValueError("number_speakers debe ser un número válido.")

        model = get_model_whisper()
        res =  model.transcribe(file_path, word_timestamps=True, **options)
        return res

    async def diarize_audio(self, file_path, num_speakers):
        """Runs audio diarization asynchronously"""
        pyannote_model = get_model_pyannote() 
        result = pyannote_model(file_path, num_speakers=int(num_speakers))

        return [
            {
                "start": f"{turn.start:.4f}",
                "end": f"{turn.end:.4f}",
                "speaker": speaker
            }
            for turn, _, speaker in result.itertracks(yield_label=True)
        ]

    async def overley_audio(self, file_path):
        """Runs audio voices overlay detection asynchronously"""
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
        """Save the transcription and diarization to the database."""
        await AudioInfo.objects.acreate(  # 🔹
            file_name=file_name, 
            transcription=transcription,
            diarization=diarization,
            overlapp = overlappR,
            result = resultX
        )
