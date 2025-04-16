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
import torchaudio
import torch

logger = logging.getLogger(__name__)

class TranscribeAudioView(AsyncAPIView): 
    parser_classes = (MultiPartParser, FormParser)

    async def post(self, request: AsyncRequest, *args, **kwargs):
        file_serializer = AudioFileSerializer(data=request.data)

        if file_serializer.is_valid():

            audio_instance = await self.save_file(file_serializer) # Save the file in the database

            # Transform the audio
            try:
                await self.transform_audio(audio_instance.file.path)
            except:
                logger.error(f"Transform Error: {e}")
                return JsonResponse(
                            data={"error": f"Transform Error: {e}"},
                            status=500,
                        )

            # Save headers
            language = request.headers.get("Language", "en")  
            speakers = request.headers.get("Speakers", 2)

            # print(f"\n\n {language} {speakers} \n\n")

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

            # DEFAULT RESPONSE
            # response = {"text": " Hola Rosita, ¿cómo estás? Bien. Más fuerte, más fuerte, más fuerte. Bien. ¿Qué estás haciendo? Comiendo. ¿Y qué ha sido de tu día? Pues fui a un centro comercial y ya volví. Ah, bueno. Este, a ver, ahora habla al mismo tiempo que estoy hablando yo. Así como, hola, ¿cómo estás? Hola, ¿cómo estás? Hola, ¿cómo estás? Hola, ¿cómo estás? Ya, gracias.", "segments": [{"id": 0, "start": 0.5200000000000014, "end": 1.88, "text": " Hola Rosita, ¿cómo estás?", "words": [{"word": " Hola", "start": 0.5200000000000014, "end": 1.0, "speaker": "Unknown"}, {"word": " Rosita,", "start": 1.0, "end": 1.38, "speaker": "SPEAKER_01"}, {"word": " ¿cómo", "start": 1.54, "end": 1.66, "speaker": "SPEAKER_01"}, {"word": " estás?", "start": 1.66, "end": 1.88, "speaker": "SPEAKER_01"}], "prob": ["Unknown: 25%", "SPEAKER_01: 75%"], "overlapp": "0%"}, {"id": 1, "start": 2.36, "end": 2.8, "text": " Bien.", "words": [{"word": " Bien.", "start": 2.36, "end": 2.8, "speaker": "SPEAKER_00"}], "prob": ["SPEAKER_00: 100%"], "overlapp": "0%"}, {"id": 2, "start": 3.4000000000000012, "end": 5.0, "text": " Más fuerte, más fuerte, más fuerte.", "words": [{"word": " Más", "start": 3.4000000000000012, "end": 3.88, "speaker": "SPEAKER_01"}, {"word": " fuerte,", "start": 3.88, "end": 4.1, "speaker": "SPEAKER_01"}, {"word": " más", "start": 4.16, "end": 4.26, "speaker": "SPEAKER_01"}, {"word": " fuerte,", "start": 4.26, "end": 4.54, "speaker": "SPEAKER_01"}, {"word": " más", "start": 4.56, "end": 4.7, "speaker": "SPEAKER_01"}, {"word": " fuerte.", "start": 4.7, "end": 5.0, "speaker": "SPEAKER_01"}], "prob": ["SPEAKER_01: 100%"], "overlapp": "0%"}, {"id": 3, "start": 5.2, "end": 5.5, "text": " Bien.", "words": [{"word": " Bien.", "start": 5.2, "end": 5.5, "speaker": "SPEAKER_00"}], "prob": ["SPEAKER_00: 100%"], "overlapp": "0%"}, {"id": 4, "start": 6.12, "end": 7.14, "text": " ¿Qué estás haciendo?", "words": [{"word": " ¿Qué", "start": 6.12, "end": 6.54, "speaker": "SPEAKER_00"}, {"word": " estás", "start": 6.54, "end": 6.76, "speaker": "SPEAKER_01"}, {"word": " haciendo?", "start": 6.76, "end": 7.14, "speaker": "SPEAKER_01"}], "prob": ["SPEAKER_00: 33%", "SPEAKER_01: 66%"], "overlapp": "0%"}, {"id": 5, "start": 7.960000000000001, "end": 8.44, "text": " Comiendo.", "words": [{"word": " Comiendo.", "start": 7.960000000000001, "end": 8.44, "speaker": "SPEAKER_00"}], "prob": ["SPEAKER_00: 100%"], "overlapp": "0%"}, {"id": 6, "start": 9.08, "end": 10.62, "text": " ¿Y qué ha sido de tu día?", "words": [{"word": " ¿Y", "start": 9.08, "end": 9.4, "speaker": "SPEAKER_01"}, {"word": " qué", "start": 9.4, "end": 9.84, "speaker": "SPEAKER_01"}, {"word": " ha", "start": 9.84, "end": 10.04, "speaker": "SPEAKER_01"}, {"word": " sido", "start": 10.04, "end": 10.22, "speaker": "SPEAKER_01"}, {"word": " de", "start": 10.22, "end": 10.34, "speaker": "SPEAKER_01"}, {"word": " tu", "start": 10.34, "end": 10.42, "speaker": "SPEAKER_01"}, {"word": " día?", "start": 10.42, "end": 10.62, "speaker": "SPEAKER_01"}], "prob": ["SPEAKER_01: 100%"], "overlapp": "0%"}, {"id": 7, "start": 11.360000000000001, "end": 14.0, "text": " Pues fui a un centro comercial y ya volví.", "words": [{"word": " Pues", "start": 11.360000000000001, "end": 11.84, "speaker": "SPEAKER_00"}, {"word": " fui", "start": 11.84, "end": 12.06, "speaker": "SPEAKER_00"}, {"word": " a", "start": 12.06, "end": 12.18, "speaker": "SPEAKER_00"}, {"word": " un", "start": 12.18, "end": 12.28, "speaker": "SPEAKER_00"}, {"word": " centro", "start": 12.28, "end": 12.58, "speaker": "SPEAKER_00"}, {"word": " comercial", "start": 12.58, "end": 13.08, "speaker": "SPEAKER_00"}, {"word": " y", "start": 13.08, "end": 13.4, "speaker": "SPEAKER_00"}, {"word": " ya", "start": 13.4, "end": 13.6, "speaker": "SPEAKER_00"}, {"word": " volví.", "start": 13.6, "end": 14.0, "speaker": "SPEAKER_00"}], "prob": ["SPEAKER_00: 100%"], "overlapp": "0%"}, {"id": 8, "start": 14.68, "end": 18.18, "text": " Ah, bueno. Este, a ver, ahora habla al mismo tiempo que estoy hablando yo.", "words": [{"word": " Ah,", "start": 14.68, "end": 14.92, "speaker": "SPEAKER_01"}, {"word": " bueno.", "start": 15.08, "end": 15.3, "speaker": "SPEAKER_01"}, {"word": " Este,", "start": 15.58, "end": 15.8, "speaker": "SPEAKER_01"}, {"word": " a", "start": 16.02, "end": 16.06, "speaker": "SPEAKER_01"}, {"word": " ver,", "start": 16.06, "end": 16.24, "speaker": "SPEAKER_01"}, {"word": " ahora", "start": 16.3, "end": 16.44, "speaker": "SPEAKER_01"}, {"word": " habla", "start": 16.44, "end": 16.74, "speaker": "SPEAKER_01"}, {"word": " al", "start": 16.74, "end": 16.86, "speaker": "SPEAKER_01"}, {"word": " mismo", "start": 16.86, "end": 17.04, "speaker": "SPEAKER_01"}, {"word": " tiempo", "start": 17.04, "end": 17.36, "speaker": "SPEAKER_01"}, {"word": " que", "start": 17.36, "end": 17.54, "speaker": "SPEAKER_01"}, {"word": " estoy", "start": 17.54, "end": 17.68, "speaker": "SPEAKER_01"}, {"word": " hablando", "start": 17.68, "end": 17.94, "speaker": "SPEAKER_01"}, {"word": " yo.", "start": 17.94, "end": 18.18, "speaker": "SPEAKER_01"}], "prob": ["SPEAKER_01: 100%"], "overlapp": "0%"}, {"id": 9, "start": 18.34, "end": 22.76, "text": " Así como, hola, ¿cómo estás? Hola, ¿cómo estás? Hola, ¿cómo estás? Hola, ¿cómo estás?", "words": [{"word": " Así", "start": 18.34, "end": 18.54, "speaker": "SPEAKER_01"}, {"word": " como,", "start": 18.54, "end": 18.78, "speaker": "SPEAKER_01"}, {"word": " hola,", "start": 18.9, "end": 19.16, "speaker": "SPEAKER_01"}, {"word": " ¿cómo", "start": 19.22, "end": 19.36, "speaker": "SPEAKER_01"}, {"word": " estás?", "start": 19.36, "end": 19.7, "speaker": "SPEAKER_01"}, {"word": " Hola,", "start": 20.02, "end": 20.3, "speaker": "SPEAKER_01"}, {"word": " ¿cómo", "start": 20.44, "end": 20.56, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " estás?", "start": 20.56, "end": 21.06, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " Hola,", "start": 21.38, "end": 21.7, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " ¿cómo", "start": 21.7, "end": 21.7, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " estás?", "start": 21.7, "end": 21.7, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " Hola,", "start": 21.7, "end": 21.7, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " ¿cómo", "start": 22.16, "end": 22.24, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " estás?", "start": 22.24, "end": 22.76, "speaker": "SPEAKER_01", "overlapp": True}], "prob": ["SPEAKER_01: 100%"], "overlapp": "57%"}, {"id": 10, "start": 23.22, "end": 23.96, "text": " Ya, gracias.", "words": [{"word": " Ya,", "start": 23.22, "end": 23.44, "speaker": "SPEAKER_01", "overlapp": True}, {"word": " gracias.", "start": 23.62, "end": 23.96, "speaker": "SPEAKER_01"}], "prob": ["SPEAKER_01: 100%"], "overlapp": "50%"}], "language": "es"}
            
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

        # Update the start time of the first segment
        diarization_result[0]["start"] = transcription_result["segments"][0]["start"]

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
    
    async def transform_audio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        segment_length = 160000
        current_length = waveform.shape[1]

        remainder = current_length % segment_length
        if remainder != 0:
            padding = segment_length - remainder
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        torchaudio.save(file_path, waveform, sample_rate)

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
        await AudioInfo.objects.acreate(
            file_name=file_name, 
            transcription=transcription,
            diarization=diarization,
            overlapp = overlappR,
            result = resultX
        )
