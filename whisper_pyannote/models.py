from django.db import models

class AudioFile(models.Model):
    file = models.FileField(upload_to="audio/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name
    
    
class AudioInfo(models.Model):
    transcription = models.JSONField(blank=True, null=True)
    diarization = models.JSONField(blank=True, null=True)

    def __str__(self):
        return self.transcription, self.diarization
