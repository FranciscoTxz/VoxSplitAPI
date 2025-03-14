from rest_framework import serializers
from .models import AudioFile, AudioInfo

class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = '__all__'

class AudioInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioInfo
        fields = "__all__"