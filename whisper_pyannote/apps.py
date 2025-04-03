from django.apps import AppConfig


class WhisperPyannoteConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "whisper_pyannote"

    def ready(self):
            # This ensures that the model is loaded when the application starts.
            import whisper_pyannote.model_loader