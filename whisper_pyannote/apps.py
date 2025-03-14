from django.apps import AppConfig


class WhisperPyannoteConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "whisper_pyannote"

    def ready(self):
            # Aquí se asegura que el modelo se cargue cuando la aplicación arranque
            import whisper_pyannote.model_loader