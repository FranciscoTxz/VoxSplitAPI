# Generated by Django 5.1.7 on 2025-03-14 17:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("whisper_pyannote", "0002_audiofile_diarization_audiofile_trasncription"),
    ]

    operations = [
        migrations.AlterField(
            model_name="audiofile",
            name="trasncription",
            field=models.TextField(blank=True, null=True),
        ),
    ]
