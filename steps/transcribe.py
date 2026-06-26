# steps/transcribe.py
import io
import os
from openai import OpenAI

WHISPER_MODEL = "whisper-1"

def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio_file)
    return result.text
