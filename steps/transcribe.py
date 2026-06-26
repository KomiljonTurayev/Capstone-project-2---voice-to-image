# steps/transcribe.py
import os
from huggingface_hub import InferenceClient

WHISPER_MODEL = "openai/whisper-large-v3"

def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    result = client.automatic_speech_recognition(audio_bytes, model=WHISPER_MODEL)
    return result.text
