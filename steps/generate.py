# steps/generate.py
import io
import os
from huggingface_hub import InferenceClient

IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

def generate(prompt: str) -> bytes:
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    image = client.text_to_image(prompt, model=IMAGE_MODEL)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
