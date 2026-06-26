# steps/enhance.py
import os
from openai import OpenAI

ENHANCE_MODEL = "gpt-4o"
_SYSTEM_PROMPT = (
    "You are an expert at writing detailed image generation prompts. "
    "Given a short description, expand it into a vivid, detailed prompt "
    "suitable for Stable Diffusion. Include style, lighting, composition, "
    "mood, and artistic details. Return only the prompt text, nothing else."
)

def enhance(transcript: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=ENHANCE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content.strip()
