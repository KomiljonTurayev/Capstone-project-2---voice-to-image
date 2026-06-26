# steps/enhance.py
import os
import anthropic

ENHANCE_MODEL = "claude-haiku-4-5-20251001"
_SYSTEM_PROMPT = (
    "You are an expert at writing detailed image generation prompts. "
    "Given a short description, expand it into a vivid, detailed prompt "
    "suitable for Stable Diffusion. Include style, lighting, composition, "
    "mood, and artistic details. Return only the prompt text, nothing else."
)

def enhance(transcript: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=ENHANCE_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    content = message.content[0].text if message.content else ""
    return content.strip()
