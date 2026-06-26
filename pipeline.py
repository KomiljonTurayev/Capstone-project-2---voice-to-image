# pipeline.py
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

from steps.transcribe import transcribe, WHISPER_MODEL
from steps.enhance import enhance, ENHANCE_MODEL
from steps.generate import generate, IMAGE_MODEL
from logger import log

load_dotenv()

@dataclass
class PipelineResult:
    transcript: str
    enhanced_prompt: str
    image: bytes
    models: dict
    duration_seconds: float

def run(audio_bytes: bytes, filename: str = "audio.wav") -> PipelineResult:
    missing = [k for k in ("OPENAI_API_KEY", "HF_TOKEN") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    start = time.time()

    log.step(1, 3, "Transcribing audio", WHISPER_MODEL)
    try:
        transcript = transcribe(audio_bytes, filename)
    except Exception as e:
        log.error("Transcription", str(e))
        raise RuntimeError(f"Transcription failed: {e}") from e
    log.done(f'Transcript: "{transcript[:80]}"')

    log.step(2, 3, "Enhancing prompt", ENHANCE_MODEL)
    try:
        enhanced_prompt = enhance(transcript)
    except Exception as e:
        log.error("Prompt enhancement", str(e))
        raise RuntimeError(f"Prompt enhancement failed: {e}") from e
    log.done(f'Prompt: "{enhanced_prompt[:80]}"')

    log.step(3, 3, "Generating image", IMAGE_MODEL)
    try:
        image = generate(enhanced_prompt)
    except Exception as e:
        log.error("Image generation", str(e))
        raise RuntimeError(f"Image generation failed: {e}") from e

    duration = time.time() - start
    log.done(f"Image generated in {duration:.1f}s")
    log.finish(duration)

    return PipelineResult(
        transcript=transcript,
        enhanced_prompt=enhanced_prompt,
        image=image,
        models={"stt": WHISPER_MODEL, "llm": ENHANCE_MODEL, "image": IMAGE_MODEL},
        duration_seconds=duration,
    )
