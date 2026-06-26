# tests/test_pipeline.py
from unittest.mock import patch
import pytest
from pipeline import run, PipelineResult

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("HF_TOKEN", "test-token")

def test_run_returns_pipeline_result_with_all_fields():
    with patch("pipeline.transcribe", return_value="a sunset") as _t, \
         patch("pipeline.enhance", return_value="A vivid sunset...") as _e, \
         patch("pipeline.generate", return_value=b"PNG_BYTES") as _g:

        result = run(b"audio_bytes", "test.wav")

    assert isinstance(result, PipelineResult)
    assert result.transcript == "a sunset"
    assert result.enhanced_prompt == "A vivid sunset..."
    assert result.image == b"PNG_BYTES"
    assert result.models["stt"] == "openai/whisper-large-v3"
    assert result.models["llm"] == "claude-haiku-4-5-20251001"
    assert result.models["image"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert result.duration_seconds >= 0

def test_run_raises_runtime_error_on_transcription_failure():
    with patch("pipeline.transcribe", side_effect=Exception("network error")):
        with pytest.raises(RuntimeError, match="Transcription failed"):
            run(b"audio_bytes")

def test_run_raises_runtime_error_on_enhancement_failure():
    with patch("pipeline.transcribe", return_value="a sunset"), \
         patch("pipeline.enhance", side_effect=Exception("quota exceeded")):
        with pytest.raises(RuntimeError, match="Prompt enhancement failed"):
            run(b"audio_bytes")

def test_run_raises_runtime_error_on_generation_failure():
    with patch("pipeline.transcribe", return_value="a sunset"), \
         patch("pipeline.enhance", return_value="expanded prompt"), \
         patch("pipeline.generate", side_effect=Exception("model loading")):
        with pytest.raises(RuntimeError, match="Image generation failed"):
            run(b"audio_bytes")

def test_run_raises_on_missing_env_vars(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        run(b"audio_bytes")
