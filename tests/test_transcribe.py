# tests/test_transcribe.py
from unittest.mock import patch, MagicMock
import pytest
from steps.transcribe import transcribe, WHISPER_MODEL

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

def test_transcribe_returns_text():
    mock_result = MagicMock()
    mock_result.text = "a sunset over snowy mountains"

    with patch("steps.transcribe.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = mock_result

        result = transcribe(b"fake_audio_bytes", "test.wav")

    assert result == "a sunset over snowy mountains"

def test_transcribe_calls_whisper_model():
    mock_result = MagicMock()
    mock_result.text = "hello"

    with patch("steps.transcribe.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = mock_result

        transcribe(b"fake_audio_bytes", "test.wav")

    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == WHISPER_MODEL

def test_transcribe_propagates_api_error():
    with patch("steps.transcribe.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            transcribe(b"fake_audio_bytes")
