# tests/test_transcribe.py
from unittest.mock import patch, MagicMock
import pytest
from steps.transcribe import transcribe, WHISPER_MODEL

@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

def test_transcribe_returns_text():
    mock_result = MagicMock()
    mock_result.text = "a sunset over snowy mountains"

    with patch("steps.transcribe.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.automatic_speech_recognition.return_value = mock_result

        result = transcribe(b"fake_audio_bytes", "test.wav")

    assert result == "a sunset over snowy mountains"

def test_transcribe_calls_whisper_model():
    mock_result = MagicMock()
    mock_result.text = "hello"

    with patch("steps.transcribe.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.automatic_speech_recognition.return_value = mock_result

        transcribe(b"fake_audio_bytes", "test.wav")

    call_kwargs = mock_client.automatic_speech_recognition.call_args[1]
    assert call_kwargs["model"] == WHISPER_MODEL

def test_transcribe_propagates_api_error():
    with patch("steps.transcribe.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.automatic_speech_recognition.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            transcribe(b"fake_audio_bytes")
