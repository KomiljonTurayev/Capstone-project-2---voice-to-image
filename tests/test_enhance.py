# tests/test_enhance.py
from unittest.mock import patch, MagicMock
import pytest
from steps.enhance import enhance, ENHANCE_MODEL

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

def test_enhance_returns_stripped_prompt():
    expected = "A breathtaking sunset casting golden rays over snow-capped peaks"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = f"  {expected}  "

    with patch("steps.enhance.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance("a sunset over snowy mountains")

    assert result == expected

def test_enhance_uses_correct_model():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "expanded prompt"

    with patch("steps.enhance.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        enhance("test transcript")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == ENHANCE_MODEL

def test_enhance_passes_transcript_as_last_user_message():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "expanded prompt"

    with patch("steps.enhance.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        enhance("a foggy forest at dawn")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "a foggy forest at dawn"}

def test_enhance_handles_none_content():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    with patch("steps.enhance.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = enhance("test transcript")

    assert result == ""
