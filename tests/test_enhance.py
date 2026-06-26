# tests/test_enhance.py
from unittest.mock import patch, MagicMock
import pytest
from steps.enhance import enhance, ENHANCE_MODEL

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

def test_enhance_returns_stripped_prompt():
    expected = "A breathtaking sunset casting golden rays over snow-capped peaks"
    mock_content = MagicMock()
    mock_content.text = f"  {expected}  "
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("steps.enhance.anthropic.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_message

        result = enhance("a sunset over snowy mountains")

    assert result == expected

def test_enhance_uses_correct_model():
    mock_content = MagicMock()
    mock_content.text = "expanded prompt"
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("steps.enhance.anthropic.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_message

        enhance("test transcript")

    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == ENHANCE_MODEL

def test_enhance_passes_transcript_as_user_message():
    mock_content = MagicMock()
    mock_content.text = "expanded prompt"
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("steps.enhance.anthropic.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_message

        enhance("a foggy forest at dawn")

    call_kwargs = mock_client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "a foggy forest at dawn"}

def test_enhance_handles_empty_content():
    mock_message = MagicMock()
    mock_message.content = []

    with patch("steps.enhance.anthropic.Anthropic") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_message

        result = enhance("test transcript")

    assert result == ""
