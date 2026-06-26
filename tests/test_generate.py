# tests/test_generate.py
import io
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image
from steps.generate import generate, IMAGE_MODEL

@pytest.fixture(autouse=True)
def set_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

def _fake_pil_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 0, 255))

def test_generate_returns_valid_png_bytes():
    with patch("steps.generate.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.text_to_image.return_value = _fake_pil_image()

        result = generate("a beautiful sunset")

    assert isinstance(result, bytes)
    img = Image.open(io.BytesIO(result))
    assert img.format == "PNG"

def test_generate_uses_correct_model():
    with patch("steps.generate.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.text_to_image.return_value = _fake_pil_image()

        generate("test prompt")

    mock_client.text_to_image.assert_called_once_with("test prompt", model=IMAGE_MODEL)

def test_generate_propagates_api_error():
    with patch("steps.generate.InferenceClient") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client
        mock_client.text_to_image.side_effect = Exception("Model loading")

        with pytest.raises(Exception, match="Model loading"):
            generate("test prompt")
