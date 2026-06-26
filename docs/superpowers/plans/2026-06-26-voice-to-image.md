# Voice to Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app that records or accepts a voice message, transcribes it with Whisper, enhances the transcript into a detailed image prompt with GPT-4o, generates an image with Stable Diffusion XL via HF Inference API, and displays all intermediate data in the UI.

**Architecture:** Layered module structure — `app.py` handles UI only, `pipeline.py` orchestrates three steps and owns the `PipelineResult` dataclass, each step lives in its own module under `steps/`. A shared `logger.py` writes structured step-by-step logs to the console.

**Tech Stack:** Python 3.10+, Streamlit ≥1.31, OpenAI SDK ≥1.0, huggingface-hub ≥0.20, Pillow ≥10, python-dotenv, pytest

## Global Constraints

- Python 3.10+ required (uses match/case-free code, but f-strings and dataclasses are fine)
- All API keys read from `os.environ` — never hardcoded
- `load_dotenv()` called once at pipeline.py module level (no-op on HF Spaces)
- `.env` must never be committed — covered by `.gitignore`
- All step functions are pure: accept inputs, return outputs, no side effects except network calls
- Console logging uses the shared `log` singleton from `logger.py`
- HF Spaces deployment: Space type = Streamlit, no Dockerfile needed

---

## File Map

| File | Responsibility |
|---|---|
| `logger.py` | `PipelineLogger` class + `log` singleton — structured console output |
| `steps/__init__.py` | Empty — marks `steps/` as a package |
| `steps/transcribe.py` | `transcribe(audio_bytes, filename) -> str` via OpenAI Whisper |
| `steps/enhance.py` | `enhance(transcript) -> str` via GPT-4o |
| `steps/generate.py` | `generate(prompt) -> bytes` via HF SDXL; exports `IMAGE_MODEL` constant |
| `pipeline.py` | `PipelineResult` dataclass + `run(audio_bytes, filename) -> PipelineResult` |
| `app.py` | Streamlit UI — calls `pipeline.run()`, displays `PipelineResult` |
| `requirements.txt` | Pinned dependency list |
| `.env.example` | Template for local secrets |
| `.gitignore` | Excludes `.env`, `__pycache__`, `.pytest_cache` |
| `README.md` | HF Spaces metadata header + setup guide + screenshots |
| `tests/__init__.py` | Empty |
| `tests/test_logger.py` | Logger unit tests |
| `tests/test_transcribe.py` | Transcription unit tests (mocked API) |
| `tests/test_enhance.py` | Enhancement unit tests (mocked API) |
| `tests/test_generate.py` | Generation unit tests (mocked API) |
| `tests/test_pipeline.py` | Pipeline integration tests (mocked steps) |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `steps/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: working Python environment with all dependencies installable

- [ ] **Step 1: Initialize git repo**

```bash
cd "D:/python_projects/Capstone project 2 - voice to image"
git init
git checkout -b main
```

- [ ] **Step 2: Create requirements.txt**

```
streamlit>=1.31.0
openai>=1.0.0
huggingface-hub>=0.20.0
python-dotenv>=1.0.0
Pillow>=10.0.0
pytest>=7.0.0
```

- [ ] **Step 3: Create .gitignore**

```
.env
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
.DS_Store
```

- [ ] **Step 4: Create .env.example**

```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

- [ ] **Step 5: Create directory structure and empty init files**

Create `steps/__init__.py` — empty file.
Create `tests/__init__.py` — empty file.

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 7: Commit scaffold**

```bash
git add requirements.txt .gitignore .env.example steps/__init__.py tests/__init__.py
git commit -m "chore: project scaffold with dependencies and structure"
```

---

### Task 2: Logger Module

**Files:**
- Create: `logger.py`
- Create: `tests/test_logger.py`

**Interfaces:**
- Produces: `log` singleton with methods `step(n, total, name, model)`, `done(message)`, `error(step, error)`, `finish(duration)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_logger.py
import logging
from logger import PipelineLogger

def test_step_logs_step_number_name_and_model(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.step(1, 3, "Transcribing audio", "whisper-1")
    assert "STEP 1/3" in caplog.text
    assert "Transcribing audio" in caplog.text
    assert "whisper-1" in caplog.text

def test_done_logs_checkmark_and_message(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.done("Transcript ready")
    assert "✓" in caplog.text
    assert "Transcript ready" in caplog.text

def test_error_logs_cross_step_and_error(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.ERROR, logger="pipeline"):
        logger.error("Transcription", "API timeout")
    assert "✗" in caplog.text
    assert "Transcription" in caplog.text
    assert "API timeout" in caplog.text

def test_finish_logs_done_and_duration(caplog):
    logger = PipelineLogger()
    with caplog.at_level(logging.INFO, logger="pipeline"):
        logger.finish(11.4)
    assert "DONE" in caplog.text
    assert "11.4" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_logger.py -v
```

Expected: `ModuleNotFoundError: No module named 'logger'`

- [ ] **Step 3: Implement logger.py**

```python
# logger.py
import logging
from datetime import datetime

class PipelineLogger:
    def __init__(self):
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        self._logger = logging.getLogger("pipeline")

    def step(self, n: int, total: int, name: str, model: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ── STEP {n}/{total} ─ {name:<35} (model: {model})")

    def done(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ✓  {message}")

    def error(self, step: str, error: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.error(f"[{ts}] ✗  {step} failed: {error}")

    def finish(self, duration: float) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ── DONE ── Total: {duration:.1f}s")

log = PipelineLogger()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_logger.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add logger.py tests/test_logger.py
git commit -m "feat: add structured pipeline logger"
```

---

### Task 3: Transcription Step

**Files:**
- Create: `steps/transcribe.py`
- Create: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `audio_bytes: bytes`, `filename: str`
- Produces: `transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_transcribe.py -v
```

Expected: `ModuleNotFoundError: No module named 'steps.transcribe'`

- [ ] **Step 3: Implement steps/transcribe.py**

```python
# steps/transcribe.py
import io
import os
from openai import OpenAI

WHISPER_MODEL = "whisper-1"

def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio_file)
    return result.text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transcribe.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add steps/transcribe.py tests/test_transcribe.py
git commit -m "feat: add Whisper transcription step"
```

---

### Task 4: Prompt Enhancement Step

**Files:**
- Create: `steps/enhance.py`
- Create: `tests/test_enhance.py`

**Interfaces:**
- Consumes: `transcript: str`
- Produces: `enhance(transcript: str) -> str`; exports `ENHANCE_MODEL = "gpt-4o"`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_enhance.py -v
```

Expected: `ModuleNotFoundError: No module named 'steps.enhance'`

- [ ] **Step 3: Implement steps/enhance.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_enhance.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add steps/enhance.py tests/test_enhance.py
git commit -m "feat: add GPT-4o prompt enhancement step"
```

---

### Task 5: Image Generation Step

**Files:**
- Create: `steps/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `prompt: str`
- Produces: `generate(prompt: str) -> bytes` (PNG bytes); exports `IMAGE_MODEL` constant

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generate.py -v
```

Expected: `ModuleNotFoundError: No module named 'steps.generate'`

- [ ] **Step 3: Implement steps/generate.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generate.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add steps/generate.py tests/test_generate.py
git commit -m "feat: add HF Stable Diffusion image generation step"
```

---

### Task 6: Pipeline Orchestrator

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `transcribe` from `steps.transcribe`, `enhance` from `steps.enhance`, `generate` and `IMAGE_MODEL` from `steps.generate`, `log` from `logger`
- Produces: `PipelineResult` dataclass, `run(audio_bytes: bytes, filename: str = "audio.wav") -> PipelineResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
from unittest.mock import patch
import pytest
from pipeline import run, PipelineResult

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
    assert result.models["stt"] == "whisper-1"
    assert result.models["llm"] == "gpt-4o"
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        run(b"audio_bytes")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Implement pipeline.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: All 15 tests PASSED (4 logger + 3 transcribe + 3 enhance + 3 generate + 5 pipeline)

- [ ] **Step 6: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with error handling and structured logging"
```

---

### Task 7: Streamlit UI

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `pipeline.run(audio_bytes, filename) -> PipelineResult`, `PipelineResult` dataclass
- Produces: Running Streamlit app at `http://localhost:8501`

**Note:** No unit tests for Streamlit UI — verify manually by running the app.

- [ ] **Step 1: Create app.py**

```python
# app.py
import streamlit as st
from pipeline import run, PipelineResult

st.set_page_config(page_title="Voice to Image", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice to Image")
st.caption("Speak your idea — get an AI-generated image.")

audio_bytes: bytes | None = None
filename = "audio.wav"

tab_record, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])

with tab_record:
    recorded = st.audio_input("Click the mic to record your message")
    if recorded:
        audio_bytes = recorded.read()
        filename = "recording.wav"

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a"],
        help="Supported formats: WAV, MP3, M4A",
    )
    if uploaded:
        audio_bytes = uploaded.read()
        filename = uploaded.name

st.divider()

if not audio_bytes:
    st.info("Record or upload a voice message above, then click Generate.")
else:
    if st.button("🖼️ Generate Image", type="primary", use_container_width=True):
        result: PipelineResult | None = None
        try:
            with st.spinner("Running pipeline… check the console for step-by-step logs."):
                result = run(audio_bytes, filename)
        except RuntimeError as e:
            st.error(str(e))

        if result:
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("📝 Transcript")
                st.write(result.transcript)

                st.subheader("✨ Image Prompt")
                st.write(result.enhanced_prompt)

                st.subheader("ℹ️ Run Info")
                st.markdown(
                    f"| Field | Value |\n"
                    f"|---|---|\n"
                    f"| STT model | `{result.models['stt']}` |\n"
                    f"| LLM model | `{result.models['llm']}` |\n"
                    f"| Image model | `{result.models['image']}` |\n"
                    f"| Duration | `{result.duration_seconds:.1f}s` |"
                )

            with col_right:
                st.subheader("🖼️ Generated Image")
                st.image(result.image, use_container_width=True)
                st.download_button(
                    label="⬇️ Download Image",
                    data=result.image,
                    file_name="generated.png",
                    mime="image/png",
                    use_container_width=True,
                )
```

- [ ] **Step 2: Create a local .env from the example**

```bash
cp .env.example .env
# Then edit .env and fill in your real OPENAI_API_KEY and HF_TOKEN
```

- [ ] **Step 3: Run the app locally**

```bash
streamlit run app.py
```

Expected: Browser opens at `http://localhost:8501`, shows title "Voice to Image" with two tabs.

- [ ] **Step 4: Verify the Record tab**

Open the Record tab. Click the microphone icon. Speak a short sentence (e.g., "a cat sitting on top of a snowy mountain"). Stop recording. The "Generate Image" button should appear.

- [ ] **Step 5: Verify the full pipeline**

Click "Generate Image". While spinning, open the terminal — you should see:

```
[HH:MM:SS] ── STEP 1/3 ─ Transcribing audio    (model: whisper-1)
[HH:MM:SS] ✓  Transcript: "a cat sitting on top of..."
[HH:MM:SS] ── STEP 2/3 ─ Enhancing prompt       (model: gpt-4o)
[HH:MM:SS] ✓  Prompt: "A majestic cat perched atop..."
[HH:MM:SS] ── STEP 3/3 ─ Generating image       (model: stabilityai/...)
[HH:MM:SS] ✓  Image generated in X.Xs
[HH:MM:SS] ── DONE ── Total: X.Xs
```

UI should then show transcript, prompt, run info table, and the generated image with a download button.

- [ ] **Step 6: Verify the Upload tab**

Upload a WAV or MP3 file. Click "Generate Image". Same flow as above.

- [ ] **Step 7: Verify error handling**

Temporarily set an invalid `OPENAI_API_KEY=bad-key` in `.env` and restart the app. Try to generate — the UI should show a red `st.error` box with a descriptive message. Restore the correct key.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI with record/upload tabs and pipeline display"
```

---

### Task 8: README and HF Spaces Deployment

**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: Deployable HF Space + documented local setup

- [ ] **Step 1: Create README.md**

The README must start with the HF Spaces YAML metadata block (required for the Space to render correctly), followed by setup instructions and workflow screenshots.

```markdown
---
title: Voice To Image
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# 🎙️ Voice to Image

Convert a spoken description into an AI-generated image in three steps:
**voice → transcript → image prompt → generated image**

## Workflow

```
[Your voice]
     │
     ▼  OpenAI Whisper
[Transcript: "a cat on a snowy mountain"]
     │
     ▼  GPT-4o
[Image prompt: "A majestic cat perched atop snow-capped peaks, golden hour..."]
     │
     ▼  Stable Diffusion XL (HF)
[Generated image]
```

## Local Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd voice-to-image
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...
```

Get your OpenAI key at platform.openai.com.
Get your HF token at huggingface.co/settings/tokens (read access is enough).

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Live Demo

[View on Hugging Face Spaces](<your-hf-space-url>)

## Usage

1. Open the **Record** tab and click the microphone to record your idea
2. Or open the **Upload** tab and upload a WAV/MP3/M4A file
3. Click **Generate Image**
4. View the transcript, enhanced prompt, model info, and generated image

> **Note:** The first generation may take 30–60 seconds on HF free tier while the model warms up.

## Screenshots

<!-- Add screenshots here after running the app -->

## Models

| Role | Model |
|---|---|
| Speech-to-Text | OpenAI `whisper-1` |
| Prompt Enhancement | OpenAI `gpt-4o` |
| Image Generation | `stabilityai/stable-diffusion-xl-base-1.0` via HF Inference API |
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: add README with HF Spaces metadata and setup guide"
```

- [ ] **Step 3: Push to GitHub**

Create a new GitHub repository named `voice-to-image` (public), then:

```bash
git remote add origin https://github.com/<your-username>/voice-to-image.git
git push -u origin main
```

- [ ] **Step 4: Create HF Space**

1. Go to huggingface.co → New Space
2. Space name: `voice-to-image`
3. SDK: **Streamlit**
4. Visibility: Public
5. Click **Create Space**

- [ ] **Step 5: Link HF Space to GitHub repo**

In the Space settings → Repository → link to your GitHub repo. Or push directly to the HF Space git remote:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/voice-to-image
git push hf main
```

- [ ] **Step 6: Add Space secrets**

In the HF Space → Settings → Repository secrets:
- Add `OPENAI_API_KEY` = your key
- Add `HF_TOKEN` = your token

- [ ] **Step 7: Verify Space is running**

Open the Space URL. Wait for the build to complete (2–3 minutes). Test the full pipeline in the browser.

- [ ] **Step 8: Add screenshots to README**

Take screenshots of:
1. The app with the Record tab open
2. The app after generation (showing transcript, prompt, run info, and image)

Add them to the README under the Screenshots section and commit:

```bash
git add README.md
git commit -m "docs: add usage screenshots to README"
git push origin main
git push hf main
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| Voice message input | Task 7 — Record tab (`st.audio_input`) |
| Audio file upload | Task 7 — Upload tab (`st.file_uploader`) |
| LLM converts to image description | Task 4 — `steps/enhance.py` (GPT-4o) |
| Image model generates picture | Task 5 — `steps/generate.py` (SDXL) |
| UI shows transcript, prompt, models | Task 7 — two-column layout |
| Console logs | Task 2 — `logger.py`; Task 6 — `pipeline.py` |
| Python + Streamlit | All tasks |
| README with screenshots | Task 8 |
| HF Spaces deployment | Task 8 |
| Git repo | Tasks 1, 2, 3, 4, 5, 6, 7, 8 — one commit per task |

All requirements covered. No placeholders remain. Type signatures are consistent across all tasks.
