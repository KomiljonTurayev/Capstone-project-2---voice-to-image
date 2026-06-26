# Voice to Image App — Design Spec

**Date:** 2026-06-26  
**Project:** Capstone Project 2 — Voice to Image  
**Status:** Approved

---

## Overview

A Streamlit web application that accepts a short voice message (live recording or file upload), converts it to text via OpenAI Whisper, enhances the transcript into a detailed image prompt via GPT-4o, and generates an image via Hugging Face Stable Diffusion. All intermediate data (transcript, prompt, model names, timing) is displayed in the UI. The pipeline logs structured output to the console at each step.

---

## AI Model Stack

| Role | Model | Provider |
|---|---|---|
| Speech-to-Text | `whisper-1` | OpenAI API |
| Prompt Enhancement | `gpt-4o` | OpenAI API |
| Image Generation | `stabilityai/stable-diffusion-xl-base-1.0` | HF Inference API |

**API Keys required:**
- `OPENAI_API_KEY` — for Whisper and GPT-4o
- `HF_TOKEN` — for HF Inference API (free tier available)

---

## Project Structure

```
voice-to-image/
├── app.py                  # Streamlit UI only — no business logic
├── pipeline.py             # Orchestrates the 3-step pipeline
├── steps/
│   ├── __init__.py
│   ├── transcribe.py       # Step 1: OpenAI Whisper STT
│   ├── enhance.py          # Step 2: GPT-4o prompt enhancement
│   └── generate.py         # Step 3: HF Stable Diffusion image generation
├── logger.py               # Structured console logger
├── requirements.txt
├── .env                    # Local secrets (never committed)
├── .env.example            # Template for API keys
├── .gitignore              # Excludes .env and __pycache__
└── README.md               # Setup guide + screenshots
```

---

## Data Flow

```
[User: mic or upload]
        │
        ▼
  audio bytes (WAV/MP3)
        │
        ▼
  transcribe.py ── Whisper API ──► transcript (str)
        │
        ▼
  enhance.py ──── GPT-4o API ───► enhanced_prompt (str)
        │
        ▼
  generate.py ─── HF API ───────► image (bytes)
        │
        ▼
  PipelineResult (dataclass)
        │
        ▼
  app.py ── Streamlit display
```

---

## Core Data Structure

```python
@dataclass
class PipelineResult:
    transcript: str           # Raw Whisper output
    enhanced_prompt: str      # GPT-4o enriched prompt
    image: bytes              # PNG image bytes from HF
    models: dict              # {"stt": "whisper-1", "llm": "gpt-4o", "image": "stabilityai/..."}
    duration_seconds: float   # Total pipeline time
```

---

## Pipeline (pipeline.py)

`pipeline.run(audio_bytes: bytes) -> PipelineResult`

1. Log Step 1 start → call `transcribe(audio_bytes)` → log result
2. Log Step 2 start → call `enhance(transcript)` → log result
3. Log Step 3 start → call `generate(enhanced_prompt)` → log result
4. Return `PipelineResult`

Each step is wrapped in `try/except`. On failure: log the error to console, raise a descriptive exception that `app.py` catches and displays via `st.error()`.

---

## Console Logging Format (logger.py)

```
[HH:MM:SS] ── STEP 1/3 ─ Transcribing audio     (model: whisper-1)
[HH:MM:SS] ✓  Transcript: "a sunset over snowy mountains"
[HH:MM:SS] ── STEP 2/3 ─ Enhancing prompt        (model: gpt-4o)
[HH:MM:SS] ✓  Prompt: "A breathtaking sunset casting golden rays..."
[HH:MM:SS] ── STEP 3/3 ─ Generating image        (model: stabilityai/sdxl)
[HH:MM:SS] ✓  Image generated in 8.2s
[HH:MM:SS] ── DONE ── Total: 11.4s
```

---

## Streamlit UI (app.py)

**Layout (top to bottom):**

1. **Title** — "Voice to Image"
2. **Input section** — Two tabs: "Record" (`st.audio_input`) and "Upload" (`st.file_uploader` accepting WAV/MP3/M4A)
3. **Generate button** — `st.button("Generate Image")`, disabled until audio is present
4. **Processing** — `st.spinner` shown during each pipeline step with step name visible
5. **Results** (shown after pipeline completes):
   - `st.subheader("Transcript")` + `st.write(result.transcript)`
   - `st.subheader("Image Prompt")` + `st.write(result.enhanced_prompt)`
   - `st.caption(f"Models: {result.models} | Duration: {result.duration_seconds:.1f}s")`
   - `st.image(result.image)`
   - `st.download_button("Download Image", result.image, "generated.png")`
6. **Error handling** — `st.error(str(e))` with a user-friendly message if any step fails

---

## Step Implementations

### transcribe.py
- Accept `audio_bytes: bytes` and a filename hint for the MIME type
- Call `openai.audio.transcriptions.create(model="whisper-1", file=audio_file)`
- Return transcript string

### enhance.py
- System prompt instructs GPT-4o to convert a short description into a detailed, vivid image generation prompt (style, lighting, composition, mood)
- User message is the transcript
- Return the enhanced prompt string

### generate.py
- Use `huggingface_hub.InferenceClient` with `HF_TOKEN`
- Call `client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")`
- Return image as bytes (convert PIL Image to bytes with `io.BytesIO`)

---

## Environment & Secrets

**Local development:**
```
# .env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```
Load with `python-dotenv` in `pipeline.py` (only when `.env` exists — no error on Spaces).

**HF Spaces:**
- Set `OPENAI_API_KEY` and `HF_TOKEN` in Space Settings → Repository secrets
- Same code reads from `os.environ` — no changes needed

---

## Deployment (HF Spaces)

- Space type: **Streamlit**
- No Dockerfile needed — HF Spaces natively supports Streamlit
- `README.md` must include HF Spaces metadata header for correct rendering:
  ```yaml
  ---
  title: Voice To Image
  emoji: 🎙️
  colorFrom: purple
  colorTo: blue
  sdk: streamlit
  sdk_version: "1.31.0"
  app_file: app.py
  pinned: false
  ---
  ```

---

## README Structure

1. Project description (1 paragraph)
2. Workflow diagram (voice → transcript → prompt → image)
3. Local setup: `git clone`, `pip install -r requirements.txt`, `.env` config, `streamlit run app.py`
4. HF Spaces live demo link
5. Screenshots showing full workflow (audio input → transcript → prompt → generated image)

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Missing API key | Clear `st.error` message: "OPENAI_API_KEY not set" |
| Whisper fails | Log error, raise with step name; UI shows which step failed |
| GPT-4o fails | Same as above |
| HF API fails / model loading | Same; suggest retrying (cold start on free tier) |
| No audio provided | Button disabled; no pipeline call made |

---

## Out of Scope

- User authentication
- Saving/history of past generations
- Custom model selection in UI
- Multi-language support beyond English
