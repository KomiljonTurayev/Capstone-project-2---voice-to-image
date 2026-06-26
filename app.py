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
