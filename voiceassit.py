import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
from dotenv import load_dotenv
import requests
from openai import OpenAI
from gtts import gTTS
from voice_pipeline import get_llm_response  # ✅ from your pipeline

# --- Load keys ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="🎙 Voice Assistant", layout="centered")
st.title("🎙 Voice Assistant — Whisper + Chat + TTS")
st.markdown("🎤 Speak → 🧠 Whisper → 💬 GPT → 🔊 Reply")

# --- 1. Record Audio ---
st.subheader("Step 1 — Record your voice")
wav_audio_data = st_audiorec()

if wav_audio_data:
    st.success("✅ Recording complete!")
    st.audio(wav_audio_data, format="audio/wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
        tfile.write(wav_audio_data)
        audio_path = tfile.name

    if st.button("▶ Transcribe & Ask Assistant"):

        # --- 2. Transcribe with Whisper ---
        transcript_text = ""
        with st.spinner("🧠 Transcribing with Whisper..."):
            try:
                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                    data = {
                        "model": "whisper-large-v3-turbo",
                        "language": "hi",
                        "response_format": "text"
                    }
                    resp = requests.post(url, headers=headers, data=data, files=files, timeout=45)
                resp.raise_for_status()
                transcript_text = resp.text.strip()
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

        # --- 3. GPT Response via Pipeline ---
        if transcript_text:
            st.subheader("📝 Transcription")
            st.write(transcript_text)

            with st.spinner("💬 Thinking..."):
                try:
                    assistant_text = get_llm_response(transcript_text)
                    st.subheader("🤖 Assistant Reply")
                    st.write(assistant_text)
                except Exception as e:
                    st.error(f"❌ GPT Response failed: {e}")
                    assistant_text = ""

            # --- 4. Convert to Speech (Using gTTS) ---
            if assistant_text:
                with st.spinner("🎙 Generating speech (gTTS)..."):
                    try:
                        # ✅ Generate Hindi speech using gTTS
                        tts = gTTS(text=assistant_text, lang="hi")
                        tts_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                        tts.save(tts_path)

                        st.success("✅ Speech generated!")
                        st.audio(tts_path, format="audio/mp3")
                    except Exception as e:
                        st.error(f"❌ TTS generation failed: {e}")
        else:
            st.warning("⚠ No transcription result.")
else:
    st.info("🎙 Click the record button above to start.")

st.markdown("---")
st.caption("✨ Powered by Streamlit, Whisper, GPT, and gTTS 🚀")
