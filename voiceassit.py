import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
from dotenv import load_dotenv
import requests
from gtts import gTTS
from voice_pipeline import get_llm_response  # ✅ your pipeline function

# -------------------------------
# ✅ 1. Load environment variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# -------------------------------
# ✅ 2. Streamlit Page Settings
# -------------------------------
st.set_page_config(page_title="🎙 Voice Assistant", layout="centered")
st.title("🎙 Voice Assistant — Whisper + Chat + TTS")
st.markdown("🎤 बोलें → 🧠 Whisper → 💬 GPT → 🔊 आवाज़ सुने")

# -------------------------------
# ✅ 3. Audio Recording
# -------------------------------
st.subheader("🎙 Step 1 — अपनी आवाज़ रिकॉर्ड करें")
wav_audio_data = st_audiorec()

if wav_audio_data:
    st.success("✅ रिकॉर्डिंग पूरी हो गई!")
    st.audio(wav_audio_data, format="audio/wav")

    # Save audio in a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
        tfile.write(wav_audio_data)
        audio_path = tfile.name

    if st.button("▶ प्रोसेस करें"):
        transcript_text = ""
        assistant_text = ""

        # -------------------------------
        # ✅ 4. Transcription with Groq Whisper
        # -------------------------------
        with st.spinner("🧠 आपकी आवाज़ को टेक्स्ट में बदल रहे हैं..."):
            try:
                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                    data = {
                        "model": "whisper-large-v3-turbo",
                        "language": "hi",
                        "response_format": "text",
                    }

                    resp = requests.post(url, headers=headers, data=data, files=files, timeout=45)
                    resp.raise_for_status()
                    transcript_text = resp.text.strip()

            except requests.exceptions.Timeout:
                st.error("⏱️ सर्वर से जवाब देर से आया। कृपया फिर कोशिश करें।")
            except Exception as e:
                st.error(f"❌ ट्रांसक्रिप्शन असफल: {e}")

        # -------------------------------
        # ✅ 5. LLM Response
        # -------------------------------
        if transcript_text:
            st.subheader("📝 ट्रांसक्रिप्शन")
            st.write(transcript_text)

            with st.spinner("💬 GPT सोच रहा है..."):
                try:
                    assistant_text = get_llm_response(transcript_text)
                    st.subheader("🤖 असिस्टेंट का जवाब")
                    st.write(assistant_text)
                except Exception as e:
                    st.error(f"❌ GPT से जवाब नहीं मिला: {e}")

        # -------------------------------
        # ✅ 6. Text-to-Speech (gTTS - Hindi)
        # -------------------------------
        if assistant_text:
            with st.spinner("🎙 जवाब को आवाज़ में बदल रहे हैं..."):
                try:
                    tts = gTTS(text=assistant_text, lang="hi", slow=False)
                    tts_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                    tts.save(tts_path)

                    st.success("✅ आवाज़ तैयार है!")
                    st.audio(tts_path, format="audio/mp3")

                except Exception as e:
                    st.error(f"❌ TTS असफल: {e}")

        # 🧹 Clean up temp file
        try:
            os.remove(audio_path)
        except:
            pass

else:
    st.info("🎙 ऊपर रिकॉर्ड बटन दबाकर बोलना शुरू करें।")

# -------------------------------
# ✅ Footer
# -------------------------------
st.markdown("---")
st.caption("✨ Powered by Streamlit + Whisper + GPT + gTTS 🚀 (Optimized for Cloud)")
