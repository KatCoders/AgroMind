import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
from dotenv import load_dotenv
import requests
from gtts import gTTS
from voice_pipeline import get_llm_response  # ✅ आपकी LLM पाइपलाइन

# --- 🔑 Keys लोड करें ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


def voice_assistant_feature():
    """
    🎙 Fun और compact Voice Assistant feature
    - रिकॉर्ड आवाज़
    - ट्रांसक्राइब (Whisper via Groq)
    - GPT से जवाब
    - आवाज़ में सुने (gTTS)
    """
   

    # --- 🎙 Record your voice ---
    audio_bytes = st_audiorec()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # अस्थाई फाइल में सेव
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
            tfile.write(audio_bytes)
            audio_path = tfile.name

        if st.button("▶ पूछें AI से"):
            with st.spinner("🧠 आपकी आवाज़ को समझ रहे हैं..."):
                transcript = ""
                try:
                    url = "https://api.groq.com/openai/v1/audio/transcriptions"
                    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                    with open(audio_path, "rb") as f:
                        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                        data = {"model": "whisper-large-v3-turbo", "language": "hi", "response_format": "text"}
                        resp = requests.post(url, headers=headers, data=data, files=files, timeout=45)
                    resp.raise_for_status()
                    transcript = resp.text.strip()
                except Exception as e:
                    st.error(f"❌ ट्रांसक्रिप्शन असफल: {e}")

            if transcript:
                st.info(f"📝 आपने कहा: {transcript}")
                with st.spinner("💬 AI जवाब सोच रहा है..."):
                    try:
                        response_text = get_llm_response(transcript)
                    except Exception as e:
                        st.error(f"❌ AI जवाब नहीं मिला: {e}")
                        response_text = ""

                if response_text:
                    st.success(f"🤖 AI: {response_text}")
                    with st.spinner("🎙 जवाब को आवाज़ में बदल रहे हैं..."):
                        try:
                            tts = gTTS(text=response_text, lang="hi")
                            tts_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                            tts.save(tts_path)
                            st.audio(tts_path, format="audio/mp3")
                        except Exception as e:
                            st.error(f"❌ आवाज़ बनाने में समस्या: {e}")
            else:
                st.warning("⚠ कोई ट्रांसक्रिप्शन नहीं मिला।")
    else:
        st.info("🎤 रिकॉर्ड बटन दबाएं और बोलें...")

    st.markdown("---")
    st.caption("✨ Whisper + GPT + gTTS से संचालित 🚀")
