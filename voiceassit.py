import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
import requests
from gtts import gTTS

# Import with error handling
try:
    from voice_pipeline import get_llm_response
except ImportError:
    st.error("❌ voice_pipeline module not found!")
    get_llm_response = None

# --- 🔑 Keys लोड करें (Streamlit Cloud Compatible) ---
# Streamlit Cloud में secrets.toml से पढ़ें, local में environment variable से
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


@st.cache_data(ttl=300)
def get_supported_languages():
    """समर्थित भाषाएं (cached)"""
    return {
        "hi": "🇮🇳 हिंदी",
        "en": "🇬🇧 English",
        "mr": "🇮🇳 मराठी",
        "gu": "🇮🇳 ગુજરાતી",
        "ta": "🇮🇳 தமிழ்",
        "te": "🇮🇳 తెలుగు"
    }


def cleanup_temp_files(*file_paths):
    """अस्थाई फाइलें साफ करें"""
    for fp in file_paths:
        try:
            if fp and os.path.exists(fp):
                os.unlink(fp)
        except Exception:
            pass


def transcribe_audio(audio_path: str, language: str = "hi") -> str:
    """
    Groq Whisper API से आवाज़ को टेक्स्ट में बदलें
    
    Args:
        audio_path: ऑडियो फाइल का path
        language: भाषा कोड (default: hi)
    
    Returns:
        ट्रांसक्रिप्ट टेक्स्ट
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY नहीं मिली!")
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {
            "model": "whisper-large-v3-turbo",
            "language": language,
            "response_format": "text",
            "temperature": 0.0  # More deterministic
        }
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
    
    resp.raise_for_status()
    return resp.text.strip()


def text_to_speech(text: str, lang: str = "hi") -> str:
    """
    टेक्स्ट को आवाज़ में बदलें (gTTS)
    
    Args:
        text: बोलने के लिए टेक्स्ट
        lang: भाषा कोड
    
    Returns:
        ऑडियो फाइल का path
    """
    # Limit text length for TTS (gTTS has limits)
    max_chars = 5000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    tts = gTTS(text=text, lang=lang, slow=False)
    tts_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    tts.save(tts_path)
    return tts_path


def initialize_session_state():
    """Session state को initialize करें"""
    defaults = {
        "audio_path": None,
        "tts_path": None,
        "transcript": "",
        "ai_response": "",
        "processing": False,
        "conversation_history": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def voice_assistant_feature():
    """
    🎙 Streamlit Cloud-friendly Voice Assistant
    - रिकॉर्ड आवाज़
    - ट्रांसक्राइब (Whisper via Groq)
    - GPT से जवाब
    - आवाज़ में सुने (gTTS)
    """
    
    # Initialize session state
    initialize_session_state()
    
    # API key check करें
    if not GROQ_API_KEY:
        st.error("❌ **GROQ_API_KEY नहीं मिली!**")
        st.info("""
        **Streamlit Cloud पर setup:**
        1. App settings में जाएं
        2. Secrets में जाएं
        3. Add करें:
        ```toml
        GROQ_API_KEY = "your_key_here"
        ```
        """)
        return
    
    if get_llm_response is None:
        st.error("❌ voice_pipeline module उपलब्ध नहीं है!")
        return
    
    # Header
    st.markdown("### 🎤 Voice Assistant")
    
    # Language selector
    languages = get_supported_languages()
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_lang = st.selectbox(
            "भाषा",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key="language_selector",
            label_visibility="collapsed"
        )
    
    with col1:
        st.caption(f"Selected: {languages[selected_lang]}")
    
    # --- 🎙 Audio Recording ---
    st.markdown("#### 🎙️ अपनी आवाज़ रिकॉर्ड करें")
    audio_bytes = st_audiorec()
    
    if audio_bytes:
        # Display recorded audio
        st.audio(audio_bytes, format="audio/wav")
        
        # Save to temporary file
        if st.session_state.audio_path:
            cleanup_temp_files(st.session_state.audio_path)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
            tfile.write(audio_bytes)
            st.session_state.audio_path = tfile.name
        
        # Process button
        col_btn1, col_btn2 = st.columns([2, 1])
        
        with col_btn1:
            process_btn = st.button(
                "▶ पूछें AI से",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing
            )
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                cleanup_temp_files(st.session_state.audio_path, st.session_state.tts_path)
                st.session_state.audio_path = None
                st.session_state.tts_path = None
                st.session_state.transcript = ""
                st.session_state.ai_response = ""
                st.rerun()
        
        if process_btn:
            st.session_state.processing = True
            transcript = ""
            
            # Step 1: Transcription
            with st.spinner("🧠 आपकी आवाज़ को समझ रहे हैं..."):
                try:
                    transcript = transcribe_audio(st.session_state.audio_path, selected_lang)
                    st.session_state.transcript = transcript
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout: API response में देरी हो रही है। कृपया फिर से कोशिश करें।")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ ट्रांसक्रिप्शन असफल: {str(e)}")
            
            # Step 2: Get AI Response
            if transcript:
                st.info(f"📝 **आपने कहा:** {transcript}")
                
                with st.spinner("💬 AI जवाब सोच रहा है..."):
                    try:
                        response_text = get_llm_response(transcript)
                        st.session_state.ai_response = response_text
                        
                        # Save to conversation history
                        st.session_state.conversation_history.append({
                            "user": transcript,
                            "ai": response_text
                        })
                        
                    except Exception as e:
                        st.error(f"❌ AI Error: {str(e)}")
                        response_text = ""
                
                # Step 3: Text to Speech
                if response_text:
                    
                    
                    with st.spinner("🎙 जवाब को आवाज़ में बदल रहे हैं..."):
                        try:
                            # Cleanup old TTS file
                            if st.session_state.tts_path:
                                cleanup_temp_files(st.session_state.tts_path)
                            
                            st.session_state.tts_path = text_to_speech(response_text, selected_lang)
                            st.audio(st.session_state.tts_path, format="audio/mp3")
                            
                        except Exception as e:
                            st.error(f"❌ TTS Error: {str(e)}")
                            st.info("💡 Tip: आप टेक्स्ट जवाब को ऊपर पढ़ सकते हैं।")
                else:
                    st.warning("⚠ AI से कोई जवाब नहीं मिला।")
            else:
                st.warning("⚠ कोई ट्रांसक्रिप्शन नहीं मिला। फिर से बोलें।")
            
            st.session_state.processing = False
    
    else:
        st.info("🎤 ऊपर रिकॉर्ड बटन दबाएं और बोलें...")
    
    # Show conversation history
    if st.session_state.conversation_history:
        with st.expander("📜 पिछली बातचीत देखें", expanded=False):
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                st.markdown(f"**👤 You:** {conv['user']}")
                st.markdown(f"**🤖 AI:** {conv['ai']}")
                if i < len(st.session_state.conversation_history) - 1:
                    st.divider()
    
    # Footer
    st.markdown("---")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.caption("✨ Powered by Whisper + GPT + gTTS 🚀")
    with col_f2:
        if st.button("🗑 Clear All History"):
            st.session_state.conversation_history = []
            cleanup_temp_files(st.session_state.audio_path, st.session_state.tts_path)
            st.session_state.audio_path = None
            st.session_state.tts_path = None
            st.success("✅ History cleared!")
            st.rerun()


# Cleanup function for temp files (best effort)
def cleanup_on_session_end():
    """Session end पर cleanup (Streamlit Cloud friendly)"""
    try:
        if hasattr(st.session_state, 'audio_path'):
            cleanup_temp_files(st.session_state.audio_path)
        if hasattr(st.session_state, 'tts_path'):
            cleanup_temp_files(st.session_state.tts_path)
    except Exception:
        pass
