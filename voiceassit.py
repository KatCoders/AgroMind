import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
import requests
from gtts import gTTS
import io
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from voice_pipeline import *
except ImportError:
    st.error("❌ voice_pipeline module not found!")
    get_llm_response = None
    chain = None

# --- 🔑 Keys लोड करें (Streamlit Cloud Compatible) ---
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
        "te": "🇮🇳 తెలుగు",
        "bn": "🇮🇳 বাংলা",
        "pa": "🇮🇳 ਪੰਜਾਬੀ"
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
    Groq Whisper API से आवाज़ को टेक्स्ट में बदलें (Optimized)
    
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
            "model": "whisper-large-v3",  # More accurate than turbo
            "language": language,
            "response_format": "verbose_json",  # Get confidence scores
            "temperature": 0.0,
            "prompt": get_language_prompt(language)  # Context hint for better accuracy
        }
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=90)
    
    resp.raise_for_status()
    result = resp.json()
    
    # Return text with confidence check
    text = result.get("text", "").strip()
    
    # Log confidence if available (for debugging)
    if "segments" in result and result["segments"]:
        avg_confidence = sum(s.get("no_speech_prob", 0) for s in result["segments"]) / len(result["segments"])
        if avg_confidence > 0.8:
            st.warning("⚠️ Audio quality low. Speak clearly near microphone.")
    
    return text


def get_language_prompt(lang: str) -> str:
    """Language-specific prompts for better transcription"""
    prompts = {
        "hi": "यह एक हिंदी वॉयस नोट है। सही शब्दों का उपयोग करें।",
        "en": "This is a clear English voice note with proper pronunciation.",
        "mr": "हे मराठी भाषेतील आवाज संदेश आहे।",
        "gu": "આ ગુજરાતી ભાષામાં વૉઇસ નોંધ છે.",
        "ta": "இது தெளிவான தமிழ் குரல் குறிப்பு.",
        "te": "ఇది స్పష్టమైన తెలుగు వాయిస్ నోట్.",
        "bn": "এটি একটি বাংলা ভয়েস নোট।",
        "pa": "ਇਹ ਇੱਕ ਪੰਜਾਬੀ ਵੌਇਸ ਨੋਟ ਹੈ।"
    }
    return prompts.get(lang, "")


def text_to_speech(text: str, lang: str = "hi") -> bytes:
    """
    टेक्स्ट को आवाज़ में बदलें (gTTS) - Returns bytes instead of file
    
    Args:
        text: बोलने के लिए टेक्स्ट
        lang: भाषा कोड
    
    Returns:
        ऑडियो bytes
    """
    # Limit text length for TTS
    max_chars = 3000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    # Use faster speech rate
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save to BytesIO instead of file (faster)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    return audio_bytes.getvalue()


def initialize_session_state():
    """Session state को initialize करें"""
    defaults = {
        "audio_path": None,
        "transcript": "",
        "ai_response": "",
        "processing": False,
        "conversation_history": [],
        "chat_history": [],
        "audio_quality_tips_shown": False,
        "voice_enabled": True,
        "tts_system": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_audio_quality_tips():
    """Audio quality tips"""
    if not st.session_state.audio_quality_tips_shown:
        with st.expander("💡 बेहतर परिणाम के लिए टिप्स", expanded=True):
            st.markdown("""
            **सही तरीके से बोलें:**
            - 📱 माइक्रोफोन के पास रहें (15-20 cm)
            - 🔇 शांत जगह चुनें
            - 🗣️ स्पष्ट और धीरे बोलें
            - ⏸️ शब्दों के बीच छोटा pause दें
            - 🎤 अच्छी audio quality के लिए हेडफोन माइक use करें
            """)
            st.session_state.audio_quality_tips_shown = True


def voice_assistant_feature():
    """
    🎙 Optimized Streamlit Cloud Voice Assistant
    - Fast processing
    - Better speech recognition
    - Improved error handling
    """
    
    # Initialize session state
    initialize_session_state()
    
    # API key check
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
    
    # Check if voice_pipeline is available
    if 'chain' not in globals() or chain is None:
        st.error("❌ voice_pipeline module उपलब्ध नहीं है!")
        return
    
    # Header with tips
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown("#### 🎙️ रिकॉर्ड करें")
    with col_head2:
        if st.button("💡 Tips"):
            st.session_state.audio_quality_tips_shown = False
    
    show_audio_quality_tips()
    
    # Language selector
    languages = get_supported_languages()
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_lang = st.selectbox(
            "भाषा चुनें",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key="language_selector",
            label_visibility="collapsed"
        )
    
    with col1:
        # Recording tips inline
        st.caption("🔴 Record बटन दबाएं → बोलें → Stop दबाएं")
    
    audio_bytes = st_audiorec()
    
    if audio_bytes:
        # Check audio size
        audio_size_kb = len(audio_bytes) / 1024
        if audio_size_kb < 5:
            st.warning("⚠️ Audio बहुत छोटी है। 2-3 सेकंड तक बोलें।")
        elif audio_size_kb > 1024:
            st.warning("⚠️ Audio बहुत बड़ी है। कम बोलें या दोबारा रिकॉर्ड करें।")
        
        # Save to temporary file
        if st.session_state.audio_path:
            cleanup_temp_files(st.session_state.audio_path)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
            tfile.write(audio_bytes)
            st.session_state.audio_path = tfile.name
        
        # Process button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            process_btn = st.button(
                "🚀 पूछें AI से",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing
            )
        
        with col_btn2:
            if st.button("🔄 Re-record", use_container_width=True):
                cleanup_temp_files(st.session_state.audio_path)
                st.session_state.audio_path = None
                st.rerun()
        
        with col_btn3:
            if st.button("🗑️ Clear", use_container_width=True):
                cleanup_temp_files(st.session_state.audio_path)
                st.session_state.audio_path = None
                st.session_state.transcript = ""
                st.session_state.ai_response = ""
                st.rerun()
        
        if process_btn:
            st.session_state.processing = True
            transcript = ""
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Transcription
            status_text.text("🧠 आपकी आवाज़ को समझ रहे हैं... (1/3)")
            progress_bar.progress(33)
            
            try:
                transcript = transcribe_audio(st.session_state.audio_path, selected_lang)
                st.session_state.transcript = transcript
                
                if not transcript:
                    st.error("❌ कोई शब्द नहीं मिला। कृपया फिर से स्पष्ट बोलें।")
                    st.session_state.processing = False
                    progress_bar.empty()
                    status_text.empty()
                    return
                    
            except requests.exceptions.Timeout:
                st.error("❌ Timeout: Network slow है। फिर से कोशिश करें।")
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
                return
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Error: {str(e)}")
                st.info("💡 Check your internet connection और GROQ API key")
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
                return
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
                return
            
            # Step 2: Get AI Response
            if transcript:
                st.success(f"✅ **आपने कहा:** {transcript}")
                
                status_text.text("💬 AI जवाब तैयार कर रहा है... (2/3)")
                progress_bar.progress(66)
                
                full_response = ""
                try:
                    response_placeholder = st.empty()
                    response_placeholder.markdown("🤖 सोच रहा हूं... 🧠")
                    
                    # Stream response from chain
                    for chunk in chain.stream({"question": transcript}):
                        full_response += chunk
                        response_placeholder.markdown(f"**🤖 AI का जवाब:**\n\n{full_response}")
                    
                    st.session_state.ai_response = full_response
                        
                except Exception as e:
                    error_msg = f"जवाब तैयार करने में समस्या: {str(e)}"
                    response_placeholder.error(f"❌ {error_msg}")
                    full_response = "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका। कृपया फिर से कोशिश करें।"
                    logger.error(f"LLM generation error: {e}")
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": transcript, 
                    "type": "text",
                    "timestamp": datetime.now().isoformat()
                })
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "type": "text",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save to conversation history
                st.session_state.conversation_history.append({
                    "user": transcript,
                    "ai": full_response,
                    "lang": selected_lang
                })
                
                # Step 3: Generate audio response
                if full_response and full_response != "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका। कृपया फिर से कोशिश करें।":
                    status_text.text("🎙 जवाब को आवाज़ में बदल रहे हैं... (3/3)")
                    progress_bar.progress(90)
                    
                    try:
                        # Check if TTS system is available from voice_pipeline
                        if st.session_state.voice_enabled and hasattr(st.session_state, 'tts_system') and st.session_state.tts_system:
                            audio_bytes_tts = st.session_state.tts_system.generate_audio(full_response)
                        else:
                            # Fallback to gTTS
                            audio_bytes_tts = text_to_speech(full_response, selected_lang)
                        
                        if audio_bytes_tts:
                            progress_bar.progress(100)
                            status_text.text("✅ पूरा हुआ!")
                            st.audio(audio_bytes_tts, format="audio/mp3")
                        else:
                            st.info("💡 टेक्स्ट जवाब तैयार है, लेकिन आवाज़ नहीं बनाई जा सकी।")
                            
                    except Exception as e:
                        st.error(f"❌ TTS Error: {str(e)}")
                        st.info("💡 आप टेक्स्ट जवाब ऊपर पढ़ सकते हैं।")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            st.session_state.processing = False
            
            # Auto-scroll to response
            st.markdown('<div id="response"></div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 ऊपर 🔴 Record बटन दबाएं और बोलें...")
    
    # Show conversation history (last 5)
    if st.session_state.conversation_history:
        st.markdown("---")
        with st.expander("📜 पिछली बातचीत", expanded=False):
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                col_hist1, col_hist2 = st.columns([1, 20])
                with col_hist1:
                    st.markdown(f"**{len(st.session_state.conversation_history)-i}.**")
                with col_hist2:
                    st.markdown(f"**👤:** {conv['user']}")
                    with st.expander("🤖 AI Response"):
                        st.markdown(conv['ai'])
                if i < min(4, len(st.session_state.conversation_history) - 1):
                    st.divider()
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    
    with col_f1:
        st.caption("✨ Powered by Whisper-v3 + GPT + gTTS")
    
    with col_f2:
        st.caption(f"💬 {len(st.session_state.conversation_history)} conversations")
    
    with col_f3:
        if st.button("🗑 Clear All"):
            st.session_state.conversation_history = []
            st.session_state.chat_history = []
            cleanup_temp_files(st.session_state.audio_path)
            st.session_state.audio_path = None
            st.success("✅ Cleared!")
            st.rerun()


# Cleanup on session end
def cleanup_on_session_end():
    """Session end पर cleanup"""
    try:
        if hasattr(st.session_state, 'audio_path'):
            cleanup_temp_files(st.session_state.audio_path)
    except Exception:
        pass
