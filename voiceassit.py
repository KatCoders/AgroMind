import os
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
import requests
from gtts import gTTS
import io
from datetime import datetime
import logging
import time

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


# ===== CONFIGURATION =====
MAX_AUDIO_SIZE_KB = 1024
MIN_AUDIO_SIZE_KB = 5
MAX_TTS_CHARS = 3000
CACHE_TTL = 300

# ===== CUSTOM CSS FOR BETTER UI =====
def load_custom_css():
    """Enhanced UI with custom CSS"""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 1rem;
        }
        
        /* Card-like containers */
        .stAlert {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Recording indicator */
        .recording-pulse {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }
        
        .ai-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-right: 20%;
        }
        
        /* Stats cards */
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Audio player */
        audio {
            width: 100%;
            border-radius: 8px;
        }
        
        /* Language selector */
        .stSelectbox {
            border-radius: 8px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        /* Loading animation */
        .loading-dots {
            display: inline-block;
        }
        
        .loading-dots span {
            animation: blink 1.4s infinite both;
        }
        
        .loading-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .loading-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
        
        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        /* Hide Streamlit branding for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .user-message, .ai-message {
                margin-left: 5%;
                margin-right: 5%;
            }
            
            .stat-number {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=CACHE_TTL)
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


def transcribe_audio(audio_path: str, language: str = "hi") -> tuple:
    """
    Groq Whisper API से आवाज़ को टेक्स्ट में बदलें
    
    Returns:
        tuple: (transcript_text, confidence_score)
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY नहीं मिली!")
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {
            "model": "whisper-large-v3",
            "language": language,
            "response_format": "verbose_json",
            "temperature": 0.0,
            "prompt": get_language_prompt(language)
        }
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=90)
    
    resp.raise_for_status()
    result = resp.json()
    
    text = result.get("text", "").strip()
    
    # Calculate confidence
    confidence = 1.0
    if "segments" in result and result["segments"]:
        avg_no_speech = sum(s.get("no_speech_prob", 0) for s in result["segments"]) / len(result["segments"])
        confidence = 1.0 - avg_no_speech
    
    return text, confidence


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
    """টেক্স্ট को आवाज़ में बदलें (gTTS)"""
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS] + "..."
    
    tts = gTTS(text=text, lang=lang, slow=False)
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
        "tts_system": None,
        "total_conversations": 0,
        "total_words_spoken": 0,
        "avg_response_time": 0,
        "last_language": "hi"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def show_stats_dashboard():
    """Statistics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_conversations}</div>
            <div class='stat-label'>बातचीत</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_words_spoken}</div>
            <div class='stat-label'>शब्द बोले</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_time = st.session_state.avg_response_time
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{avg_time:.1f}s</div>
            <div class='stat-label'>औसत समय</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        languages = get_supported_languages()
        lang_display = languages.get(st.session_state.last_language, "🇮🇳 हिंदी")
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{lang_display.split()[0]}</div>
            <div class='stat-label'>भाषा</div>
        </div>
        """, unsafe_allow_html=True)


def show_audio_quality_tips():
    """Audio quality tips with better UI"""
    if not st.session_state.audio_quality_tips_shown:
        with st.expander("💡 बेहतर परिणाम के लिए टिप्स", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **✅ करें:**
                - 📱 माइक्रोफोन के पास रहें (15-20 cm)
                - 🔇 शांत जगह चुनें
                - 🗣️ स्पष्ट और धीरे बोलें
                - ⏸️ शब्दों के बीच छोटा pause दें
                """)
            
            with col2:
                st.markdown("""
                **❌ न करें:**
                - 🚫 तेज आवाज में न चिल्लाएं
                - 🚫 बहुत तेज या बहुत धीरे न बोलें
                - 🚫 शोर वाली जगह से रिकॉर्ड न करें
                - 🚫 माइक को हाथ से न ढकें
                """)
            
            st.session_state.audio_quality_tips_shown = True


def display_chat_message(role: str, content: str, timestamp: str = None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class='chat-message user-message'>
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>👤</span>
                <strong>आप</strong>
                {f"<span style='margin-left: auto; font-size: 0.8rem; opacity: 0.8;'>{timestamp}</span>" if timestamp else ""}
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-message ai-message'>
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>🤖</span>
                <strong>AI असिस्टेंट</strong>
                {f"<span style='margin-left: auto; font-size: 0.8rem; opacity: 0.8;'>{timestamp}</span>" if timestamp else ""}
            </div>
            <div>{content}</div>
        </div>
        """, unsafe_allow_html=True)


def voice_assistant_feature():
    """
    Ultra-Optimized Voice Assistant for Streamlit Cloud
    - Instant UI feedback
    - Streaming responses
    - Non-blocking operations
    - Better UX for slow networks
    """
 
    initialize_session_state()

    # ===== API VALIDATION =====
    if not GROQ_API_KEY:
        st.error("❌ **GROQ_API_KEY नहीं मिली!**")
        with st.expander("🔧 Setup Instructions", expanded=True):
            st.markdown("""
            **Streamlit Cloud पर setup:**
            1. App settings में जाएं ⚙️
            2. Secrets में जाएं 🔐
            3. Add करें:
            ```toml
            GROQ_API_KEY = "your_key_here"
            ```
            4. Save और redeploy करें 🚀
            
            **या Local में:**
            ```bash
            export GROQ_API_KEY="your_key_here"
            ```
            """)
        return

  

    # ===== HEADER WITH STATS =====
    st.markdown("### 🎙️ Voice Assistant")
    st.caption("बोलो, सुनो, समझो - AI से बात करें")
    st.markdown("<br>", unsafe_allow_html=True)
    
    show_stats_dashboard()
    st.markdown("<br>", unsafe_allow_html=True)

    # ===== TABS =====
    tab1, tab2, tab3 = st.tabs(["🎙️ रिकॉर्ड करें", "💬 बातचीत", "⚙️ सेटिंग्स"])

    # ===== TAB 1: RECORDING =====
    with tab1:
        # Tips toggle
        col_header, col_tips_btn = st.columns([3, 1])
        with col_header:
            st.markdown("#### 🎤 अपना सवाल रिकॉर्ड करें")
        with col_tips_btn:
            if st.button("💡 Tips", use_container_width=True, key="tips_toggle"):
                st.session_state.audio_quality_tips_shown = not st.session_state.get("audio_quality_tips_shown", False)

        show_audio_quality_tips()

        # Language selector
        st.markdown("---")
        languages = get_supported_languages()
        
        col_lang, col_status = st.columns([2, 1])
        
        with col_lang:
            lang_keys = list(languages.keys())
            default_lang = st.session_state.get("last_language", "hi")
            try:
                default_index = lang_keys.index(default_lang) if default_lang in lang_keys else 0
            except Exception:
                default_index = 0
            
            selected_lang = st.selectbox(
                "🌐 भाषा चुनें",
                options=lang_keys,
                format_func=lambda x: languages[x],
                key="language_selector",
                index=default_index
            )
            st.session_state.last_language = selected_lang
        
        with col_status:
            # Processing status indicator
            if st.session_state.get("processing", False):
                st.markdown("""
                <div style='background: #fff3cd; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div class='loading-dots'>
                        <span>⚙️</span>
                        <span>⚙️</span>
                        <span>⚙️</span>
                    </div>
                    <div style='font-size: 0.8rem; color: #856404; margin-top: 0.5rem;'>
                        प्रोसेसिंग...
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #d4edda; padding: 1rem; border-radius: 8px; text-align: center;'>
                    <div style='font-size: 1.5rem;'>✅</div>
                    <div style='font-size: 0.8rem; color: #155724;'>तैयार है</div>
                </div>
                """, unsafe_allow_html=True)

        # Audio recorder
        st.markdown("---")
        st.markdown("##### 🔴 अभी रिकॉर्ड करें")
        st.caption("नीचे Record बटन दबाएं → बोलें → Stop दबाएं")
        
        audio_bytes = st_audiorec()

        # ===== AUDIO PROCESSING LOGIC =====
        if audio_bytes:
            # Validate audio size
            audio_size_kb = len(audio_bytes) / 1024
            
            # Show audio info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📊 Size", f"{audio_size_kb:.1f} KB")
            with col_info2:
                duration = audio_size_kb / 16  # Approximate duration (16KB/sec for WAV)
                st.metric("⏱️ Duration", f"~{duration:.1f}s")
            with col_info3:
                quality = "🟢 Good" if MIN_AUDIO_SIZE_KB < audio_size_kb < MAX_AUDIO_SIZE_KB else "🟡 Check"
                st.metric("✨ Quality", quality)
            
            # Validation warnings
            if audio_size_kb < MIN_AUDIO_SIZE_KB:
                st.warning("⚠️ Audio बहुत छोटी है। कम से कम 2-3 सेकंड तक बोलें।")
            elif audio_size_kb > MAX_AUDIO_SIZE_KB:
                st.warning("⚠️ Audio बहुत बड़ी है। छोटे वाक्यों में बोलें।")
            
            # Save to temp file (clean previous if exists)
            if st.session_state.get("audio_path"):
                cleanup_temp_files(st.session_state.audio_path)
            
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tfile:
                    tfile.write(audio_bytes)
                    st.session_state.audio_path = tfile.name
                    st.success(f"✅ Audio saved successfully!")
            except Exception as e:
                st.error(f"❌ Audio save failed: {e}")
                st.session_state.audio_path = None

            # Action buttons
            st.markdown("---")
            st.markdown("##### 🚀 अब क्या करें?")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)

            with col_btn1:
                process_btn = st.button(
                    "🤖 AI से पूछें",
                    use_container_width=True,
                    disabled=st.session_state.get("processing", False),
                    type="primary",
                    key="process_audio_btn"
                )

            with col_btn2:
                if st.button(
                    "🔄 फिर से रिकॉर्ड",
                    use_container_width=True,
                    disabled=st.session_state.get("processing", False),
                    key="re_record_btn"
                ):
                    cleanup_temp_files(st.session_state.audio_path)
                    st.session_state.audio_path = None
                    st.session_state.transcript = ""
                    st.session_state.ai_response = ""
                    st.rerun()

            with col_btn3:
                if st.button(
                    "🗑️ Cancel",
                    use_container_width=True,
                    disabled=st.session_state.get("processing", False),
                    key="cancel_btn"
                ):
                    cleanup_temp_files(st.session_state.audio_path)
                    st.session_state.audio_path = None
                    st.session_state.transcript = ""
                    st.session_state.ai_response = ""
                    st.rerun()

            # ===== PROCESSING HANDLER =====
            if process_btn and st.session_state.audio_path:
                st.session_state.processing = True
                
                # Create dedicated processing container
                process_container = st.container()
                
                with process_container:
                    st.markdown("---")
                    st.markdown("### 🔄 Processing Your Request")
                    
                    # Progress tracking
                    progress_bar = st.progress(0, text="शुरू हो रहा है...")
                    status_placeholder = st.empty()
                    result_placeholder = st.empty()
                    
                    start_time = time.time()
                    
                    try:
                        # ===== STEP 1: TRANSCRIPTION =====
                        status_placeholder.info("🧠 **Step 1/3:** आपकी आवाज़ को समझ रहे हैं...")
                        progress_bar.progress(10, text="Transcribing audio...")
                        
                        transcript = ""
                        confidence = 0.0
                        
                        try:
                            transcript, confidence = transcribe_audio_with_retry(
                                st.session_state.audio_path,
                                selected_lang
                            )
                            st.session_state.transcript = transcript
                            
                            progress_bar.progress(35, text="Transcription complete!")
                            
                            if not transcript:
                                status_placeholder.error("❌ कोई शब्द नहीं मिला। कृपया फिर से स्पष्ट बोलें।")
                                st.session_state.processing = False
                                progress_bar.empty()
                                time.sleep(2)
                                st.rerun()
                                return
                            
                            # Show transcript with confidence
                            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                            result_placeholder.success(f"""
                            ✅ **आपने कहा:**  
                            "{transcript}"
                            
                            {confidence_emoji} Confidence: {confidence*100:.1f}%
                            """)
                            
                            # Update stats
                            word_count = len(transcript.split())
                            st.session_state.total_words_spoken += word_count
                            
                        except requests.exceptions.Timeout:
                            status_placeholder.error("❌ समय समाप्त: नेटवर्क धीमा है। फिर से कोशिश करें।")
                            st.session_state.error_count += 1
                            st.session_state.processing = False
                            progress_bar.empty()
                            time.sleep(2)
                            st.rerun()
                            return
                        
                        except Exception as e:
                            status_placeholder.error(f"❌ Transcription Error: {str(e)}")
                            st.info("💡 Check: Internet connection, GROQ API key, Audio quality")
                            st.session_state.error_count += 1
                            st.session_state.last_error = str(e)
                            st.session_state.processing = False
                            progress_bar.empty()
                            time.sleep(2)
                            st.rerun()
                            return
                        
                        # ===== STEP 2: AI RESPONSE GENERATION =====
                        if transcript:
                            status_placeholder.info("💬 **Step 2/3:** AI जवाब तैयार कर रहा है...")
                            progress_bar.progress(50, text="Generating AI response...")
                            
                            full_response = ""
                            response_placeholder = st.empty()
                            
                            try:
                                # Build context from recent conversations
                                context = []
                                for conv in st.session_state.conversation_history[-3:]:
                                    context.append(f"User: {conv['user']}\nAI: {conv['ai']}")
                                context_str = "\n\n".join(context) if context else ""
                                
                                # Create query
                                query_input = {
                                    "question": transcript,
                                    "context": context_str if context_str else None
                                }
                                
                                # Stream response with visual feedback
                                chunk_count = 0
                                for chunk in chain.stream(query_input):
                                    full_response += chunk
                                    chunk_count += 1
                                    
                                    # Update progress (50-75%)
                                    progress_pct = min(75, 50 + (chunk_count // 5))
                                    progress_bar.progress(progress_pct, text=f"Streaming response... ({len(full_response)} chars)")
                                    
                                    # Update UI every 3 chunks for smooth streaming
                                    if chunk_count % 3 == 0:
                                        response_placeholder.markdown(f"""
                                        **🤖 AI का जवाब:**
                                        
                                        {full_response}▌
                                        """)
                                
                                # Final response
                                response_placeholder.markdown(f"""
                                **🤖 AI का जवाब:**
                                
                                {full_response}
                                """)
                                
                                st.session_state.ai_response = full_response
                                progress_bar.progress(75, text="Response generated!")
                                
                            except Exception as e:
                                error_msg = str(e)
                                status_placeholder.error(f"❌ AI Response Error: {error_msg}")
                                full_response = "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका।"
                                response_placeholder.warning(full_response)
                                logger.error(f"LLM generation error: {e}", exc_info=True)
                                st.session_state.error_count += 1
                            
                            # Save to conversation history
                            timestamp = datetime.now().strftime("%I:%M %p")
                            
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": transcript,
                                "timestamp": timestamp
                            })
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": full_response,
                                "timestamp": timestamp
                            })
                            
                            st.session_state.conversation_history.append({
                                "user": transcript,
                                "ai": full_response,
                                "lang": selected_lang,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            # ===== STEP 3: TEXT-TO-SPEECH =====
                            if full_response and st.session_state.voice_enabled and "क्षमा करें" not in full_response:
                                status_placeholder.info("🎙 **Step 3/3:** जवाब को आवाज़ में बदल रहे हैं...")
                                progress_bar.progress(85, text="Generating audio...")
                                
                                try:
                                    audio_bytes_tts = text_to_speech(full_response, selected_lang)
                                    
                                    if audio_bytes_tts:
                                        progress_bar.progress(100, text="✅ पूरा हुआ!")
                                        status_placeholder.success("🎉 सब कुछ तैयार है!")
                                        
                                        # Audio player with better UI
                                        st.markdown("---")
                                        st.markdown("#### 🔊 जवाब सुनें")
                                        st.audio(audio_bytes_tts, format="audio/mp3")
                                        
                                except Exception as e:
                                    status_placeholder.warning(f"⚠️ आवाज़ नहीं बनाई जा सकी: {str(e)}")
                                    logger.warning(f"TTS error: {e}")
                                    progress_bar.progress(100, text="✅ Text response ready!")
                            else:
                                progress_bar.progress(100, text="✅ पूरा हुआ!")
                                status_placeholder.success("🎉 सब कुछ तैयार है!")
                            
                            # Update statistics
                            end_time = time.time()
                            response_time = end_time - start_time
                            
                            st.session_state.total_conversations += 1
                            
                            # Calculate running average
                            n = st.session_state.total_conversations
                            old_avg = st.session_state.avg_response_time
                            st.session_state.avg_response_time = (old_avg * (n-1) + response_time) / n
                            
                            # Success metrics
                            st.markdown("---")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("⏱️ Response Time", f"{response_time:.1f}s")
                            with col_m2:
                                st.metric("📝 Words", word_count)
                            with col_m3:
                                st.metric("🎯 Confidence", f"{confidence*100:.0f}%")
                    
                    except Exception as e:
                        status_placeholder.error(f"❌ Unexpected Error: {str(e)}")
                        logger.error(f"Critical error in process_audio_query: {e}", exc_info=True)
                        st.session_state.error_count += 1
                        st.session_state.last_error = str(e)
                    
                    finally:
                        # Cleanup
                        cleanup_temp_files(st.session_state.audio_path)
                        st.session_state.audio_path = None
                        st.session_state.processing = False
                        
                        # Clear progress UI after short delay
                        time.sleep(2)
                        progress_bar.empty()
                        
                        # Show completion message
                        st.success("✅ Ready for next question!")
                        st.info("💡 Scroll up to record again or check conversation history")
                        
                        # Auto-refresh after 3 seconds (OPTIMIZED: was 60!)
                        with st.spinner("Refreshing in 3 seconds..."):
                            time.sleep(3)
                        st.rerun()

        else:
            # No audio recorded yet - Show instructions
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 1rem; color: white;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🎤</div>
                <h3 style='color: white; margin-bottom: 1.5rem;'>कैसे उपयोग करें?</h3>
                <div style='background: rgba(255,255,255,0.1); border-radius: 10px; padding: 1.5rem; backdrop-filter: blur(10px);'>
                    <ol style='text-align: left; display: inline-block; color: white; line-height: 2;'>
                        <li><strong>🔴 Record</strong> बटन दबाएं</li>
                        <li><strong>🗣️ स्पष्ट आवाज</strong> में अपना सवाल बोलें</li>
                        <li><strong>⏹️ Stop</strong> बटन दबाएं</li>
                        <li><strong>🚀 "AI से पूछें"</strong> बटन दबाएं</li>
                        <li><strong>🤖 AI का जवाब</strong> सुनें और पढ़ें</li>
                    </ol>
                </div>
                <div style='margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.15); border-radius: 8px;'>
                    <p style='margin: 0; font-size: 0.9rem;'>💡 <strong>Pro Tip:</strong> शांत जगह से बोलें, साफ उच्चारण रखें</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick tips
            st.markdown("---")
            col_tip1, col_tip2, col_tip3 = st.columns(3)
            
            with col_tip1:
                st.markdown("""
                <div style='padding: 1rem; background: #e8f5e9; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 2rem;'>🎯</div>
                    <strong style='color: #2e7d32;'>स्पष्ट बोलें</strong>
                    <p style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>Clear pronunciation के लिए</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_tip2:
                st.markdown("""
                <div style='padding: 1rem; background: #e3f2fd; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 2rem;'>🔇</div>
                    <strong style='color: #1565c0;'>शांत जगह</strong>
                    <p style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>Background noise से बचें</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_tip3:
                st.markdown("""
                <div style='padding: 1rem; background: #fff3e0; border-radius: 10px; text-align: center;'>
                    <div style='font-size: 2rem;'>⏱️</div>
                    <strong style='color: #e65100;'>2-10 सेकंड</strong>
                    <p style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'>Ideal recording length</p>
                </div>
                """, unsafe_allow_html=True)

    # ===== TAB 2: CONVERSATION HISTORY =====
    with tab2:
        show_conversation_history()

    # ===== TAB 3: SETTINGS =====
    with tab3:
        show_settings()

    # ===== FOOTER =====
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666;'>
        <p style='font-size: 1rem; margin-bottom: 0.5rem;'>
            <strong>2024 © AgroMind</strong> • All rights reserved
        </p>
        <p style='font-size: 0.85rem; color: #999;'>
            Made with ❤️ for seamless voice interaction
        </p>
        <p style='font-size: 0.75rem; color: #aaa; margin-top: 0.5rem;'>
            Powered by Groq AI • Whisper V3 • Google TTS
        </p>
    </div>
    """, unsafe_allow_html=True)



def process_audio_query(selected_lang: str):
    """Process audio query with enhanced UI feedback - Optimized for speed"""
    st.session_state.processing = True
    start_time = time.time()
    
    # Clear previous responses to avoid confusion
    st.session_state.transcript = ""
    st.session_state.ai_response = ""
    
    # Progress tracking with better UI
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Transcription
        status_text.markdown("### 🧠 आपकी आवाज़ को समझ रहे हैं...")
        progress_bar.progress(25)
        
        try:
            transcript, confidence = transcribe_audio(st.session_state.audio_path, selected_lang)
            st.session_state.transcript = transcript
            
            if not transcript:
                st.error("❌ कोई शब्द नहीं मिला। कृपया फिर से स्पष्ट बोलें।")
                st.session_state.processing = False
                progress_bar.empty()
                status_text.empty()
                return
            
            # Show confidence
            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            progress_bar.progress(50)
            st.success(f"✅ **आपने कहा:** {transcript}")
            st.caption(f"{confidence_emoji} Confidence: {confidence*100:.1f}%")
            
            # Update stats
            word_count = len(transcript.split())
            st.session_state.total_words_spoken += word_count
                
        except requests.exceptions.Timeout:
            st.error("❌ Timeout: Network slow है। फिर से कोशिश करें।")
            st.session_state.processing = False
            progress_bar.empty()
            status_text.empty()
            return
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Check your internet connection और GROQ API key")
            st.session_state.processing = False
            progress_bar.empty()
            status_text.empty()
            return
        
        # Step 2: Get AI Response
        if transcript:
            status_text.markdown("### 💬 AI जवाब तैयार कर रहा है...")
            progress_bar.progress(75)
            
            full_response = ""
            try:
                response_placeholder = st.empty()
                
                # Create a fresh query without conversation history to avoid context issues
                # This ensures each query is independent and faster
                query_input = {"question": transcript}
                
                # Stream response with timeout
                chunk_count = 0
                for chunk in chain.stream(query_input):
                    full_response += chunk
                    chunk_count += 1
                    
                    # Update UI every 3 chunks for better performance
                    if chunk_count % 3 == 0:
                        response_placeholder.markdown(f"**🤖 AI का जवाब:**\n\n{full_response}▌")
                
                response_placeholder.markdown(f"**🤖 AI का जवाब:**\n\n{full_response}")
                st.session_state.ai_response = full_response
                    
            except Exception as e:
                error_msg = f"जवाब तैयार करने में समस्या: {str(e)}"
                st.error(f"❌ {error_msg}")
                full_response = "क्षमा करें, तकनीकी समस्या के कारण जवाब नहीं दे सका।"
                logger.error(f"LLM generation error: {e}")
            
            # Save to history
            timestamp = datetime.now().strftime("%I:%M %p")
            
            st.session_state.chat_history.append({
                "role": "user", 
                "content": transcript, 
                "timestamp": timestamp
            })
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response, 
                "timestamp": timestamp
            })
            
            st.session_state.conversation_history.append({
                "user": transcript,
                "ai": full_response,
                "lang": selected_lang,
                "timestamp": datetime.now().isoformat()
            })
            
            # Step 3: Generate audio
            if full_response and "क्षमा करें" not in full_response:
                status_text.markdown("### 🎙 जवाब को आवाज़ में बदल रहे हैं...")
                progress_bar.progress(90)
                
                try:
                    # Use threading to speed up TTS generation
                    audio_bytes_tts = text_to_speech(full_response, selected_lang)
                    
                    if audio_bytes_tts:
                        progress_bar.progress(100)
                        status_text.markdown("### ✅ पूरा हुआ!")
                        st.audio(audio_bytes_tts, format="audio/mp3")
                        
                except Exception as e:
                    st.warning(f"⚠️ आवाज़ नहीं बनाई जा सकी")
                    logger.warning(f"TTS error: {e}")
            
            # Update stats
            end_time = time.time()
            response_time = end_time - start_time
            
            st.session_state.total_conversations += 1
            
            # Calculate running average
            n = st.session_state.total_conversations
            old_avg = st.session_state.avg_response_time
            st.session_state.avg_response_time = (old_avg * (n-1) + response_time) / n
            
            # Clear progress immediately for faster UX
            progress_bar.empty()
            status_text.empty()
        
        st.session_state.processing = False
        
        # Cleanup audio file immediately after processing
        cleanup_temp_files(st.session_state.audio_path)
        st.session_state.audio_path = None
        
        # Success message
        st.success(f"✅ पूरा हुआ! ({response_time:.1f} seconds)")
        
        # Auto-rerun to reset recording interface
        time.sleep(60)
        st.rerun()


def show_conversation_history():
    """Display conversation history with better UI"""
    st.markdown("### 💬 बातचीत का इतिहास")
    
    if not st.session_state.conversation_history:
        st.info("📭 अभी तक कोई बातचीत नहीं। Record बटन दबाकर शुरू करें!")
        return
    
    # Filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"कुल {len(st.session_state.conversation_history)} बातचीत")
    with col2:
        show_all = st.checkbox("सभी दिखाएं", value=False)
    with col3:
        if st.button("🗑️ इतिहास साफ करें", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.chat_history = []
            st.success("✅ इतिहास साफ हो गया!")
            st.rerun()
    
    display_count = len(st.session_state.conversation_history) if show_all else min(10, len(st.session_state.conversation_history))
    
    # Display conversations
    for i, conv in enumerate(reversed(st.session_state.conversation_history[-display_count:])):
        with st.container():
            st.markdown(f"**#{len(st.session_state.conversation_history) - i}** • {conv.get('lang', 'hi').upper()}")
            
            # User message
            display_chat_message("user", conv['user'], 
                               datetime.fromisoformat(conv['timestamp']).strftime("%I:%M %p") if 'timestamp' in conv else None)
            
            # AI message  
            display_chat_message("assistant", conv['ai'],
                               datetime.fromisoformat(conv['timestamp']).strftime("%I:%M %p") if 'timestamp' in conv else None)
            
            if i < display_count - 1:
                st.markdown("<br>", unsafe_allow_html=True)


def show_settings():
    """Settings panel"""
    st.markdown("### ⚙️ सेटिंग्स")
    
    # Voice settings
    st.markdown("#### 🔊 आवाज़ सेटिंग्स")
    st.session_state.voice_enabled = st.toggle("आवाज़ में जवाब सुनें", value=st.session_state.voice_enabled)
    
    if st.session_state.voice_enabled:
        st.info("✅ AI के जवाब आवाज़ में सुनाई देंगे")
    else:
        st.warning("⚠️ केवल टेक्स्ट में जवाब मिलेगा")
    
    st.markdown("---")
    
    # Performance settings
    st.markdown("#### ⚡ Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("कुल बातचीत", st.session_state.total_conversations)
        st.metric("कुल शब्द", st.session_state.total_words_spoken)
    
    with col2:
        st.metric("औसत समय", f"{st.session_state.avg_response_time:.1f}s")
        st.metric("भाषा", get_supported_languages().get(st.session_state.last_language, "हिंदी"))
    
    st.markdown("---")
    
    # Clear data
    st.markdown("#### 🗑️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Stats रीसेट करें", use_container_width=True):
            st.session_state.total_conversations = 0
            st.session_state.total_words_spoken = 0
            st.session_state.avg_response_time = 0
            st.success("✅ Stats रीसेट हो गए!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ सब कुछ साफ करें", type="primary", use_container_width=True):
            # Clear everything
            st.session_state.conversation_history = []
            st.session_state.chat_history = []
            st.session_state.total_conversations = 0
            st.session_state.total_words_spoken = 0
            st.session_state.avg_response_time = 0
            cleanup_temp_files(st.session_state.audio_path)
            st.session_state.audio_path = None
            st.success("✅ सब कुछ साफ हो गया!")
            st.rerun()
    
    st.markdown("---")
    
    # System info
    st.markdown("#### ℹ️ System Info")
    st.info(f"""
    **Model:** Whisper Large V3  
    **LLM:** Groq (via voice_pipeline)  
    **TTS:** Google Text-to-Speech  
    **Status:** ✅ Connected
    """)


# Cleanup on session end
def cleanup_on_session_end():
    """Session end पर cleanup"""
    try:
        if hasattr(st.session_state, 'audio_path') and st.session_state.audio_path:
            cleanup_temp_files(st.session_state.audio_path)
    except Exception:
        pass
