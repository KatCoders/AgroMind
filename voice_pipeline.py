import os
import io
import time
import tempfile
import requests
import streamlit as st
import logging
from typing import Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS

# Langchain / Groq imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------- Logging Setup -------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- Load Environment Variables -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Validation
if not GROQ_API_KEY:
    st.error("❌ .env फ़ाइल में `GROQ_API_KEY` सेट करें")
    st.info("💡 Groq API key: https://console.groq.com")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ------------------- Unified TTS System -------------------
class UnifiedTTSSystem:
    """Unified TTS with OpenAI primary and gTTS fallback"""
    
    def __init__(self):
        self.openai_available = openai_client is not None
        self.cache = {}
        
    def generate_audio(self, text: str, use_cache: bool = True) -> Optional[bytes]:
        """Generate audio with fallback"""
        if not text or not text.strip():
            return None
        
        # Truncate long text
        if len(text) > 500:
            text = self._truncate_intelligently(text, 500)
        
        # Check cache
        text_hash = hash(text[:500])
        if use_cache and text_hash in self.cache:
            logger.info("Using cached audio")
            return self.cache[text_hash]
        
        # Try OpenAI first
        if self.openai_available:
            audio_bytes = self._openai_tts(text)
            if audio_bytes:
                if use_cache and len(self.cache) < 20:
                    self.cache[text_hash] = audio_bytes
                return audio_bytes
            logger.warning("OpenAI TTS failed, using gTTS")
        
        # Fallback to gTTS
        audio_bytes = self._gtts_tts(text)
        if audio_bytes and use_cache and len(self.cache) < 20:
            self.cache[text_hash] = audio_bytes
        return audio_bytes
    
    def _openai_tts(self, text: str) -> Optional[bytes]:
        """OpenAI TTS implementation"""
        try:
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text[:4096]
            )
            return response.read()
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None
    
    def _gtts_tts(self, text: str) -> Optional[bytes]:
        """gTTS fallback"""
        try:
            tts = gTTS(text=text, lang="hi", slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.getvalue()
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return None
    
    def _truncate_intelligently(self, text: str, max_length: int) -> str:
        """Truncate at sentence boundaries"""
        if len(text) <= max_length:
            return text
        
        sentences = text.split('।')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + "।") <= max_length:
                truncated += sentence + "।"
            else:
                break
        
        return truncated if truncated else text[:max_length] + "..."

# ------------------- Speech-to-Text -------------------
class SpeechToText:
    """STT using Groq Whisper API"""
    
    @staticmethod
    def transcribe(file_path: str, language: str = "hi") -> str:
        """Transcribe audio file"""
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return ""
        
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            
            with open(file_path, "rb") as audio_file:
                files = {"file": (os.path.basename(file_path), audio_file, "audio/wav")}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "language": "hi",
                    "response_format": "text"
                }
                response = requests.post(url, headers=headers, data=data, files=files, timeout=45)
            
            response.raise_for_status()
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""



# ------------------- Session State -------------------
def init_session_state():
    """Initialize session state"""
    if "initialized" not in st.session_state:
        st.session_state.update({
            "initialized": True,
            "chat_history": [],
            "processing": False,
            "last_audio": None,
            "tts_system": UnifiedTTSSystem(),
            "stt": SpeechToText(),
            # Mock data for demo
            "city": "इंदौर",
            "weather_data": {"temperature": 28, "humidity": 65},
            "soil_data": {"ph": 6.5, "nitrogen": 45},
            "predicted_crop": "गेहूं",
            "confidence": 85.5
        })

init_session_state()


# ------------------- LLM Setup -------------------
try:
    MODEL_NAME = "llama-3.3-70b-versatile"
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.7,
        streaming=True,
        max_tokens=1024
    )

    # Enhanced prompt template with context
    template_text = """
आप एक अनुभवी और दोस्ताना किसान मित्र हैं जो कृषि सलाहकार का काम करते हैं। 
Agar koi aap se puche apko kisne banaya, to kaho "AgroMind team ne, jo aapke kisan bhaiyon ke liye best AI assistant banane mein laga hai".

आपकी विशेषताएं:
- हमेशा सरल, समझने योग्य हिंदी में बात करना
- स्थानीय परिस्थितियों (मौसम, मिट्टी) के अनुसार व्यावहारिक सलाह देना  
- "भाई", "जी", "आइए" जैसे दोस्ताना शब्दों का उपयोग करना
- छोटे, actionable steps में जवाब देना


वर्तमान स्थानीय डेटा:
- स्थान: {location}
- तापमान: {temperature}°C, आर्द्रता: {humidity}%
- मिट्टी pH: {soil_ph}, नाइट्रोजन: {nitrogen}
- AI सुझाव: {crop_suggestion} (विश्वास: {confidence:.1f}%)
आपके जवाब देने के लिए इन डेटा का उपयोग करें।




नियम:
1. यदि मार्केट रेट/मंडी भाव पूछें तो कहें: "यह सुविधा अभी विकास में है, जल्द ही उपलब्ध होगी"
2. फसल सुझाव के लिए ऊपर दिए गए स्थानीय डेटा का उपयोग करें
3. हमेशा प्रैक्टिकल और लागू करने योग्य सलाह दें
4. अगर कोई चिकित्सा सलाह पूछे तो डॉक्टर से मिलने को कहें
Aur jis salwal ka jawab aapko nahi pata, usme aap seedha "मुझे खेद है, मैं इस बारे में जानकारी नहीं दे सकता। कृपया विशेषज्ञ से संपर्क करें।" keh dena.

उपयोगकर्ता का सवाल: {question}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template_text),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    
except Exception as e:
    st.error(f"❌ LLM मॉडल लोड करने में समस्या: {e}")
    st.info("कृपया अपनी इंटरनेट कनेक्शन और GROQ_API_KEY की जांच करें")
    st.stop()
# ------------------- LLM Response Function -------------------
def get_llm_response(user_question: str) -> str:
    """Generate LLM response"""
    if not user_question.strip():
        return "कृपया सवाल लिखें।"
    
    try:
        full_response = ""
        response_placeholder = st.empty()
        
        for chunk in chain.stream({
            "question": user_question,
            "location": st.session_state.city,
            "temperature": st.session_state.weather_data['temperature'],
            "humidity": st.session_state.weather_data['humidity'],
            "soil_ph": st.session_state.soil_data['ph'],
            "nitrogen": st.session_state.soil_data['nitrogen'],
            "crop_suggestion": st.session_state.predicted_crop,
            "confidence": st.session_state.confidence
        }):
            full_response += chunk
            response_placeholder.markdown(f"**🤖 जवाब:** {full_response}▌")
        
        response_placeholder.markdown(f"**🤖 जवाब:** {full_response}")
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return full_response
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "क्षमा करें, जवाब नहीं दे पा रहा हूँ। कृपया फिर से प्रयास करें।"

# ------------------- Voice Input Section -------------------

