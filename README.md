# 🌾AI कृषि सहायक - AI-Powered Agricultural Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP-orange.svg)]()

> **AI आधारित फसल सलाह सहायक** - Your Digital Farming Consultant with Voice Support

AgroMind is an intelligent agricultural assistant that provides farmers with personalized crop recommendations based on real-time weather data, soil conditions, and machine learning predictions. The app features bilingual support (Hindi/English) and includes voice interaction capabilities.

---

## 🌟 Key Features

### 🎯 Core Functionality
- **Smart Crop Recommendation** - ML-based predictions using weather and soil data
- **Real-time Weather Integration** - Live weather data from WeatherAPI
- **Soil Analysis** - Integrated SoilGrids API for soil composition data
- **Location-Aware** - GPS/IP-based location detection for localized advice

### 🎤 Voice Capabilities
- **Voice Input** - Ask questions using audio recording or file upload
- **Voice Output** - Text-to-Speech responses in Hindi
- **Conversational AI** - Natural language processing for farmer queries
- **ElevenLabs Integration** - Advanced voice agent support

### 🤖 AI-Powered Intelligence
- **LLM Integration** - Powered by Groq AI (Llama models)
- **Context-Aware Responses** - Considers location, weather, and soil data
- **Chat History** - Maintains conversation context
- **Export Functionality** - Download chat history as JSON

### 📊 Data Visualization
- Comprehensive weather dashboard
- Soil composition metrics
- Confidence scores for predictions
- Real-time statistics

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Valid API keys for:
  - WeatherAPI (weather data)
  - Groq AI (LLM responses)
  - OpenAI (optional, for STT)
  - ElevenLabs (optional, for advanced voice)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/katcoders/agromind.git
cd agromind
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:

```env
WEATHER_API_KEY=your_weatherapi_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional
ELEVENLABS_API_KEY=your_elevenlabs_key  # Optional
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📦 Dependencies

### Core Libraries
- **streamlit** - Web application framework
- **openai** - OpenAI API integration
- **langchain-groq** - LLM integration
- **scikit-learn** - Machine learning models
- **pandas** & **numpy** - Data processing

### Voice Processing
- **gtts** - Google Text-to-Speech
- **st-audiorec** - Audio recording widget
- **voiceassit** - Custom voice assistant module

### API Integration
- **requests** - HTTP requests for weather/soil data
- **streamlit-geolocation** - GPS location access
- **python-dotenv** - Environment variable management

### Full Requirements
See `requirements.txt` for complete list of dependencies.

---

## 🏗️ Project Structure

```
agromind/
├── app.py                 # Main application file
├── voice_pipeline.py      # Voice processing pipeline
├── voiceassit.py         # Voice assistant utilities
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md            # This file
```

---

## 🎯 How It Works

### 1. Location Detection
- User grants location permission (GPS or IP-based)
- City and coordinates are stored for contextual advice

### 2. Data Collection
- **Weather Data**: Fetched from WeatherAPI based on coordinates
- **Soil Data**: Retrieved from SoilGrids API for soil composition
- **ML Model**: Random Forest classifier trained on agricultural data

### 3. Crop Prediction
The ML model considers:
- Temperature (°C)
- Humidity (%)
- Soil pH
- Nitrogen content

Predicts optimal crops: 🌾 Wheat, 🌱 Rice, or 🌽 Corn

### 4. AI Consultation
- Users ask questions via text or voice
- LLM (Groq/Llama) generates contextual responses
- Responses consider location, weather, soil, and crop predictions
- Optional TTS converts responses to Hindi audio

---

## 💡 Usage Examples

### Text Queries
```
✍️ "इस मौसम में कौन सी फसल बेहतर होगी?"
✍️ "मिट्टी की गुणवत्ता कैसे सुधारें?"
✍️ "बारिश के बाद क्या करना चाहिए?"
```

### Voice Queries
1. Click the 🎤 Record button
2. Speak your question in Hindi
3. Stop recording
4. Get AI-generated voice response

### Topics Covered
- 🌾 Crop selection advice
- 🌱 Soil improvement techniques
- 🌧️ Weather-based farming guidance
- 🐛 Pest and disease control
- 💧 Irrigation management
- 🌿 Organic farming methods

---

## 🔧 Configuration

### API Keys Setup

#### WeatherAPI (Required)
1. Sign up at [weatherapi.com](https://www.weatherapi.com/)
2. Get free API key (1M calls/month)
3. Add to `.env`: `WEATHER_API_KEY=your_key`

#### Groq AI (Required)
1. Sign up at [groq.com](https://groq.com/)
2. Generate API key
3. Add to `.env`: `GROQ_API_KEY=your_key`

#### OpenAI (Optional)
- For advanced STT capabilities
- Add to `.env`: `OPENAI_API_KEY=your_key`

### Customization

**Change LLM Model:**
```python
# In app.py, modify:
llm = ChatGroq(
    model="llama3-70b-8192",  # Change model here
    temperature=0.7
)
```

**Adjust Crop Database:**
```python
# In app.py, modify crop_map:
crop_map = {
    0: "🌾 गेहूँ",
    1: "🌱 धान", 
    2: "🌽 मक्का"
    # Add more crops
}
```

---

## 🌍 Supported Languages

- **Primary**: Hindi (हिंदी)
- **Secondary**: English
- **Voice Support**: Hindi TTS/STT

---

## 📊 Features Roadmap

### ✅ Current Features (MVP)
- [x] Location-based recommendations
- [x] Weather and soil data integration
- [x] ML crop prediction
- [x] Voice input/output
- [x] Chat history
- [x] Export functionality

### 🔜 Upcoming Features
- [ ] Market price integration
- [ ] Multi-crop comparison
- [ ] Historical weather trends
- [ ] Disease detection from images
- [ ] Offline mode
- [ ] SMS/WhatsApp alerts
- [ ] Regional language support

---

## 🐛 Troubleshooting

### Common Issues

**Voice not recognized:**
- Ensure quiet environment
- Speak clearly near microphone
- Check internet connection

**Location not detected:**
- Allow browser location permission
- Use IP-based fallback option
- Check GPS is enabled on device

**API errors:**
- Verify API keys in `.env`
- Check API rate limits
- Ensure internet connectivity

**Slow performance:**
- Check internet speed
- Reduce response length in settings
- Clear browser cache

---

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest tests/

# Format code
black app.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Credits

**Developed by:** AgroMind Team

**Powered by:**
- 🤖 Groq AI (LLM)
- 🌍 SoilGrids (Soil Data)
- ☁️ WeatherAPI (Weather Data)
- 🗣️ ElevenLabs (Voice Agent)
- 🐍 Streamlit (Web Framework)

---

## ⚠️ Disclaimer

This is an **MVP (Minimum Viable Product)** intended for demonstration and educational purposes. All agricultural advice provided by AgroMind is informational only. For critical farming decisions, please consult local agricultural experts and extension services.

---

## 🙏 Acknowledgments

Special thanks to:
- Indian farmers for inspiration
- Open source community
- API providers (Groq, WeatherAPI, SoilGrids)
- Streamlit team for excellent framework

---

<div align="center">

**Made with ❤️ for Indian Farmers**

🌾 **AgroMind** - आपके खेत का डिजिटल मित्र

[Website](https://agromind.app) • [Documentation](https://docs.agromind.app) • [Demo](https://demo.agromind.app)

</div>
