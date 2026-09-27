"""
Online Voice-Enabled Chatbot Using Speech Recognition and Deep Learning
Streamlit Web Application
"""

import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components

# Import predictor & audio utility
from predict import get_predictor
from utils.chatbot_utils import transcribe_audio_file

# ----------------- PAGE CONFIGURATION ----------------- #
st.set_page_config(
    page_title="VoiceBot AI | Speech & Deep Learning Chatbot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM MODERN STYLING ----------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-title-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .subtitle {
        color: #C7D2FE;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(165, 180, 252, 0.3);
        margin-right: 8px;
    }

    .glass-card {
        background: #1E293B;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    .speech-box {
        background: #0F172A;
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 8px;
        color: #E2E8F0;
        font-size: 1.05rem;
        margin: 10px 0;
    }

    .response-box {
        background: #0F172A;
        border-left: 4px solid #10B981;
        padding: 16px 18px;
        border-radius: 8px;
        color: #F8FAFC;
        font-size: 1.1rem;
        line-height: 1.5;
        margin: 10px 0;
    }

    .intent-pill {
        display: inline-block;
        padding: 5px 14px;
        background: #312E81;
        color: #A5B4FC;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
        border: 1px solid #4338CA;
    }

    .confidence-badge {
        display: inline-block;
        padding: 5px 12px;
        background: #064E3B;
        color: #6EE7B7;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #059669;
    }

    .user-bubble {
        background: #334155;
        color: #F8FAFC;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 10px;
        max-width: 80%;
        margin-left: auto;
    }

    .bot-bubble {
        background: #1E293B;
        border: 1px solid #3b4252;
        color: #F1F5F9;
        padding: 12px 18px;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 10px;
        max-width: 85%;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE ----------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_intent" not in st.session_state:
    st.session_state.last_intent = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_top_predictions" not in st.session_state:
    st.session_state.last_top_predictions = []


# ----------------- MODEL LOADER ----------------- #
@st.cache_resource(show_spinner="Loading Deep Learning Bi-LSTM Model...")
def load_cached_predictor():
    return get_predictor()

try:
    predictor = load_cached_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ----------------- HEADER SECTION ----------------- #
st.markdown("""
<div class="main-title-container">
    <h1 class="main-title">🎙️ Online Voice-Enabled AI Chatbot</h1>
    <p class="subtitle">Speech Recognition + Deep Learning (Bidirectional LSTM) Intent Classification Engine</p>
    <div style="margin-top: 12px;">
        <span class="stat-badge">TensorFlow / Keras</span>
        <span class="stat-badge">Bi-LSTM Neural Network</span>
        <span class="stat-badge">Web Speech API</span>
        <span class="stat-badge">Client & Cloud Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- SIDEBAR ----------------- #
with st.sidebar:
    st.header("⚙️ System Control & Architecture")
    
    st.markdown("### 🧠 Deep Learning Architecture")
    st.markdown("""
    - **Input Layer:** Sequence Length = 16
    - **Embedding:** 64-dimensional vector space
    - **Encoder:** Bidirectional LSTM (48 units)
    - **Feature Pooling:** Global MaxPooling 1D
    - **Dense Layer:** 64 units + ReLU
    - **Regularization:** Dropout (0.3)
    - **Output Layer:** 18-class Softmax
    """)

    st.markdown("---")
    st.markdown("### 📊 Trained Model Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Train Accuracy", value="100.0%")
    with col_m2:
        st.metric(label="Val Accuracy", value="63.4%")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Queries below this threshold trigger conversational fallback."
    )

    st.markdown("---")
    st.markdown("### 📈 Evaluation Visualizations")
    show_curves = st.checkbox("Show Loss & Accuracy Curves", value=False)
    show_cm = st.checkbox("Show Confusion Matrix", value=False)

    if show_curves:
        curves_path = os.path.join(os.path.dirname(__file__), "assets", "screenshots", "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training & Validation Curves", use_container_width=True)

    if show_cm:
        cm_path = os.path.join(os.path.dirname(__file__), "assets", "screenshots", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query = ""
        st.session_state.last_intent = ""
        st.session_state.last_confidence = 0.0
        st.session_state.last_response = ""
        st.session_state.last_top_predictions = []
        st.rerun()


# ----------------- MAIN INTERFACE ----------------- #
if not model_loaded:
    st.error(f"❌ Model could not be loaded: {model_error}")
    st.info("💡 Please execute `python train.py` in your terminal to train and generate the model artifacts.")
    st.stop()

# Layout: Two columns (Left: Interaction & Results, Right: Conversation History & Inspector)
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown("### 🎤 Step 1: Voice Input (Speak into Microphone)")
    st.caption("Click the button below and speak your question. The browser converts speech to text in real-time.")

    # Interactive HTML5 Web Speech API Component
    speech_html = """
    <div style="background: #1E293B; border-radius: 12px; padding: 18px; border: 1px solid #334155; text-align: center; font-family: sans-serif;">
        <button id="micBtn" onclick="toggleListening()" style="
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 30px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        ">
            <span id="micIcon" style="font-size: 20px;">🎙️</span>
            <span id="micText">Start Voice Input</span>
        </button>
        <div id="statusMsg" style="margin-top: 12px; font-size: 14px; color: #94A3B8; font-weight: 500;">
            Ready to listen. Click microphone to speak.
        </div>
        <div id="transcriptBox" style="
            margin-top: 14px;
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            min-height: 48px;
            color: #F8FAFC;
            font-size: 14px;
            text-align: left;
            word-wrap: break-word;
        ">
            <span style="color: #64748B; font-style: italic;">Recognized speech will appear here...</span>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: center; gap: 10px;">
            <button id="copyBtn" onclick="copyTranscript()" style="
                background: #334155;
                color: #CBD5E1;
                border: 1px solid #475569;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 13px;
                cursor: pointer;
            ">📋 Copy to Text Box</button>
            <button id="speakAnswerBtn" onclick="speakLastResponse()" style="
                background: #065F46;
                color: #6EE7B7;
                border: 1px solid #059669;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 13px;
                cursor: pointer;
            ">🔊 Read Answer Aloud</button>
        </div>
    </div>

    <script>
        let recognition = null;
        let isListening = false;
        let finalTranscript = "";

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isListening = true;
                document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)';
                document.getElementById('micText').innerText = 'Listening... Speak now';
                document.getElementById('statusMsg').innerText = '🔴 Recording audio from microphone...';
                document.getElementById('statusMsg').style.color = '#F87171';
            };

            recognition.onresult = function(event) {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                const currentText = finalTranscript || interim;
                document.getElementById('transcriptBox').innerHTML = '<strong>Speech:</strong> ' + currentText;
            };

            recognition.onerror = function(event) {
                console.error("Speech Recognition Error:", event.error);
                let errText = "Speech recognition error: " + event.error;
                if (event.error === 'not-allowed') {
                    errText = "⚠️ Microphone access was blocked. Please permit microphone access in your browser.";
                }
                document.getElementById('statusMsg').innerText = errText;
                document.getElementById('statusMsg').style.color = '#F87171';
                stopListening();
            };

            recognition.onend = function() {
                stopListening();
                document.getElementById('statusMsg').innerText = '✅ Speech captured! Use text box below or click Copy.';
                document.getElementById('statusMsg').style.color = '#34D399';
            };
        } else {
            document.getElementById('statusMsg').innerText = '⚠️ Web Speech API not supported in this browser. Please use Chrome/Edge or text fallback.';
            document.getElementById('statusMsg').style.color = '#FBBF24';
        }

        function toggleListening() {
            if (!recognition) return;
            if (isListening) {
                recognition.stop();
            } else {
                finalTranscript = "";
                try {
                    recognition.start();
                } catch(e) {
                    console.log(e);
                }
            }
        }

        function stopListening() {
            isListening = false;
            document.getElementById('micBtn').style.background = 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)';
            document.getElementById('micText').innerText = 'Start Voice Input';
        }

        function copyTranscript() {
            const rawText = document.getElementById('transcriptBox').innerText.replace('Speech: ', '');
            navigator.clipboard.writeText(rawText).then(() => {
                alert('Copied recognized speech: "' + rawText + '" to clipboard! Paste it into the text box below.');
            });
        }

        function speakLastResponse() {
            const botBox = parent.document.querySelector('.response-box');
            if (botBox && 'speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(botBox.innerText);
                utterance.pitch = 1.0;
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            } else {
                alert('Response text not ready or speech synthesis unavailable.');
            }
        }
    </script>
    """
    components.html(speech_html, height=210)

    # Audio File Upload Fallback
    with st.expander("📁 Or Upload Audio File (.wav)", expanded=False):
        uploaded_audio = st.file_uploader("Upload audio recording", type=["wav"], key="audio_uploader")
        if uploaded_audio is not None:
            if st.button("Transcribe Uploaded Audio", key="btn_transcribe_audio"):
                with st.spinner("Processing speech using SpeechRecognition..."):
                    transcription = transcribe_audio_file(uploaded_audio)
                    if transcription["success"]:
                        st.success(f"Recognized Speech: {transcription['text']}")
                        st.session_state.pending_query = transcription["text"]
                    else:
                        st.error(transcription["error"])

    st.markdown("### 💬 Step 2: Text Input Fallback")
    st.caption("You can also type your message directly. Both voice and text use the identical Deep Learning classifier.")

    # Pre-fill query if pending from audio
    default_input_val = st.session_state.get("pending_query", "")
    if default_input_val:
        st.session_state.pending_query = ""

    user_query = st.text_input(
        "Enter or verify query:",
        value=default_input_val,
        placeholder="e.g. What are your working hours? Or click voice above...",
        key="main_text_input"
    )

    # Prompt Suggestions
    st.markdown("<span style='font-size:0.85rem; color:#94A3B8;'>Quick queries to test:</span>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🕒 Working hours", use_container_width=True):
            user_query = "What are your working hours?"
    with q_col2:
        if st.button("📍 Campus location", use_container_width=True):
            user_query = "Where are you located?"
    with q_col3:
        if st.button("🎭 Tell a joke", use_container_width=True):
            user_query = "Tell me a joke"

    send_btn = st.button("🚀 Process & Generate Response", type="primary", use_container_width=True)

    if send_btn and user_query:
        with st.spinner("Running Deep Learning Intent Classifier..."):
            result = predictor.process_query(user_query, threshold=confidence_threshold)

            # Store in session state
            st.session_state.last_query = result["query"]
            st.session_state.last_intent = result["intent"]
            st.session_state.last_confidence = result["confidence"]
            st.session_state.last_response = result["response"]
            st.session_state.last_top_predictions = result["top_predictions"]

            # Append to conversation history
            st.session_state.messages.append({
                "query": result["query"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "response": result["response"]
            })

    # Display Current Processing Result
    if st.session_state.last_query:
        st.markdown("---")
        st.markdown("### 🎯 Live Inference Output")
        
        # 1. Recognized Speech / Input
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#94A3B8; font-weight:600;">1. RECOGNIZED SPEECH / INPUT:</div>
        <div class="speech-box">"{st.session_state.last_query}"</div>
        """, unsafe_allow_html=True)

        # 2. Predicted Intent & Confidence
        conf_pct = st.session_state.last_confidence * 100
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; margin: 12px 0;">
            <div>
                <span style="font-size:0.85rem; color:#94A3B8; font-weight:600;">2. PREDICTED INTENT:</span><br>
                <span class="intent-pill">🏷️ {st.session_state.last_intent}</span>
            </div>
            <div>
                <span style="font-size:0.85rem; color:#94A3B8; font-weight:600;">3. CONFIDENCE SCORE:</span><br>
                <span class="confidence-badge">⚡ {conf_pct:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(float(st.session_state.last_confidence), 0.0), 1.0))

        # 3. Chatbot Response
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#94A3B8; font-weight:600; margin-top:14px;">4. CHATBOT RESPONSE:</div>
        <div class="response-box">🤖 {st.session_state.last_response}</div>
        """, unsafe_allow_html=True)


with col_right:
    st.markdown("### 💬 Conversation History")
    if not st.session_state.messages:
        st.info("No messages yet. Speak through the microphone or type a query to begin.")
    else:
        history_container = st.container(height=360)
        with history_container:
            for msg in st.session_state.messages:
                st.markdown(f"""
                <div class="user-bubble">
                    <span style="font-size:0.75rem; opacity:0.7;">You</span><br>
                    {msg['query']}
                </div>
                <div class="bot-bubble">
                    <span style="font-size:0.75rem; color:#818CF8; font-weight:600;">VoiceBot (Intent: {msg['intent']} | {msg['confidence']*100:.0f}%)</span><br>
                    {msg['response']}
                </div>
                """, unsafe_allow_html=True)

    # Deep Learning Model Probability Inspector
    st.markdown("### 🔬 Model Diagnostic Inspector")
    with st.expander("Show Top 3 Intent Probabilities (Deep Learning Softmax)", expanded=True):
        if st.session_state.last_top_predictions:
            for item in st.session_state.last_top_predictions:
                c_val = item["confidence"]
                c_col1, c_col2 = st.columns([1, 2])
                with c_col1:
                    st.write(f"**`{item['intent']}`**")
                with c_col2:
                    st.progress(c_val)
                    st.caption(f"{c_val*100:.2f}%")
        else:
            st.caption("Submit a query to inspect softmax layer output probabilities.")


# ----------------- FOOTER ----------------- #
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
    🎓 <strong>Academic Demonstration Project</strong> | Online Voice-Enabled Chatbot Using Speech Recognition and Deep Learning<br>
    Built with Python, Streamlit, TensorFlow/Keras, and Web Speech API
</div>
""", unsafe_allow_html=True)
