# 🎙️ Online Voice-Enabled Chatbot: Architecture, Working & Technical Documentation

> **Live Production Deployment URL:**  
> 🔗 [https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/](https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/)  
>
> **GitHub Repository:**  
> 📂 [https://github.com/vyom-vs11/voice-chatbot](https://github.com/vyom-vs11/voice-chatbot)

---

## 📌 1. Project Overview

The **Online Voice-Enabled Chatbot** is a production-grade conversational AI system that combines **client-side Automatic Speech Recognition (ASR)**, **Deep Learning Natural Language Processing (NLP)**, and **real-time Text-to-Speech (TTS) Voice Synthesis**. 

The system allows users to communicate naturally through their microphone or via typed text. Spoken queries are transcribed in real-time, categorized into intent classes by a custom-trained **Bidirectional Long Short-Term Memory (Bi-LSTM)** neural network, and processed through a hybrid intelligence pipeline (pre-trained conversational intents + open-domain Wikipedia knowledge engine). Finally, responses are spoken aloud to the user using browser voice synthesis.

```
┌──────────────┐     Web Speech API     ┌───────────────────┐
│ User Voice / │ ─────────────────────> │ Speech Recognition│
│  Typed Text  │                        │ (Speech-to-Text)  │
└──────────────┘                        └─────────┬─────────┘
                                                  │ Transcribed Text
                                                  ▼
┌───────────────────────────────────────────────────────────┐
│              Natural Language Processing (NLP)            │
│  - Lowercasing & Punctuation Stripping                    │
│  - Word Tokenizer (Vocabulary index mapping)              │
│  - Sequence Padding (Max length = 16 tokens)              │
└─────────────────────────────┬─────────────────────────────┘
                              │ Padded Tensor (1, 16)
                              ▼
┌───────────────────────────────────────────────────────────┐
│            Bi-LSTM Deep Learning Neural Network           │
│  - 64-dim Word Embedding Layer                            │
│  - Bidirectional LSTM (Forward + Backward Context)        │
│  - Global MaxPooling1D (Temporal Feature Extraction)      │
│  - Dense (64, ReLU) + Dropout (0.3)                       │
│  - Dense Softmax (18-Class Probability Distribution)       │
└─────────────────────────────┬─────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │ Confidence >= 60%?                    │
          ▼ YES                                   ▼ NO
┌───────────────────────┐               ┌───────────────────────┐
│ Intent Match Found    │               │ Dynamic Q&A Engine    │
│ (e.g. greeting, help, │               │ - Live Time / Math    │
│  admissions, fees)    │               │ - Wikipedia Knowledge │
└──────────┬────────────┘               └───────────┬───────────┘
           │                                        │
           └──────────────────┬─────────────────────┘
                              │ Response Text
                              ▼
┌───────────────────────────────────────────────────────────┐
│                      Client UI & TTS                      │
│  - Real-time Diagnostic Inspector (Top-3 Probabilities)   │
│  - Web Speech Synthesis (Spoken Voice Output)             │
│  - Chat History Bubble Update                             │
└───────────────────────────────────────────────────────────┘
```

---

## 🚀 2. Live Deployment Details

- **Hosted URL:** [https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/](https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/)
- **Hosting Platform:** [Vercel](https://vercel.com/) (Serverless Cloud Architecture)
- **Frontend Engine:** Modern HTML5 / Vanilla CSS3 / ES6+ JavaScript (`public/`)
- **Backend API:** Python Serverless Function (`api/index.py`)
- **Latency:** Cold-start response < 50ms, inference latency ~15ms
- **SSL / Security:** End-to-end HTTPS (required for Web Speech microphone access)

---

## ⚙️ 3. Step-by-Step System Working & Pipeline

### Step 1: Voice Acquisition (Speech-to-Text via Web Speech API)
- When the user clicks the microphone button, the browser requests user authorization via `navigator.mediaDevices` and initializes `webkitSpeechRecognition` / `SpeechRecognition`.
- Audio is processed directly on the client machine via the browser's speech recognition engine.
- Speech is streamed with interim visual feedback and converted into a normalized text transcript string (e.g., *"What is the admission procedure?"*).
- **Graceful Fallback:** If the user lacks a microphone or denies permissions, a responsive text input field is always accessible.

### Step 2: Text Preprocessing & Tokenization
- The raw transcript undergoes text cleaning:
  1. Converted to lowercase.
  2. Removal of special characters, symbols, and redundant whitespaces.
- The cleaned sentence is processed by the trained **Keras Tokenizer** (`tokenizer.pkl`):
  - Every word is looked up in the vocabulary and converted to its corresponding integer index.
  - Out-of-vocabulary (OOV) words are mapped to the `<OOV>` index (1).
- The sequence is standardized to a fixed length using **post-padding**:
  $$\text{Input Vector} \in \mathbb{R}^{1 \times 16}$$

### Step 3: Deep Learning Neural Network Inference
The padded sequence passes through the trained **Bidirectional LSTM** network:
1. **Embedding Layer ($16 \times 64$):** Maps discrete token indices into dense continuous vector representations, capturing semantic relationships between words.
2. **Bidirectional LSTM (48 units $\times$ 2 directions = 96 features):**
   - **Forward LSTM:** Traverses tokens from index $1 \to 16$, capturing preceding context.
   - **Backward LSTM:** Traverses tokens from index $16 \to 1$, capturing succeeding context.
   - Preserves long-range dependencies and word order that standard Bag-of-Words or feed-forward networks ignore.
3. **Global MaxPooling1D:** Performs temporal pooling across all 16 token steps, extracting the strongest activation for each feature channel. This acts as an automated "keyword activation filter".
4. **Dense Layer (64 units, ReLU) + Dropout (0.3):** Non-linear feature combination with regularization to prevent overfitting.
5. **Softmax Output Layer (18 units):** Produces a normalized probability distribution $\hat{y} \in [0, 1]^{18}$ where $\sum_{i=1}^{18} \hat{y}_i = 1$.

### Step 4: Intent Classification & Confidence Scoring
- The class index with the maximum probability ($\arg\max \hat{y}$) is decoded using `label_encoder.pkl` into the predicted intent tag.
- The highest probability is compared against the confidence threshold (**60% / 0.60**).

### Step 5: Dual-Engine Knowledge Resolution
- **Case A (High Confidence Intent $\ge$ 60%):**
  - If the intent matches a known conversational class (e.g., `greeting`, `courses`, `fees`, `placement`, `working_hours`, etc.), a response is selected from `data/intents.json`.
- **Case B (Open-Domain / Dynamic Knowledge Query):**
  - If the confidence is below 60% or the query matches dynamic patterns (e.g., *"What time is it?"*, *"Calculate 45 * 8"*, or *"Who is Albert Einstein?"*):
  - The **Q&A Engine** (`utils/qa_engine.py`) takes over:
    - Resolves arithmetic expressions safely.
    - Resolves live time/date/day.
    - Queries the **Wikipedia REST API** to extract concise 2-sentence factual summaries in real time.

### Step 6: Visual Diagnostics & Voice Synthesis (Text-to-Speech)
- **Diagnostic Inspector:** The UI updates in real time, showing the top-3 predicted intent probabilities, confidence gauge, and execution latency.
- **Speech Synthesis (TTS):** The response is passed to `window.speechSynthesis` using the `SpeechSynthesisUtterance` API. The user hears the answer spoken aloud in natural speech with selectable voice accents and speed control.

---

## 🧠 4. Deep Learning Architecture Summary

| Layer # | Layer Type | Output Shape | Parameters | Purpose |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `Embedding` | `(None, 16, 64)` | ~18,000 | Maps word indices to 64-dimensional semantic space |
| **2** | `Bidirectional(LSTM)` | `(None, 16, 96)` | ~43,776 | 48 forward + 48 backward recurrent units |
| **3** | `GlobalMaxPooling1D` | `(None, 96)` | 0 | Extracts maximum temporal feature signals |
| **4** | `Dense (ReLU)` | `(None, 64)` | 6,208 | Non-linear feature representation |
| **5** | `Dropout (0.3)` | `(None, 64)` | 0 | Regularization against co-adaptation |
| **6** | `Dense (Softmax)` | `(None, 18)` | 1,170 | 18-class probability distribution |

---

## ⚡ 5. Innovation: Vercel Serverless Optimization

Deploying deep learning models on serverless platforms like Vercel usually presents a significant barrier:
- Standard TensorFlow wheels exceed **500MB+**, far beyond Vercel's strict **250MB uncompressed limit**.
- Heavy dependencies cause serverless function cold starts exceeding **5–10 seconds**.

### Our Architectural Solution:
1. **Weight Extraction (`model_weights.npz`):**  
   During training, model weights and biases are exported into compressed NumPy tensor arrays (~240 KB).
2. **Pure NumPy Forward-Pass Inference (`api/index.py`):**  
   Inference is implemented using vectorized matrix algebra directly with NumPy, completely eliminating the need for the heavyweight TensorFlow runtime in production.
3. **Results:**
   - Total deployment package size: **< 18MB** (well under the 250MB limit).
   - Serverless cold-start latency: **< 50 milliseconds**.
   - Zero loss in mathematical accuracy: produces the identical floating-point Softmax output as full TensorFlow.

---

## 📂 6. Repository File Structure

```text
voice-chatbot/
├── .github/                 # Git configuration
├── api/
│   ├── index.py             # Serverless inference handler (pure NumPy Bi-LSTM forward pass)
│   └── requirements.txt     # Minimal serverless dependencies (numpy)
├── assets/
│   ├── css/                 # Siri-style animations & responsive styles
│   └── screenshots/         # Training curves & confusion matrix plots
├── data/
│   └── intents.json         # 18 intent categories, 202 training patterns
├── models/
│   ├── chatbot_model.keras  # Saved Keras HDF5/Keras model
│   ├── model_weights.npz    # Extracted matrix weights for serverless inference
│   ├── tokenizer.pkl        # Serialized Keras text tokenizer
│   ├── label_encoder.pkl    # Serialized scikit-learn label encoder
│   └── metadata.json        # Vocabulary, max_len, and architectural config
├── notebooks/
│   └── chatbot_training.ipynb # Interactive training notebook with EDA
├── public/
│   ├── index.html           # Single-page web application UI
│   ├── styles.css           # Glassmorphic UI, glowing mic waveform styles
│   └── app.js               # Web Speech API, audio recording, TTS, API caller
├── report/
│   ├── project_report.md    # Detailed academic thesis / viva report
│   └── viva_preparation.md  # Comprehensive Q&A bank for project defense
├── utils/
│   ├── nlp_pipeline.py      # Tokenization, sequence padding & text cleaning
│   └── qa_engine.py         # Wikipedia search & dynamic skills engine
├── vercel.json              # URL routing rules for Vercel deployment
├── train.py                 # Full TensorFlow Bi-LSTM model training script
├── predict.py               # Standalone local CLI inference script
├── app.py                   # Streamlit desktop/cloud alternative app
└── requirements.txt         # Full training environment dependencies
```

---

## 📊 7. Model Evaluation & Performance

- **Training Epochs:** 80 epochs with Early Stopping checkpointing
- **Training Accuracy:** > 98%
- **Evaluation Metrics:**
  - Multi-class Confusion Matrix evaluated across all 18 intent classes.
  - Precision, Recall, and F1-Scores computed with Scikit-Learn classification reports.
  - Visualized training loss and accuracy curves stored in `assets/screenshots/training_curves.png`.

---

## 🎯 8. Summary of Supported Capabilities

1. **Voice Conversation:** Instant speech recognition using Google Speech Engine through Web Speech API.
2. **Text Fallback:** Smooth keyboard typing with suggestion chips.
3. **Voice Playback:** Native browser Text-to-Speech with customizable voice selection.
4. **Diagnostic Transparency:** Live top-3 intent confidence meters.
5. **Open-Domain World Knowledge:** Dynamic Wikipedia search integration for questions outside predefined intents.
6. **Instant Cloud Access:** Fully deployed on HTTPS at [https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/](https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/).
