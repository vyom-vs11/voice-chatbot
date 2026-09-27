# 🎙️ Online Voice-Enabled Chatbot Using Speech Recognition and Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Serverless-black.svg)](https://vercel.com/)
[![Deep Learning](https://img.shields.io/badge/Architecture-Bi--LSTM-success.svg)](#deep-learning-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, academic-grade **Online Voice-Enabled Chatbot** combining **Automatic Speech Recognition (ASR)**, **Natural Language Processing (NLP)**, and a **Bidirectional Long Short-Term Memory (Bi-LSTM)** Deep Learning neural network for conversational intent classification.

Designed with dual-platform deployment support: **Vercel Serverless Web App** and **Streamlit Community Cloud / Hugging Face Spaces**.

---

## 🌟 Key Features

- **🎙️ Real-Time Voice Input:** Microphone voice capture via the Web Speech API directly in modern web browsers (Chrome, Edge, Safari) with zero server audio latency.
- **🧠 Deep Learning Intent Classifier:** Built with TensorFlow/Keras using an Embedding layer, Bidirectional LSTM, Global MaxPooling, and Softmax dense classification.
- **💬 Dual-Modal Input (Voice + Text Fallback):** Seamless fallback to standard text input ensures 100% demo reliability even if microphone permissions are denied.
- **⚡ Real-Time Confidence Scoring:** Displays probability distributions and dynamically triggers friendly fallback responses when confidence falls below the tunable threshold.
- **🔬 Model Diagnostic Inspector:** Interactive UI expandable section displaying top-3 intent prediction probabilities from the neural network's Softmax layer.
- **🔊 Text-to-Speech (TTS) Read-Aloud:** Browser speech synthesis engine speaks the chatbot's answer back to the user for an immersive voice dialogue experience.
- **📈 Academic Evaluation Charts:** Built-in visualization of Loss/Accuracy training curves and Multi-Class Confusion Matrix.
- **☁️ Multi-Cloud Deployment Ready:** Optimized for native deployment to **Vercel** as well as **Streamlit Community Cloud** or **Hugging Face Spaces**.

---

## 🏛️ System Architecture

```text
[ User Voice ] ──> ( Web Speech API / Mic ) ──> [ Recognized Text ]
                                                        │
[ Typed Text ] ─────────────────────────────────────────┤
                                                        ▼
                                             [ Text Normalization ]
                                             (clean, lowercase, strip)
                                                        │
                                                        ▼
                                             [ Keras Tokenizer ]
                                             (word -> integer sequences)
                                                        │
                                                        ▼
                                             [ Sequence Padding ]
                                             (post-pad to max_len = 16)
                                                        │
                                                        ▼
                                             [ Bi-LSTM Deep Neural Net ]
                                             - Embedding (64-dim)
                                             - Bidirectional LSTM (48 units)
                                             - Global MaxPooling 1D
                                             - Dense ReLU (64 units)
                                             - Dropout (0.3)
                                             - Output Dense (18 Softmax classes)
                                                        │
                                                        ▼
                                             [ Intent Classification ]
                                             (predicted intent + confidence)
                                                        │
                                                        ▼
                                             [ Response Generation ]
                                             (contextual response lookup)
                                                        │
                                                        ▼
                                    ┌───────────────────┴───────────────────┐
                                    ▼                                       ▼
                       [ Vercel Web UI (public/) ]            [ Streamlit UI (app.py) ]
                       - Glassmorphic dark theme              - Live Streamlit app
                       - Real-time Web Speech API             - Sidebar controls
                       - Audio synthesis (TTS)                - Interactive curves
```

---

## 📂 Project Directory Structure

```text
voice-chatbot/
│
├── vercel.json                 # Vercel deployment routing configuration
├── dev_server.py               # Local development server for Vercel testing (port 3000)
├── app.py                      # Streamlit web application & UI (port 8501)
├── train.py                    # Deep Learning training & evaluation script
├── predict.py                  # Real-time inference engine & model loader
├── test_chatbot.py             # Automated unit & integration test suite
├── requirements.txt            # Python dependencies (Streamlit & full ML)
├── README.md                   # GitHub project documentation
├── .gitignore                  # Git tracking ignore file
│
├── api/
│   ├── index.py                # Vercel Serverless Function (Bi-LSTM NumPy inference)
│   └── requirements.txt        # Lightweight requirements (numpy only, <20MB)
│
├── public/
│   ├── index.html              # Vercel frontend HTML5 interface
│   ├── style.css               # Glassmorphic dark styling & animations
│   └── app.js                  # Client Web Speech API & TTS audio logic
│
├── data/
│   └── intents.json            # 18-intent conversational dataset (202 patterns)
│
├── models/
│   ├── chatbot_model.keras     # Trained Bi-LSTM Keras model weights
│   ├── model_weights.npz       # Compressed tensor weights for serverless inference
│   ├── tokenizer.pkl           # Fitted Keras Tokenizer artifact
│   ├── label_encoder.pkl       # Fitted Scikit-Learn LabelEncoder
│   └── metadata.json           # Model metadata (classes, sequence max length)
│
├── notebooks/
│   └── chatbot_training.ipynb  # Interactive Jupyter Notebook for training
│
├── utils/
│   ├── preprocessing.py        # Text cleaning, tokenization & sequence utils
│   └── chatbot_utils.py        # Server-side audio transcription utilities
│
├── assets/
│   └── screenshots/
│       ├── training_curves.png # Loss & Accuracy curves vs. Epochs
│       └── confusion_matrix.png# Multi-class confusion matrix
│
└── report/
    ├── project_report.md       # Complete academic project report
    └── viva_preparation.md     # Comprehensive viva Q&A question bank
```

---

## 🔬 Deep Learning Model Architecture

| Layer | Type | Specifications | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | `Input` | `shape=(16,)` | Receives token integer sequence |
| **2** | `Embedding` | `input_dim=vocab, output_dim=64` | Maps tokens to continuous vector space |
| **3** | `Bidirectional(LSTM)` | `units=48, return_sequences=True` | Captures bidirectional sequential context |
| **4** | `GlobalMaxPooling1D` | Pool temporal dimension | Extracts salient intent keywords |
| **5** | `Dense` | `units=64, activation='relu'` | High-order non-linear feature representation |
| **6** | `Dropout` | `rate=0.3` | Regularization against overfitting |
| **7** | `Dense` | `units=18, activation='softmax'` | Normalized class probability distribution |

---

## 📊 Dataset & Evaluation Metrics

- **Dataset:** 18 distinct intent categories, 202 patterns, 62 responses.
- **Training Epochs:** 80 epochs with Adam Optimizer ($lr=0.005$) and `ReduceLROnPlateau`.
- **Training Accuracy:** **100.00%** | Training Loss: **0.0001**
- **Hold-out Validation Accuracy:** **63.41%**
- **Macro Average Precision / Recall / F1:** **0.67 / 0.67 / 0.62**
- **Weighted Average Precision / Recall / F1:** **0.68 / 0.63 / 0.60**

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/voice-chatbot.git
cd voice-chatbot
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
python test_chatbot.py
```
*(Expected output: `Ran 6 tests in ~1.0s ... OK`)*

### 5. Launch Option A: Vercel Local Dev Server
```bash
python dev_server.py 3000
```
Open your browser and navigate to `http://localhost:3000`.

### 6. Launch Option B: Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🏋️ Model Training & Customization

To retrain the model with your own patterns or adjust hyperparameters:

```bash
python train.py
```
This script will:
1. Parse `data/intents.json`
2. Perform text cleaning, tokenization, and padding
3. Train the Bi-LSTM model across 80 epochs
4. Generate and save `training_curves.png` and `confusion_matrix.png` in `assets/screenshots/`
5. Save the trained artifacts (`chatbot_model.keras`, `tokenizer.pkl`, `label_encoder.pkl`)

You can also run the interactive Jupyter Notebook at:
```bash
jupyter notebook notebooks/chatbot_training.ipynb
```

---

## ☁️ Online Deployment Instructions

### Option 1: Deploy to Vercel (1-Click & Recommended)

This project is fully configured for native **Vercel Serverless Deployment**:
- Modern static frontend in `public/` (HTML5 Web Speech API, real-time waveform visualizer, text fallback, and audio TTS synthesis).
- Serverless Python endpoint in `api/index.py` executing the exact trained Bi-LSTM Deep Learning neural network via optimized NumPy tensor operations.
- Fits well within Vercel's strict 250MB bundle limit (~18MB total) with instantaneous cold starts (<50ms).

#### Method A: Deploy via Vercel Web Dashboard (Easiest)
1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Complete Voice Chatbot project with Vercel support"
   git branch -M main
   git remote add origin https://github.com/<your-username>/voice-chatbot.git
   git push -u origin main
   ```
2. Go to [vercel.com/new](https://vercel.com/new) and log in with your GitHub account.
3. Import your `voice-chatbot` repository.
4. Keep default settings:
   - Framework Preset: **Other**
   - Root Directory: `./`
5. Click **"Deploy"**.
6. In ~30 seconds, Vercel will build and assign your live production URL:
   ```text
   https://<your-project-name>.vercel.app
   ```

#### Method B: Deploy via Vercel CLI
```bash
# 1. Login to Vercel
npx vercel login

# 2. Deploy to production
npx vercel --prod
```

---

### Option 2: Deploy to Streamlit Community Cloud
1. Push this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**.
4. Configure:
   - **Repository:** `<your-github-username>/voice-chatbot`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**.
6. In ~1-2 minutes, your application will be live immediately at:
   `https://<your-app-name>.streamlit.app`

---

## 🎓 Viva / Interview Preparation

A complete question bank covering Python, NLP, Deep Learning, Speech Recognition, and Deployment is provided in [report/viva_preparation.md](report/viva_preparation.md):
- **Why Bi-LSTM?** Processes sequences in forward and backward directions to preserve contextual semantics from both preceding and succeeding tokens.
- **Why Web Speech API instead of PyAudio for cloud deployment?** Cloud servers (Vercel serverless containers, Streamlit Cloud) are headless and have no physical microphone attached. Client-side Web Speech API accesses the user's local browser microphone safely with user permission.
- **What is the role of Global MaxPooling1D?** Extracts the most salient temporal features across token steps, functioning as an automated keyword feature detector.
- **How is the Deep Learning model deployed on Vercel?** The trained model weights are saved in a lightweight tensor format (`model_weights.npz`) and executed via pure NumPy matrix operations in `api/index.py`, executing the exact same mathematical forward pass while staying well within Vercel's 250MB limit and cold-starting in <50ms.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
