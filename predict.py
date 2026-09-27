"""
Inference Module for Voice-Enabled Deep Learning Chatbot
- Siri-Style Open Knowledge Answering Engine (Wikipedia, Clock, Calculator)
- Bi-LSTM Keras Deep Learning Intent Classifier
- Preprocesses user queries and retrieves contextual responses
"""

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf

from utils.preprocessing import clean_text, preprocess_input, load_artifacts
from utils.qa_engine import resolve_knowledge_query, fetch_wikipedia_knowledge

# ----------------- PATHS ----------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")

# Default Confidence Threshold
CONFIDENCE_THRESHOLD = 0.55


class ChatbotPredictor:
    """Singleton predictor that caches loaded model artifacts for fast real-time inference."""
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH,
                 label_encoder_path=LABEL_ENCODER_PATH, metadata_path=METADATA_PATH,
                 intents_path=INTENTS_PATH):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.metadata_path = metadata_path
        self.intents_path = intents_path

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_len = 16
        self.intents_dict = {}

        self._load_all()

    def _load_all(self):
        """Loads model and preprocessing utilities into memory."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please run train.py first."
            )
        
        # Load Deep Learning Model
        self.model = tf.keras.models.load_model(self.model_path)

        # Load Tokenizer & Label Encoder
        self.tokenizer, self.label_encoder, self.max_len = load_artifacts(
            self.tokenizer_path, self.label_encoder_path, self.metadata_path
        )

        # Load Response Map from JSON
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("intents", []):
                self.intents_dict[item["tag"]] = item["responses"]

    def process_query(self, user_query: str, threshold: float = CONFIDENCE_THRESHOLD):
        """
        Siri-style processing:
        1. Check dynamic skills (Clock, Calculator, Sports, Wikipedia)
        2. Deep Learning Bi-LSTM Intent Classifier
        3. Open Knowledge fallback
        """
        user_query = user_query.strip()
        if not user_query:
            return {
                "query": "",
                "intent": "empty_input",
                "confidence": 0.0,
                "response": "Please speak or type a message to start our conversation.",
                "source": "System",
                "top_predictions": []
            }

        # 1. Siri Dynamic Skills & Knowledge Search
        kb_res = resolve_knowledge_query(user_query)
        if kb_res["answered"]:
            return {
                "query": user_query,
                "intent": kb_res["intent"],
                "confidence": kb_res["confidence"],
                "is_fallback": False,
                "response": kb_res["response"],
                "source": kb_res.get("source", "Siri Knowledge Engine"),
                "top_predictions": [
                    {"intent": kb_res["intent"], "confidence": kb_res["confidence"]}
                ]
            }

        # 2. Deep Learning Bi-LSTM Inference
        cleaned = clean_text(user_query)
        input_seq = preprocess_input(cleaned, self.tokenizer, max_len=self.max_len)
        predictions = self.model.predict(input_seq, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        predicted_tag = str(self.label_encoder.classes_[top_idx])

        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {"intent": str(self.label_encoder.classes_[idx]), "confidence": round(float(predictions[idx]), 4)}
            for idx in top_3_indices
        ]

        words = cleaned.split()
        oov_count = sum(1 for w in words if w not in self.tokenizer.word_index)
        is_oov_heavy = len(words) > 2 and (oov_count / len(words)) >= 0.50

        # 3. High confidence domain match
        if confidence >= threshold and not is_oov_heavy:
            responses = self.intents_dict.get(predicted_tag, ["Here is what I found for your request."])
            return {
                "query": user_query,
                "intent": predicted_tag,
                "confidence": confidence,
                "is_fallback": False,
                "response": random.choice(responses),
                "source": "Bi-LSTM Neural Network",
                "top_predictions": top_predictions
            }

        # 4. Open-domain Wikipedia lookup fallback
        wiki_ans = fetch_wikipedia_knowledge(user_query)
        if wiki_ans:
            return {
                "query": user_query,
                "intent": "open_knowledge",
                "confidence": 0.94,
                "is_fallback": False,
                "response": wiki_ans,
                "source": "Wikipedia Knowledge Base",
                "top_predictions": [
                    {"intent": "open_knowledge", "confidence": 0.94},
                    {"intent": predicted_tag, "confidence": confidence}
                ]
            }

        # 5. Polite fallback
        fallback_responses = [
            "I'm not quite sure about that, but you can ask me about our courses, operating hours, campus location, world topics, sports, or ask me to calculate something!",
            "I didn't quite catch that. You can speak into your microphone or ask about admissions, contact information, sports, or facts!",
            "I'm still learning new concepts every day. Could you try rephrasing your question or ask about our services?"
        ]
        return {
            "query": user_query,
            "intent": "fallback",
            "confidence": confidence,
            "is_fallback": True,
            "response": random.choice(fallback_responses),
            "source": "Conversational Fallback",
            "top_predictions": top_predictions
        }


# Global cached instance
_PREDICTOR_INSTANCE = None

def get_predictor():
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ChatbotPredictor()
    return _PREDICTOR_INSTANCE


if __name__ == "__main__":
    predictor = get_predictor()
    test_queries = [
        "Who is the best cricketer",
        "What is the capital of Japan",
        "What is the time right now",
        "What are your working hours?",
        "Where is your campus located?",
        "What is 25 percent of 400"
    ]
    print("\n--- Running Siri Engine Tests ---")
    for q in test_queries:
        res = predictor.process_query(q)
        print(f"\nUser Query: {res['query']}")
        print(f"Source: {res['source']} | Intent: {res['intent']} ({res['confidence']*100:.1f}%)")
        print(f"Chatbot Response: {res['response']}")
