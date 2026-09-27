"""
Text Preprocessing and Sequence Utility Functions
for Voice-Enabled Deep Learning Chatbot
"""

import json
import re
import string
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def clean_text(text: str) -> str:
    """
    Cleans raw input text:
    - Lowercases text
    - Removes punctuation and special characters
    - Normalizes multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Replace contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # Remove remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_intents(json_path: str):
    """
    Loads intents from a JSON file and extracts patterns, tags, and responses.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    patterns = []
    tags = []
    responses = {}

    for intent in data.get("intents", []):
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            cleaned_p = clean_text(pattern)
            if cleaned_p:
                patterns.append(cleaned_p)
                tags.append(tag)

    return patterns, tags, responses


def prepare_training_data(patterns, tags, max_vocab_size=2000, max_len=20):
    """
    Tokenizes text patterns and encodes categorical labels.
    Returns:
    - padded_sequences: numpy array of shape (N, max_len)
    - one_hot_labels: numpy array of shape (N, num_classes)
    - tokenizer: fitted Keras Tokenizer
    - label_encoder: fitted scikit-learn LabelEncoder
    - max_len: sequence padding length
    """
    # 1. Fit Tokenizer
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # 2. Fit LabelEncoder
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(tags)
    num_classes = len(label_encoder.classes_)

    # 3. One-hot encode targets for categorical crossentropy
    one_hot_labels = np.zeros((len(integer_labels), num_classes), dtype=np.float32)
    for i, idx in enumerate(integer_labels):
        one_hot_labels[i, idx] = 1.0

    return padded_sequences, one_hot_labels, tokenizer, label_encoder, max_len


def preprocess_input(text: str, tokenizer, max_len: int = 20) -> np.ndarray:
    """
    Prepares a single raw user utterance (voice transcript or typed text)
    for model inference by cleaning, tokenizing, and padding.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return np.zeros((1, max_len), dtype=np.int32)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded


def save_artifacts(tokenizer, label_encoder, max_len: int,
                   tokenizer_path: str, label_encoder_path: str, metadata_path: str = None):
    """
    Persists tokenizer, label encoder, and metadata for deployment.
    """
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    if metadata_path:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"max_len": max_len, "classes": list(label_encoder.classes_)}, f, indent=2)


def load_artifacts(tokenizer_path: str, label_encoder_path: str, metadata_path: str = None):
    """
    Loads saved tokenizer, label encoder, and configuration metadata.
    """
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    max_len = 20
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                max_len = meta.get("max_len", 20)
        except Exception:
            max_len = 20
    return tokenizer, label_encoder, max_len
