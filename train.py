"""
Model Training Pipeline for Voice-Enabled Deep Learning Chatbot
- Loads and parses intents.json
- Performs text preprocessing and tokenization
- Builds an Embedding + Bidirectional LSTM + GlobalMaxPooling1D Deep Learning architecture
- Trains and evaluates the model with stratified train/validation split
- Generates metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- Saves model and artifacts for inference and deployment
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils.preprocessing import (
    load_intents,
    prepare_training_data,
    save_artifacts
)

# ----------------- CONFIGURATION ----------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "screenshots")

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
TOKENIZER_SAVE_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_SAVE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_SAVE_PATH = os.path.join(MODEL_DIR, "metadata.json")

MAX_VOCAB_SIZE = 2000
MAX_LEN = 16
EMBEDDING_DIM = 64
LSTM_UNITS = 48
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
EPOCHS = 80
BATCH_SIZE = 8
TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_model(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    """
    Constructs the Deep Learning architecture:
    1. Input layer for token index sequences
    2. Embedding layer: maps token indices to 64-dimensional dense vectors
    3. Bidirectional LSTM: captures sequence semantics in both forward and reverse directions
    4. GlobalMaxPooling1D: extracts salient temporal features across token steps
    5. Dense feature transformation layer with ReLU activation
    6. Dropout: regularization to mitigate overfitting
    7. Output Dense Layer with Softmax: probability distribution across intent classes
    """
    model = Sequential([
        Input(shape=(max_len,), name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            name="embedding_layer"
        ),
        Bidirectional(
            LSTM(LSTM_UNITS, return_sequences=True),
            name="bidirectional_lstm"
        ),
        GlobalMaxPooling1D(name="global_max_pooling"),
        Dense(DENSE_UNITS, activation="relu", name="dense_features"),
        Dropout(DROPOUT_RATE, name="dropout_regularization"),
        Dense(num_classes, activation="softmax", name="output_intents")
    ], name="VoiceChatbot_BiLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_and_save_training_history(history, save_path: str):
    """Generates and saves Loss and Accuracy curves across training epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history["accuracy"], label="Training Accuracy", color="#2563EB", lw=2)
    if "val_accuracy" in history.history:
        axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#10B981", lw=2, linestyle="--")
    axes[0].set_title("Model Accuracy vs Epochs", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    axes[1].plot(history.history["loss"], label="Training Loss", color="#EF4444", lw=2)
    if "val_loss" in history.history:
        axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#F59E0B", lw=2, linestyle="--")
    axes[1].set_title("Model Loss vs Epochs", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("Categorical Loss", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Training curves saved to: {save_path}")


def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    """Computes and visualizes the Confusion Matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Intent Classification Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Intent", fontsize=12)
    plt.ylabel("Ground Truth Intent", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix saved to: {save_path}")


def main():
    print("=" * 60)
    print(" Voice-Enabled Chatbot: Deep Learning Training Pipeline")
    print("=" * 60)

    # 1. Load Dataset
    print(f"[*] Loading dataset from {DATA_PATH}...")
    patterns, tags, responses = load_intents(DATA_PATH)
    print(f"[+] Total pattern samples loaded: {len(patterns)}")
    print(f"[+] Unique intent categories: {len(set(tags))}")

    # 2. Preprocess and Vectorize
    print("[*] Preprocessing text and generating numerical sequences...")
    X, y, tokenizer, label_encoder, max_len = prepare_training_data(
        patterns, tags, max_vocab_size=MAX_VOCAB_SIZE, max_len=MAX_LEN
    )
    vocab_size = len(tokenizer.word_index) + 2  # account for OOV and padding
    num_classes = len(label_encoder.classes_)
    print(f"[+] Vocabulary size: {vocab_size}")
    print(f"[+] Sequence max length: {max_len}")
    print(f"[+] Output classes: {num_classes}")

    # 3. Train/Validation Split (Stratified)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=np.argmax(y, axis=1)
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    print(f"[+] Training samples: {X_train.shape[0]} | Validation samples: {X_val.shape[0]}")

    # 4. Build Deep Learning Model
    print("[*] Building Deep Learning Architecture (Embedding + BiLSTM + Dense)...")
    model = build_model(vocab_size=vocab_size, max_len=max_len, num_classes=num_classes)
    model.summary()

    # 5. Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    lr_reduction = ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    # 6. Train Model
    print("\n[*] Training deep neural network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, lr_reduction],
        verbose=1
    )

    # Load best checkpointed model for final evaluation
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[*] Loading best checkpoint from {MODEL_SAVE_PATH}...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # 7. Final Model Evaluation
    print("\n" + "=" * 60)
    print(" Model Evaluation on Hold-Out Validation Set")
    print("=" * 60)
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"[+] Final Training Accuracy:   {train_acc * 100:.2f}% | Loss: {train_loss:.4f}")
    print(f"[+] Final Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred_probs = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)

    print("\nDetailed Classification Report:")
    report = classification_report(
        y_val_true,
        y_val_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print(report)

    # 8. Save Evaluation Plots
    curves_path = os.path.join(ASSETS_DIR, "training_curves.png")
    cm_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    plot_and_save_training_history(history, curves_path)
    plot_and_save_confusion_matrix(y_val_true, y_val_pred, label_encoder.classes_, cm_path)

    # 9. Save Preprocessing Artifacts
    print("[*] Saving Tokenizer and Label Encoder...")
    save_artifacts(
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_len=max_len,
        tokenizer_path=TOKENIZER_SAVE_PATH,
        label_encoder_path=LABEL_ENCODER_SAVE_PATH,
        metadata_path=METADATA_SAVE_PATH
    )
    print(f"[+] Model saved to: {MODEL_SAVE_PATH}")
    print(f"[+] Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
    print(f"[+] Label Encoder saved to: {LABEL_ENCODER_SAVE_PATH}")
    print(f"[+] Metadata saved to: {METADATA_SAVE_PATH}")
    print("=" * 60)
    print(" Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
