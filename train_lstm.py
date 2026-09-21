"""
train_lstm.py
Trains a BiLSTM deep learning model on the cyberbullying dataset using
Keras word embeddings (learned from scratch) and compares it against
the classical ML baselines.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
DATA_PATH = 'data/cleaned_dataset.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

MAX_VOCAB = 15000
MAX_LEN = 60
EMBED_DIM = 128

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def main():
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_PATH).dropna(subset=['clean_text', 'label'])
    labels_sorted = sorted(df['label'].unique())

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    np.save(f'{MODELS_DIR}/label_classes.npy', le.classes_)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

    import pickle
    with open(f'{MODELS_DIR}/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    print("Building BiLSTM model...")
    model = Sequential([
        Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    print("Training...")
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_split=0.1,
        epochs=8,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=2,
    )

    model.save(f'{MODELS_DIR}/bilstm_model.keras')

    print("Evaluating on test set...")
    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n=== BiLSTM ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - BiLSTM')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_bilstm.png', dpi=150)
    plt.close()

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/bilstm_training_curves.png', dpi=150)
    plt.close()

    # Save metrics for the final comparison across all models
    lstm_metrics = {'model': 'BiLSTM', 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1}
    with open(f'{RESULTS_DIR}/bilstm_metrics.json', 'w') as f:
        json.dump(lstm_metrics, f, indent=2)

    print(f"\nSaved BiLSTM model, confusion matrix, and training curves to {MODELS_DIR}/ and {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
