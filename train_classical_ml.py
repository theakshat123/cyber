"""
train_classical_ml.py
Trains and compares classical ML models (Naive Bayes, Logistic Regression, SVM)
on TF-IDF features for cyberbullying detection.

Handles class imbalance via class_weight='balanced' (SVM, LogReg) and
saves the best model + vectorizer + all metrics/plots to disk.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

RANDOM_STATE = 42
DATA_PATH = 'data/cleaned_dataset.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['clean_text', 'label'])
    return df


def build_models():
    return {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
        ),
        # LinearSVC has no predict_proba; wrap with Platt scaling so we can
        # get probabilities for ROC-AUC comparisons later if needed.
        'SVM (Linear)': CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000),
            cv=3
        ),
    }


def evaluate_model(name, model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_{safe_name}.png', dpi=150)
    plt.close()

    return {'model': name, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1}


def main():
    print("Loading cleaned data...")
    df = load_data()
    labels = sorted(df['label'].unique())

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df['label']
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, f'{MODELS_DIR}/tfidf_vectorizer.joblib')

    results = []
    best_model, best_f1, best_name = None, -1, None

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        metrics = evaluate_model(name, model, X_test_vec, y_test, labels)
        results.append(metrics)

        if metrics['f1_score'] > best_f1:
            best_f1, best_model, best_name = metrics['f1_score'], model, name

    # Save comparison table
    results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
    results_df.to_csv(f'{RESULTS_DIR}/classical_ml_comparison.csv', index=False)
    print("\n=== Model comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))

    # Bar chart comparing all models
    plt.figure(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.2
    for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1_score']):
        plt.bar(x + i * width, results_df[metric], width, label=metric)
    plt.xticks(x + 1.5 * width, results_df['model'], rotation=15)
    plt.ylabel('Score')
    plt.title('Classical ML Model Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/classical_ml_comparison.png', dpi=150)
    plt.close()

    # Save best classical model
    joblib.dump(best_model, f'{MODELS_DIR}/best_classical_model.joblib')
    with open(f'{RESULTS_DIR}/best_classical_model.json', 'w') as f:
        json.dump({'name': best_name, 'f1_score': best_f1}, f, indent=2)

    print(f"\nBest classical model: {best_name} (F1 = {best_f1:.4f}) saved to {MODELS_DIR}/best_classical_model.joblib")


if __name__ == '__main__':
    main()
