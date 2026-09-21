"""
compare_all_models.py
Combines classical ML results + BiLSTM results (if available) into one
final comparison table and chart for the report/paper.

If you skipped train_lstm.py (e.g. no TensorFlow on your Python version),
this script still works fine using just the classical ML results.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'


def main():
    classical_df = pd.read_csv(f'{RESULTS_DIR}/classical_ml_comparison.csv')

    lstm_path = f'{RESULTS_DIR}/bilstm_metrics.json'
    if os.path.exists(lstm_path):
        with open(lstm_path) as f:
            lstm_metrics = json.load(f)
        all_df = pd.concat([classical_df, pd.DataFrame([lstm_metrics])], ignore_index=True)
    else:
        print("Note: bilstm_metrics.json not found (train_lstm.py was not run) — "
              "comparing classical ML models only.\n")
        all_df = classical_df.copy()

    all_df = all_df.sort_values('f1_score', ascending=False).reset_index(drop=True)

    all_df.to_csv(f'{RESULTS_DIR}/final_model_comparison.csv', index=False)
    print("=== FINAL MODEL COMPARISON (all models) ===")
    print(all_df.to_string(index=False))

    # Final comparison bar chart (for report/paper Results section)
    plt.figure(figsize=(9, 5))
    x = range(len(all_df))
    width = 0.2
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for i, m in enumerate(metrics):
        plt.bar([p + i * width for p in x], all_df[m], width, label=m)
    plt.xticks([p + 1.5 * width for p in x], all_df['model'], rotation=15)
    plt.ylabel('Score')
    plt.ylim(0.9, 1.01)
    plt.title('Final Model Comparison: Classical ML vs Deep Learning')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/final_model_comparison.png', dpi=150)
    plt.close()

    print(f"\nSaved final_model_comparison.csv and .png to {RESULTS_DIR}/")
    print(f"\nBest overall model: {all_df.iloc[0]['model']} (F1 = {all_df.iloc[0]['f1_score']:.4f})")


if __name__ == '__main__':
    main()
