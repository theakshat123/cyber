# Cyberbullying Detection using Machine Learning

Final Year Major Project — GNIOT, Greater Noida | Group 24
Akshat Srivastav, Ankit Kumar, Ambuj Singh | Supervisor: Mr. Imran Ansari

## Overview

This project detects cyberbullying in social media text using both classical
Machine Learning (Naive Bayes, Logistic Regression, SVM with TF-IDF features)
and Deep Learning (Bidirectional LSTM with learned word embeddings), then
compares their performance.

## Dataset (v2 — merged)

- **Base dataset**: `mohamadahj/Cyberbully-Detection-Dataset` (GitHub) —
  ~100,000 labeled tweets, identity-based bullying only (race, gender,
  religion).
- **Added**: ~16,000 offensive/profane comments and ~15,000 extra clean
  comments from the **Jigsaw Toxic Comment Classification** dataset
  (Wikipedia talk-page comments), to add a general `offensive/profanity`
  category. This was added specifically because the original dataset
  missed generic swearing/insults with no identity reference (e.g. "fuck
  you", "you're stupid") — a limitation discovered during testing.
- Merge script: `merge_datasets.py` (creates `data/cyberbullying_dataset_v2.csv`).
  Note: `data/cyberbullying_dataset_v2.csv` (the final merged dataset) is
  already included, so you don't need to re-run `merge_datasets.py` unless
  you want to change how it's built — it needs the raw Jigsaw
  `toxic_comments_raw.csv` file (not included here, to save space) to run.
- **Final classes**: `not_cyberbullying`, `ethnicity/race`, `gender/sexual`,
  `religion`, `offensive/profanity`.
- **Remaining limitation to mention in your report**: this is still not a
  fully general toxicity detector — subtle/implicit bullying without slurs
  or profanity (e.g. sarcasm, backhanded comments) can still be missed.
  State this as future work (e.g. adding sarcasm-aware or context-aware
  models like BERT).

## Project Structure

```
cyberbullying_project/
├── data/
│   ├── cyberbullying_dataset.csv     # raw dataset
│   └── cleaned_dataset.csv            # after preprocessing
├── models/                            # saved trained models (generated)
├── results/                           # metrics, confusion matrices, charts (generated)
├── app/
│   └── streamlit_app.py               # interactive demo
├── preprocess.py                      # text cleaning pipeline
├── train_classical_ml.py              # Naive Bayes, Logistic Regression, SVM
├── train_lstm.py                      # BiLSTM deep learning model
├── compare_all_models.py              # final comparison table + chart
└── requirements.txt
```

## How to Run (in order)

```bash
pip install -r requirements_no_tf.txt   # or requirements.txt if you have Python <=3.12 and want BiLSTM too

python merge_datasets.py        # merges base + Jigsaw offensive data -> data/cyberbullying_dataset_v2.csv
python preprocess.py            # cleans merged data -> data/cleaned_dataset.csv
python train_classical_ml.py    # trains NB, LogReg, SVM -> models/, results/
python train_lstm.py            # OPTIONAL (needs TensorFlow): trains BiLSTM -> models/, results/
python compare_all_models.py    # final comparison -> results/final_model_comparison.csv/.png

streamlit run app/streamlit_app.py   # launch the live demo
```

## Results (5-class problem, on merged dataset v2)

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| SVM (Linear)         | 0.9387   | 0.9366    | 0.9387 | 0.9361   |
| Logistic Regression  | 0.9355   | 0.9374    | 0.9355 | 0.9362   |
| Naive Bayes          | 0.8825   | 0.8829    | 0.8825 | 0.8766   |

Note: this is a harder 5-class problem than the original 4-class version
(adding a distinct `offensive/profanity` class), so ~93-94% F1 here is
actually a strong result, not a regression. Per-class performance on
`offensive/profanity` is the weakest (recall ~0.68-0.80) since profane
language is more lexically diverse than clear-cut hate speech — worth
discussing as a limitation/future-work point. All confusion matrices are
in `results/`, ready to drop into your report/paper's Results section.

If you retrain the BiLSTM (`train_lstm.py`) on this new dataset, expect a
similar or better F1-score than the classical models on this harder
5-class task — that's a legitimate result to report.

## NEW: KYC-Gated Access System

The project now requires KYC verification before anyone can access the
cyberbullying detection tool — simulating a real-world platform access
control flow.

**Flow:**
1. **Login/Signup** — dark/light mode toggle, username + password.
2. **KYC Verification** — new users submit a demo KYC form (name, ID
   number, DOB, address). ⚠️ **Use only fake/sample data** — this is a
   mock form for academic demo purposes, not real government ID
   verification, and must never be used with real Aadhaar/ID data.
3. **Admin Approval** — a hidden admin panel (expand "🔧 Admin access" on
   the login page; default demo credentials `admin` / `admin123`, **change
   this** in `app/db.py` before showing to anyone) lets an admin approve
   or reject pending KYC requests.
4. **Main Project Access** — once approved, the user reaches the full
   cyberbullying detection dashboard (all 5 tabs).
5. **Auto-Cancellation** — if a user posts `FLAG_THRESHOLD` (default: 5,
   configurable in `app/db.py`) cyberbullying-flagged comments in the
   Social Feed Demo, their KYC is **automatically cancelled** and access
   is revoked immediately. Admins can manually "Reinstate" a cancelled
   user from the "👥 All Users" tab in the admin dashboard.

New files: `app/db.py` (SQLite storage), `app/auth_pages.py` (login/signup/
KYC/status screens), `app/admin_page.py` (admin dashboard). A local
`app_data.db` SQLite file is created automatically on first run.

## What to use for each deliverable

- **Synopsis**: use the Introduction/Methodology structure below + the
  dataset description above.
- **Major Project Report**: use `results/*.png` for figures, and
  `results/final_model_comparison.csv` for your results table.
- **Research Paper**: condense Methodology + Results sections; cite the
  dataset paper and any 2-3 related works.
- **Working demo for viva**: run `streamlit run app/streamlit_app.py`.

## Future Work / Extensions (mentioned in your original PPT)

- Fine-tune a transformer model (BERT/DistilBERT) for higher accuracy on
  sarcasm/context-heavy posts.
- Extend to multilingual / code-mixed (Hindi-English) text.
- Integrate with a real-time moderation pipeline (e.g., a browser extension
  or API webhook).
