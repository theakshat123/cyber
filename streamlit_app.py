"""
streamlit_app.py
Full project dashboard: live detection, batch analysis, model performance
comparison, and project overview — for demo/viva purposes.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pickle

from preprocess import clean_text
import db
from auth_pages import render_login_signup, render_kyc_form, render_status_screen
from admin_page import render_admin_dashboard

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MAX_LEN = 60

st.set_page_config(page_title="Cyberbullying Detection using ML", page_icon="🛡️", layout="wide")

db.init_db()


@st.cache_resource
def load_classical():
    model = joblib.load(f'{MODELS_DIR}/best_classical_model.joblib')
    vectorizer = joblib.load(f'{MODELS_DIR}/tfidf_vectorizer.joblib')
    return model, vectorizer


@st.cache_resource
def load_deep():
    model = load_model(f'{MODELS_DIR}/bilstm_model.keras')
    with open(f'{MODELS_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    classes = np.load(f'{MODELS_DIR}/label_classes.npy', allow_pickle=True)
    return model, tokenizer, classes


def predict_classical(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, proba, model.classes_, cleaned


def predict_deep(text, model, tokenizer, classes):
    cleaned = clean_text(text)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LEN, padding='post')
    proba = model.predict(seq, verbose=0)[0]
    pred = classes[np.argmax(proba)]
    return pred, proba, classes, cleaned



def main_project(user):
    # ---------------------------------------------------------------------------
    # Sidebar — project info (always visible, gives instant credibility in a demo)
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.title("🛡️ Project Info")

        kyc_status = db.get_kyc_status(user['id'])
        flagged = kyc_status['flagged_count'] if kyc_status else 0
        st.success(f"Logged in as **{user['username']}** ✅ KYC Verified")
        st.caption(f"⚠️ Flagged comments: {flagged}/{db.FLAG_THRESHOLD}")
        if st.button("Log out", use_container_width=True):
            del st.session_state.user
            st.rerun()
        st.divider()

        st.markdown("""
        **Cyberbullying Detection using ML**
        Final Year Major Project

        **Team:** Akshat Srivastav, Ankit Kumar, Ambuj Singh
        **Supervisor:** Mr. Imran Ansari
        **Institution:** GNIOT, Greater Noida

        ---
        **Dataset:** ~131,000 labeled comments
        (base identity-bullying dataset + Jigsaw
        Toxic Comment dataset, merged)

        **Classes detected:**
        - Not Cyberbullying
        - Ethnicity / Race
        - Gender / Sexual
        - Religion
        - Offensive / Profanity

        **Models trained:** Naive Bayes, Logistic
        Regression, SVM (TF-IDF) + optional BiLSTM
        """)

    st.title("🛡️ Cyberbullying Detection using Machine Learning")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📱 Social Feed Demo", "🔍 Live Detection", "📊 Batch Analysis", "📈 Model Performance", "ℹ️ About the Project"
    ])

    # ---------------------------------------------------------------------------
    # TAB 0 (tab1 var): Instagram/Facebook-style comment section demo
    # ---------------------------------------------------------------------------
    with tab1:
        st.markdown("""
        <style>
        .ig-post {
            max-width: 500px; margin: 0 auto; border: 1px solid #dbdbdb;
            border-radius: 12px; overflow: hidden; background: white; font-family: -apple-system, sans-serif;
        }
        .ig-header {
            display: flex; align-items: center; padding: 12px 14px; border-bottom: 1px solid #efefef;
        }
        .ig-avatar {
            width: 36px; height: 36px; border-radius: 50%;
            background: linear-gradient(45deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888);
            display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 10px;
        }
        .ig-username { font-weight: 600; color: #262626; font-size: 14px; }
        .ig-image {
            width: 100%; height: 320px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex; align-items: center; justify-content: center; color: white; font-size: 60px;
        }
        .ig-actions { padding: 10px 14px; font-size: 22px; }
        .ig-caption { padding: 0 14px 8px 14px; font-size: 14px; color: #262626; }
        .ig-caption b { margin-right: 6px; }
        .ig-comments { padding: 4px 14px 14px 14px; max-height: 320px; overflow-y: auto; }
        .comment-row { display: flex; margin-bottom: 12px; }
        .comment-avatar {
            width: 28px; height: 28px; border-radius: 50%; background: #8e8e8e; color: white;
            display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;
            margin-right: 10px; flex-shrink: 0;
        }
        .comment-body { font-size: 13px; color: #262626; line-height: 1.4; }
        .comment-body b { margin-right: 6px; }
        .comment-time { color: #8e8e8e; font-size: 11px; margin-top: 2px; }
        .flagged-comment {
            background: #fff0f0; border: 1px solid #ffd6d6; border-radius: 8px; padding: 8px 10px;
        }
        .flagged-label {
            color: #ed4956; font-size: 11px; font-weight: 600; margin-bottom: 3px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Live moderation demo — post a comment like on Instagram/Facebook")
        st.caption("This simulates a real social media comment section. Every comment you post "
                   "is run through the trained ML model in real time, exactly like automated "
                   "content moderation on real platforms.")

        if 'feed_comments' not in st.session_state:
            st.session_state.feed_comments = [
                {"user": "priya_23", "text": "This looks amazing! Where was this taken?",
                 "flagged": False, "category": None},
                {"user": "rahul.k", "text": "Great shot 🔥🔥",
                 "flagged": False, "category": None},
            ]

        left, right = st.columns([1.3, 1])

        with left:
            html = '<div class="ig-post">'
            html += '<div class="ig-header"><div class="ig-avatar">GN</div><div class="ig-username">gniot_official</div></div>'
            html += '<div class="ig-image">🖼️</div>'
            html += '<div class="ig-actions">❤️ 💬 📤</div>'
            html += '<div class="ig-caption"><b>gniot_official</b>Final year project showcase day! 🎓</div>'
            html += '<div class="ig-comments">'

            for c in st.session_state.feed_comments:
                initials = c['user'][:2].upper()
                if c['flagged']:
                    html += f'''
                    <div class="comment-row">
                        <div class="comment-avatar">{initials}</div>
                        <div class="comment-body">
                            <div class="flagged-comment">
                                <div class="flagged-label">⚠️ Hidden by AI Moderation — {c['category']}</div>
                                <b>{c['user']}</b>[content hidden]
                            </div>
                            <div class="comment-time">Just now</div>
                        </div>
                    </div>'''
                else:
                    html += f'''
                    <div class="comment-row">
                        <div class="comment-avatar">{initials}</div>
                        <div class="comment-body">
                            <b>{c['user']}</b>{c['text']}
                            <div class="comment-time">Just now</div>
                        </div>
                    </div>'''
            html += '</div></div>'
            st.markdown(html, unsafe_allow_html=True)

        with right:
            st.markdown("**Post a comment as:**")
            username = st.text_input("Your username", value="you.demo", label_visibility="collapsed")
            new_comment = st.text_area("Write a comment...", height=100, placeholder="Add a comment...")

            if st.button("💬 Post Comment", type="primary", use_container_width=True):
                if new_comment.strip():
                    model, vectorizer = load_classical()
                    pred, proba, classes, cleaned = predict_classical(new_comment, model, vectorizer)
                    is_flagged = pred != "not_cyberbullying"

                    st.session_state.feed_comments.append({
                        "user": username or "you.demo",
                        "text": new_comment,
                        "flagged": is_flagged,
                        "category": pred if is_flagged else None,
                    })

                    if is_flagged:
                        new_count = db.increment_flag_count(user['id'])
                        remaining = db.FLAG_THRESHOLD - new_count
                        st.error(f"⚠️ Your comment was auto-hidden — detected as **{pred}** "
                                 f"({max(proba)*100:.0f}% confidence)")
                        if remaining > 0:
                            st.warning(f"🚨 Flagged comment {new_count}/{db.FLAG_THRESHOLD} on your account. "
                                       f"{remaining} more will result in your KYC/access being **cancelled**.")
                        else:
                            st.error("🚫 You have reached the flagged-comment limit. "
                                     "Your KYC has been **cancelled** and your access is now revoked.")
                            del st.session_state.user
                            st.rerun()
                    else:
                        st.success("✅ Comment posted — no issues detected")
                    st.rerun()
                else:
                    st.warning("Type a comment first.")

            st.divider()
            st.caption("Try posting a clean comment, then try one with slurs/profanity/hate "
                       "speech to see the moderation trigger live.")

            if st.button("🔄 Reset feed", use_container_width=True):
                del st.session_state.feed_comments
                st.rerun()


    with tab2:
        st.subheader("Analyze a single message")

        model_options = ["SVM (Classical ML)"]
        if TF_AVAILABLE and os.path.exists(f'{MODELS_DIR}/bilstm_model.keras'):
            model_options.append("BiLSTM (Deep Learning)")
        else:
            st.info("Only the classical ML (SVM) model is loaded. It performs "
                    "on par with or better than deep learning here, so this is not a downgrade.")

        col_input, col_examples = st.columns([2, 1])

        with col_examples:
            st.caption("Quick test examples:")
            examples = {
                "Friendly message": "Have a wonderful day, hope you feel better soon!",
                "Religion-based": "All people of that religion are terrorists",
                "Profanity/insult": "You are such a fucking bastard",
                "Ethnicity-based": "Go back to your country, you don't belong here",
            }
            for label, ex in examples.items():
                if st.button(label, use_container_width=True):
                    st.session_state['text_input'] = ex

        with col_input:
            text_input = st.text_area(
                "Enter a message / tweet / comment to analyze:",
                height=120, key='text_input',
                placeholder="Type or paste text here, or click an example on the right...",
            )
            model_choice = st.radio("Model:", model_options, horizontal=True)
            analyze_clicked = st.button("🔍 Analyze", type="primary")

        if analyze_clicked:
            if not text_input.strip():
                st.warning("Please enter some text first.")
            else:
                if model_choice == "SVM (Classical ML)":
                    model, vectorizer = load_classical()
                    pred, proba, classes, cleaned = predict_classical(text_input, model, vectorizer)
                else:
                    model, tokenizer, classes = load_deep()
                    pred, proba, classes, cleaned = predict_deep(text_input, model, tokenizer, classes)

                st.divider()
                r1, r2 = st.columns([1, 1])
                with r1:
                    st.subheader("Result")
                    if pred == "not_cyberbullying":
                        st.success("✅ Not Cyberbullying")
                    else:
                        st.error(f"⚠️ Cyberbullying detected — category: **{pred}**")

                    with st.expander("Cleaned/preprocessed text used by the model"):
                        st.code(cleaned or "(empty after cleaning)")

                with r2:
                    st.subheader("Confidence per class")
                    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                        st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

    # ---------------------------------------------------------------------------
    # TAB 3: Batch analysis (upload CSV of messages, get predictions for all)
    # ---------------------------------------------------------------------------
    with tab3:
        st.subheader("Analyze many messages at once")
        st.caption("Upload a CSV with a column of text (e.g. tweets, comments), "
                   "or paste multiple lines below.")

        upload = st.file_uploader("Upload CSV file", type=['csv'])
        pasted = st.text_area("...or paste one message per line", height=150,
                               placeholder="Message 1\nMessage 2\nMessage 3...")

        batch_texts = []
        if upload is not None:
            df_up = pd.read_csv(upload)
            col_choice = st.selectbox("Which column has the text?", df_up.columns.tolist())
            batch_texts = df_up[col_choice].dropna().astype(str).tolist()
        elif pasted.strip():
            batch_texts = [line.strip() for line in pasted.split('\n') if line.strip()]

        if batch_texts and st.button("Run batch analysis", type="primary"):
            model, vectorizer = load_classical()
            with st.spinner(f"Analyzing {len(batch_texts)} messages..."):
                rows = []
                for t in batch_texts:
                    pred, proba, classes, _ = predict_classical(t, model, vectorizer)
                    confidence = max(proba)
                    rows.append({'text': t, 'prediction': pred, 'confidence': f"{confidence*100:.1f}%"})
                result_df = pd.DataFrame(rows)

            st.success(f"Done — analyzed {len(result_df)} messages.")
            st.dataframe(result_df, use_container_width=True)

            flagged = (result_df['prediction'] != 'not_cyberbullying').sum()
            c1, c2 = st.columns(2)
            c1.metric("Flagged as cyberbullying", flagged)
            c2.metric("Clean messages", len(result_df) - flagged)

            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download results as CSV", csv_out, "batch_results.csv", "text/csv")

    # ---------------------------------------------------------------------------
    # TAB 4: Model performance (charts generated during training)
    # ---------------------------------------------------------------------------
    with tab4:
        st.subheader("Model comparison")

        comp_csv = f'{RESULTS_DIR}/final_model_comparison.csv'
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("Run compare_all_models.py first to generate this table.")

        comp_png = f'{RESULTS_DIR}/final_model_comparison.png'
        if os.path.exists(comp_png):
            st.image(comp_png, caption="Model comparison across metrics", use_container_width=True)

        st.divider()
        st.subheader("Confusion matrices")
        cm_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('confusion_matrix')] if os.path.isdir(RESULTS_DIR) else []
        if cm_files:
            cols = st.columns(min(3, len(cm_files)))
            for i, fname in enumerate(sorted(cm_files)):
                with cols[i % len(cols)]:
                    st.image(f'{RESULTS_DIR}/{fname}', use_container_width=True)
        else:
            st.info("No confusion matrix images found yet — run the training scripts first.")

    # ---------------------------------------------------------------------------
    # TAB 5: About / methodology
    # ---------------------------------------------------------------------------
    with tab5:
        st.subheader("About this project")
        st.markdown("""
        ### Problem Statement
        Cyberbullying on social media platforms causes significant psychological
        harm, and manual moderation cannot scale to the volume of daily posts.
        This project builds an automated ML-based system to detect and
        categorize cyberbullying in text.

        ### Methodology
        1. **Data collection**: merged an identity-based bullying dataset
           (~100K tweets: ethnicity, gender, religion) with the Jigsaw Toxic
           Comment dataset (~16K profane/offensive comments), to cover both
           identity-based hate speech and generic abusive language.
        2. **Preprocessing**: lowercasing, URL/mention removal, stopword
           removal, lemmatization.
        3. **Feature extraction**: TF-IDF (unigrams + bigrams, classical ML)
           and learned word embeddings (deep learning).
        4. **Modeling**: Naive Bayes, Logistic Regression, and SVM as
           classical baselines; a Bidirectional LSTM as the deep learning
           approach.
        5. **Evaluation**: accuracy, precision, recall, F1-score, and
           confusion matrices on a held-out 20% test set.

        ### Limitations & Future Work
        - Subtle/implicit bullying without explicit slurs or profanity
          (e.g. sarcasm, backhanded comments) can still be missed.
        - Could be extended with transformer models (BERT/DistilBERT) for
          better context understanding.
        - Could be extended to multilingual / code-mixed text (e.g.
          Hindi-English).
        """)


# ---------------------------------------------------------------------------
# ROUTING — decides which screen to show based on login/KYC state
# ---------------------------------------------------------------------------
if 'user' not in st.session_state:
    render_login_signup()
else:
    current_user = st.session_state.user

    if current_user['is_admin']:
        render_admin_dashboard(current_user)
    else:
        kyc = db.get_kyc_status(current_user['id'])
        status = kyc['status'] if kyc else 'not_submitted'

        if status == 'not_submitted' or st.session_state.get('force_kyc_form'):
            st.session_state.force_kyc_form = False
            render_kyc_form(current_user)
        elif status == 'approved':
            main_project(current_user)
        else:
            # pending, rejected, or cancelled
            render_status_screen(status, current_user)

