"""
preprocess.py
Cleans and preprocesses the raw cyberbullying tweet dataset.

Steps:
1. Lowercase
2. Remove URLs, mentions (@user), hashtags symbol, numbers, punctuation
3. Tokenize
4. Remove stopwords
5. Lemmatize
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_SYMBOL_RE = re.compile(r'#')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')
MULTISPACE_RE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """Clean a single tweet/text string."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = HASHTAG_SYMBOL_RE.sub('', text)   # keep the word, drop the '#'
    text = NON_ALPHA_RE.sub(' ', text)        # remove numbers/punctuation/emojis
    text = MULTISPACE_RE.sub(' ', text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def load_and_clean(csv_path: str, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
    """Load raw CSV and return a cleaned DataFrame with 'clean_text' and 'label' columns."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col])
    df['clean_text'] = df[text_col].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]  # drop rows that became empty after cleaning
    df = df.rename(columns={label_col: 'label'})
    return df[['clean_text', 'label']].reset_index(drop=True)


if __name__ == '__main__':
    RAW_PATH = 'data/cyberbullying_dataset_v2.csv'
    OUT_PATH = 'data/cleaned_dataset.csv'

    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_and_clean(RAW_PATH)
    print(f"Cleaned dataset shape: {df.shape}")
    print(df['label'].value_counts())

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUT_PATH}")
