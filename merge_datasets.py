"""
merge_datasets.py
Merges the original identity-based cyberbullying dataset with the Jigsaw
Toxic Comment dataset to add a general "offensive/profanity" category.

This fixes the key limitation: the original dataset only flags
identity-based bullying (race/gender/religion) and misses generic
abusive language (swearing, insults, threats with no identity reference).

Final label set:
    not_cyberbullying
    ethnicity/race
    gender/sexual
    religion
    offensive/profanity   <-- NEW: generic swearing, insults, threats
"""

import pandas as pd
import numpy as np

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ORIGINAL_PATH = 'data/cyberbullying_dataset.csv'
JIGSAW_PATH = 'data/toxic_comments_raw.csv'
OUTPUT_PATH = 'data/cyberbullying_dataset_v2.csv'

# How many additional "not_cyberbullying" examples to pull from Jigsaw's
# clean comments, to add topic diversity (Wikipedia talk-page style text,
# not just tweets).
N_EXTRA_CLEAN = 15000


def main():
    print("Loading original identity-based dataset...")
    orig = pd.read_csv(ORIGINAL_PATH)
    orig = orig.rename(columns={'text': 'text', 'label': 'label'})[['text', 'label']]
    print(f"  {orig.shape[0]} rows, labels: {orig['label'].unique().tolist()}")

    print("Loading Jigsaw toxic comment dataset...")
    jig = pd.read_csv(JIGSAW_PATH)
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    is_offensive = jig[toxic_cols].sum(axis=1) > 0

    offensive_texts = jig.loc[is_offensive, 'comment_text']
    clean_texts = jig.loc[~is_offensive, 'comment_text']

    print(f"  Offensive/profane comments found: {len(offensive_texts)}")
    print(f"  Clean comments available: {len(clean_texts)}")

    offensive_df = pd.DataFrame({
        'text': offensive_texts,
        'label': 'offensive/profanity'
    })

    extra_clean_df = pd.DataFrame({
        'text': clean_texts.sample(n=min(N_EXTRA_CLEAN, len(clean_texts)), random_state=RANDOM_STATE),
        'label': 'not_cyberbullying'
    })

    merged = pd.concat([orig, offensive_df, extra_clean_df], ignore_index=True)
    merged = merged.dropna(subset=['text', 'label']).drop_duplicates(subset=['text'])
    merged = merged.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    print("\nFinal merged dataset:")
    print(merged['label'].value_counts())

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged dataset to {OUTPUT_PATH}")
    print("IMPORTANT: update preprocess.py's RAW_PATH to point to this new file, "
          "or just overwrite cyberbullying_dataset.csv with it.")


if __name__ == '__main__':
    main()
