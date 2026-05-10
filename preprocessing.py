"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 1 : Text Preprocessing                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Input  : IMDB-Dataset.csv                                           ║
║  Output : generated/IMDB_cleaned.csv                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import nltk
import pandas as pd
from paths import DATASET, CLEANED_CSV, GEN_DIR

# ── Download required NLTK data (safe to run multiple times) ──────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

print("=" * 60)
print("  STEP 1 — Text Preprocessing")
print("=" * 60)

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"\n[ERROR] '{DATASET}' not found.\n"
        f"  -> Place 'IMDB-Dataset.csv' in the same folder as this script."
    )

df = pd.read_csv(DATASET)
print(f"  Loaded dataset   : {len(df):,} rows")
print(f"  Columns          : {list(df.columns)}")
print(f"  Output folder    : {GEN_DIR}/")

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove punctuation and numbers
      4. Tokenize
      5. Remove stopwords and short tokens (<= 2 chars)
      6. Re-join
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

print("\n  Cleaning reviews — may take ~1 min on 50k rows...")
df['clean_review'] = df['review'].apply(clean_text)

before = len(df)
df = df[df['clean_review'].str.strip() != ""].reset_index(drop=True)
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped} empty documents after cleaning.")

df.to_csv(CLEANED_CSV, index=False)

print(f"\n  Clean documents saved : {len(df):,}")
print(f"  Saved -> {CLEANED_CSV}")
print(f"\n  Sample:")
print(f"  RAW  : {df['review'].iloc[0][:80]}...")
print(f"  CLEAN: {df['clean_review'].iloc[0][:80]}...")
print("\n  STEP 1 COMPLETE")
print("=" * 60)