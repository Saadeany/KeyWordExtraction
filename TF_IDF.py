"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 2 : TF-IDF Feature Extraction & Baseline              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Input  : generated/IMDB_cleaned.csv                                 ║
║  Output : generated/tfidf_keywords_results.csv                       ║
║           generated/tfidf_feature_names.npy                          ║
╚══════════════════════════════════════════════════════════════════════╝

WHY TF-IDF?
  Term Frequency-Inverse Document Frequency scores a word by:
    - How often it appears in THIS document     (TF  up -> score up)
    - How rare it is across ALL documents       (IDF up -> score up)
  Common words like "the" get low IDF -> low score.
  Specific words like "acting" appear rarely -> high score.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from paths import CLEANED_CSV, TFIDF_RESULTS_CSV, TFIDF_FEATURES_NPY, GEN_DIR

print("=" * 60)
print("  STEP 2 — TF-IDF Feature Extraction (Baseline)")
print("=" * 60)

if not os.path.exists(CLEANED_CSV):
    raise FileNotFoundError(f"[ERROR] '{CLEANED_CSV}' not found — run preprocessing.py first.")

df = pd.read_csv(CLEANED_CSV)
df["clean_review"] = df["clean_review"].fillna("")
corpus = df["clean_review"].tolist()

print(f"  Documents loaded   : {len(corpus):,}")
print(f"  Output folder      : {GEN_DIR}/")

# ── Build TF-IDF matrix ───────────────────────────────────────────────
tfidf_vectorizer = TfidfVectorizer(
    max_features = 5000,
    ngram_range  = (1, 2),   # unigrams + bigrams ("machine learning")
    min_df       = 2,         # ignore terms in < 2 docs
    max_df       = 0.85,      # ignore terms in > 85% of docs
)

tfidf_matrix  = tfidf_vectorizer.fit_transform(corpus)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}")
print(f"  Vocabulary size     : {len(feature_names):,} features")

TOP_N = 10

def extract_tfidf_keywords(row_index: int, top_n: int = TOP_N):
    """Return top-N (keyword, score) pairs for one document."""
    row = tfidf_matrix[row_index].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    return [
        (feature_names[i], row[i])
        for i in top_indices if row[i] > 0
    ]

print(f"\n  Extracting top-{TOP_N} keywords per document...")

all_keywords = []
for i in range(len(df)):
    pairs = extract_tfidf_keywords(i, top_n=TOP_N)
    all_keywords.append({
        "document_id"   : i,
        "review"        : df.loc[i, "review"],
        "clean_review"  : df.loc[i, "clean_review"],
        "sentiment"     : df.loc[i, "sentiment"],
        "tfidf_keywords": [kw          for kw, _  in pairs],
        "tfidf_scores"  : [float(round(sc, 4)) for _, sc in pairs],  # plain float, not np.float64
    })

tfidf_results_df = pd.DataFrame(all_keywords)

# ── Save ──────────────────────────────────────────────────────────────
tfidf_results_df.to_csv(TFIDF_RESULTS_CSV, index=False)
np.save(TFIDF_FEATURES_NPY, feature_names)

print(f"  Saved -> {TFIDF_RESULTS_CSV}")
print(f"  Saved -> {TFIDF_FEATURES_NPY}")

print(f"\n  Sample — Document 0:")
print(f"    Review   : {df['review'].iloc[0][:70]}...")
print(f"    Keywords : {tfidf_results_df['tfidf_keywords'].iloc[0][:5]}")
print(f"    Scores   : {tfidf_results_df['tfidf_scores'].iloc[0][:5]}")

print("\n  STEP 2 COMPLETE")
print("=" * 60)