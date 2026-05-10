"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 3 : Word2Vec Embeddings                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Inputs  : generated/IMDB_cleaned.csv                                ║
║            generated/tfidf_keywords_results.csv                      ║
║  Outputs : generated/word2vec_model.model                            ║
║            generated/document_embeddings.npy                         ║
║            generated/keyword_embeddings.pkl                          ║
╚══════════════════════════════════════════════════════════════════════╝

WHY Word2Vec?
  TF-IDF treats every word as isolated — it has no idea "good" and
  "great" are similar. Word2Vec learns dense vectors where semantically
  related words are close in space, enabling meaning-based similarity.
"""

import os
import pickle
import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from paths import (CLEANED_CSV, TFIDF_RESULTS_CSV,
                   W2V_MODEL, DOC_EMBEDDINGS_NPY, KEYWORD_EMB_PKL, GEN_DIR)

print("=" * 60)
print("  STEP 3 — Word2Vec Embeddings")
print("=" * 60)

for f in [CLEANED_CSV, TFIDF_RESULTS_CSV]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"[ERROR] '{f}' not found — run previous steps first.")

df      = pd.read_csv(CLEANED_CSV)
df["clean_review"] = df["clean_review"].fillna("")

df_tfidf = pd.read_csv(TFIDF_RESULTS_CSV)
df_tfidf["tfidf_keywords"] = df_tfidf["tfidf_keywords"].apply(ast.literal_eval)

corpus = df["clean_review"].tolist()
print(f"  Documents loaded  : {len(df):,}")
print(f"  Output folder     : {GEN_DIR}/")

# ── Tokenize ──────────────────────────────────────────────────────────
tokenized_corpus = [review.split() for review in corpus]
print(f"  Total tokens      : {sum(len(t) for t in tokenized_corpus):,}")

# ── Train Word2Vec ────────────────────────────────────────────────────
"""
Hyperparameters:
  vector_size = 100  -> each word = 100-dim vector
  window      = 5    -> 5 words left + right context
  min_count   = 2    -> ignore words appearing < 2 times
  epochs      = 10   -> passes over corpus
  seed        = 42   -> reproducibility
"""
print("\n  Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences   = tokenized_corpus,
    vector_size = 100,
    window      = 5,
    min_count   = 2,
    workers     = 4,
    epochs      = 10,
    seed        = 42,
)
print(f"  Vocabulary size   : {len(w2v_model.wv):,} words")

# ── Document embeddings — average token vectors ────────────────────────
print("  Building document embeddings...")

def document_to_vector(tokens, model, dim=100):
    """Average Word2Vec vectors for all known tokens in a document."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else np.zeros(dim, dtype=np.float32)

doc_embeddings = np.array(
    [document_to_vector(tokens, w2v_model) for tokens in tokenized_corpus],
    dtype=np.float32,
)
print(f"  Document embeddings shape : {doc_embeddings.shape}")

# ── Keyword embeddings ────────────────────────────────────────────────
"""
For bigram keywords (e.g. "machine learning") we average the two word
vectors.  If neither word is in vocabulary we store None.
Structure: { doc_id : { "keyword_string" : np.array | None } }
"""
print("  Building keyword embeddings...")

def keyword_to_vector(keyword, model, dim=100):
    """Embed a keyword — handles unigrams and bigrams."""
    parts = keyword.split()
    vecs  = [model.wv[p] for p in parts if p in model.wv]
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else None

keyword_embeddings = {}
for _, row in df_tfidf.iterrows():
    doc_id   = int(row["document_id"])
    keywords = row["tfidf_keywords"]
    keyword_embeddings[doc_id] = {kw: keyword_to_vector(kw, w2v_model) for kw in keywords}

embedded = sum(1 for d in keyword_embeddings.values() for v in d.values() if v is not None)
oov      = sum(1 for d in keyword_embeddings.values() for v in d.values() if v is None)
print(f"  Embedded keywords : {embedded:,}  |  OOV (None) : {oov:,}")

# ── Save all outputs (once each) ──────────────────────────────────────
print("\n  Saving outputs...")
w2v_model.save(W2V_MODEL)
np.save(DOC_EMBEDDINGS_NPY, doc_embeddings)
with open(KEYWORD_EMB_PKL, "wb") as f:
    pickle.dump(keyword_embeddings, f)

print(f"  Saved -> {W2V_MODEL}")
print(f"  Saved -> {DOC_EMBEDDINGS_NPY}")
print(f"  Saved -> {KEYWORD_EMB_PKL}")

print(f"\n  Word2Vec sanity check — words similar to 'acting':")
try:
    for word, score in w2v_model.wv.most_similar("acting", topn=5):
        print(f"    {word:<20} {score:.4f}")
except KeyError:
    print("    ('acting' not in vocabulary)")

print("\n  STEP 3 COMPLETE")
print("=" * 60)