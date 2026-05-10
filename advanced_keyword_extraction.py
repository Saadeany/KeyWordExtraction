"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 4 : Advanced Embedding-Based Keyword Extraction        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Inputs  : generated/IMDB_cleaned.csv                                ║
║            generated/tfidf_keywords_results.csv                      ║
║            generated/word2vec_model.model                            ║
║            generated/document_embeddings.npy                         ║
║            generated/keyword_embeddings.pkl                          ║
║                                                                      ║
║  Methods:                                                            ║
║    A. Cosine Similarity Re-ranking  (semantic closeness to doc)      ║
║    B. Per-Document KMeans Clustering (diverse keyword selection)     ║
║    C. Global Corpus Clustering       (macro topic discovery)         ║
║                                                                      ║
║  Outputs : generated/advanced_keywords_results.csv                   ║
║            generated/global_cluster_topics.csv                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from paths import (CLEANED_CSV, TFIDF_RESULTS_CSV,
                   W2V_MODEL, DOC_EMBEDDINGS_NPY, KEYWORD_EMB_PKL,
                   ADVANCED_CSV, TOPICS_CSV, GEN_DIR)

warnings.filterwarnings("ignore")


def safe_parse(val):
    """
    Parse a list stored as string in a CSV column.
    Handles clean Python literals AND numpy repr like np.float64(0.12).
    """
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        cleaned = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', str(val))
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return json.loads(cleaned.replace("'", '"'))


# ── Guard: all required inputs must exist ─────────────────────────────
REQUIRED = [CLEANED_CSV, TFIDF_RESULTS_CSV, W2V_MODEL, DOC_EMBEDDINGS_NPY, KEYWORD_EMB_PKL]
for f in REQUIRED:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"[ERROR] '{f}' not found.\n"
            f"  -> Run the previous steps first."
        )

# ══════════════════════════════════════════════════════════════════════
# LOAD INPUTS
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 4 — Advanced Embedding-Based Keyword Extraction")
print("=" * 65)
print(f"  Output folder : {GEN_DIR}/")

df = pd.read_csv(CLEANED_CSV)
df["clean_review"] = df["clean_review"].fillna("")

df_tfidf = pd.read_csv(TFIDF_RESULTS_CSV)
df_tfidf["tfidf_keywords"] = df_tfidf["tfidf_keywords"].apply(safe_parse)
df_tfidf["tfidf_scores"]   = df_tfidf["tfidf_scores"].apply(safe_parse)

w2v_model      = Word2Vec.load(W2V_MODEL)
doc_embeddings = np.load(DOC_EMBEDDINGS_NPY)

with open(KEYWORD_EMB_PKL, "rb") as f:
    keyword_embeddings = pickle.load(f)

print(f"  Documents loaded       : {len(df):,}")
print(f"  Word2Vec vocabulary    : {len(w2v_model.wv):,} words")
print(f"  Document embeddings    : {doc_embeddings.shape}")
print(f"  Keyword-embedding docs : {len(keyword_embeddings):,}")
print()


# ══════════════════════════════════════════════════════════════════════
# METHOD A — COSINE SIMILARITY RE-RANKING
# ══════════════════════════════════════════════════════════════════════
"""
WHY: TF-IDF ignores meaning. Cosine similarity measures the angle
between a keyword vector and the document vector — keywords that are
semantically closest to the document's overall meaning rank highest.
"""
print("=" * 65)
print("  METHOD A — Cosine Similarity Re-ranking")
print("=" * 65)

doc_emb_norm = normalize(doc_embeddings, norm="l2")   # pre-normalise once

def cosine_rerank(doc_id: int, top_n: int = 10) -> list:
    """Re-rank TF-IDF keywords by cosine similarity to the doc vector."""
    doc_vec = doc_emb_norm[doc_id].reshape(1, -1)
    kw_dict = keyword_embeddings.get(doc_id, {})
    scored  = []
    for kw, kw_vec in kw_dict.items():
        if kw_vec is None:
            continue
        kw_norm = kw_vec / (np.linalg.norm(kw_vec) + 1e-10)
        scored.append((kw, float(doc_vec @ kw_norm)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

cosine_rows = []
for doc_id in tqdm(range(len(df)), desc="  Cosine re-ranking", unit="doc"):
    ranked = cosine_rerank(doc_id, top_n=10)
    cosine_rows.append({
        "document_id"    : doc_id,
        "cosine_keywords": [kw          for kw, _  in ranked],
        "cosine_scores"  : [round(sc, 4) for _, sc in ranked],
    })
df_cosine = pd.DataFrame(cosine_rows)
print(f"  Method A done — {len(df_cosine):,} documents\n")


# ══════════════════════════════════════════════════════════════════════
# METHOD B — PER-DOCUMENT KMEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════
"""
WHY: Cosine re-ranking can return near-synonyms wasting keyword slots.
Clustering forces diversity: each cluster = one semantic facet of the
document; we pick the single most-representative word per cluster.
"""
print("=" * 65)
print("  METHOD B — Per-Document KMeans Clustering")
print("=" * 65)

N_CLUSTERS = 5

def cluster_keywords(doc_id: int, n_clusters: int = N_CLUSTERS) -> list:
    """Return one representative keyword per cluster."""
    kw_dict = keyword_embeddings.get(doc_id, {})
    valid   = [(kw, vec) for kw, vec in kw_dict.items() if vec is not None]
    if not valid:
        return []
    k        = min(n_clusters, len(valid))
    keywords = [kw  for kw, _  in valid]
    vectors  = np.array([vec for _, vec in valid], dtype=np.float32)
    kmeans   = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(vectors)
    selected = []
    for ci in range(k):
        mask   = kmeans.labels_ == ci
        if not mask.any():
            continue
        c_vecs = vectors[mask]
        c_kws  = [keywords[i] for i, m in enumerate(mask) if m]
        sims   = cosine_similarity(kmeans.cluster_centers_[ci].reshape(1, -1), c_vecs)[0]
        selected.append((c_kws[sims.argmax()], float(sims.max())))
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected

cluster_rows = []
for doc_id in tqdm(range(len(df)), desc="  KMeans clustering ", unit="doc"):
    kw_scores = cluster_keywords(doc_id, n_clusters=N_CLUSTERS)
    cluster_rows.append({
        "document_id"     : doc_id,
        "cluster_keywords": [kw          for kw, _  in kw_scores],
        "cluster_scores"  : [round(sc, 4) for _, sc in kw_scores],
    })
df_cluster = pd.DataFrame(cluster_rows)
print(f"  Method B done — {len(df_cluster):,} documents\n")


# ══════════════════════════════════════════════════════════════════════
# METHOD C — GLOBAL CORPUS CLUSTERING (Topic Discovery)
# ══════════════════════════════════════════════════════════════════════
"""
WHY: Individual methods give per-review keywords.  Global clustering
reveals macro topics across the whole corpus.
"""
print("=" * 65)
print("  METHOD C — Global Corpus Clustering (Topic Discovery)")
print("=" * 65)

K_TOPICS           = 10
KEYWORDS_PER_TOPIC = 15

print(f"  Running KMeans  K={K_TOPICS}  on {doc_embeddings.shape[0]:,} documents...")
global_kmeans = KMeans(n_clusters=K_TOPICS, random_state=42, n_init=10, max_iter=500)
global_kmeans.fit(doc_embeddings)
doc_labels = global_kmeans.labels_

print(f"  Cluster distribution:")
for cl, cnt in zip(*np.unique(doc_labels, return_counts=True)):
    print(f"    Topic {cl:2d}  ->  {cnt:5,} documents")
print()

topic_rows = []
for topic_id in range(K_TOPICS):
    doc_ids  = np.where(doc_labels == topic_id)[0]
    kw_freq  = {}
    for did in doc_ids:
        kws = df_tfidf.loc[df_tfidf["document_id"] == did, "tfidf_keywords"]
        if kws.empty:
            continue
        for kw in kws.iloc[0]:
            kw_freq[kw] = kw_freq.get(kw, 0) + 1
    top_kws   = sorted(kw_freq.items(), key=lambda x: x[1], reverse=True)[:KEYWORDS_PER_TOPIC]
    centroid  = global_kmeans.cluster_centers_[topic_id].reshape(1, -1)
    sims      = cosine_similarity(centroid, doc_embeddings[doc_ids])[0]
    rep_idx   = doc_ids[sims.argmax()]
    snippet   = df.loc[rep_idx, "review"][:120].replace("\n", " ")
    topic_rows.append({
        "topic_id"              : topic_id,
        "num_documents"         : len(doc_ids),
        "top_keywords"          : [kw   for kw, _ in top_kws],
        "keyword_frequencies"   : [freq for _, freq in top_kws],
        "representative_snippet": snippet + "...",
    })
    print(f"  Topic {topic_id:2d}  ({len(doc_ids):,} docs)")
    print(f"    Keywords : {[kw for kw, _ in top_kws[:8]]}")
    print(f"    Snippet  : {snippet[:80]}...")
    print()

df_topics = pd.DataFrame(topic_rows)


# ══════════════════════════════════════════════════════════════════════
# MERGE & SAVE
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  SAVING RESULTS")
print("=" * 65)

df_final = (
    df_tfidf[["document_id", "review", "clean_review", "sentiment",
              "tfidf_keywords", "tfidf_scores"]]
    .merge(df_cosine,  on="document_id")
    .merge(df_cluster, on="document_id")
)
df_final["global_topic"] = doc_labels

df_final.to_csv(ADVANCED_CSV, index=False)
df_topics.to_csv(TOPICS_CSV,  index=False)

print(f"  Saved -> {ADVANCED_CSV}")
print(f"  Saved -> {TOPICS_CSV}")

print("\n  Sample — 3 documents:")
for doc_id in range(3):
    row = df_final[df_final["document_id"] == doc_id].iloc[0]
    print(f"\n  Doc {doc_id} | {df.loc[doc_id, 'sentiment']}")
    print(f"    TF-IDF    -> {row['tfidf_keywords'][:5]}")
    print(f"    Cosine    -> {row['cosine_keywords'][:5]}")
    print(f"    KMeans    -> {row['cluster_keywords'][:5]}")
    print(f"    Topic     -> Topic {row['global_topic']}")

print("\n  STEP 4 COMPLETE")
print("=" * 65)