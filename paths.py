"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          paths.py  —  Central Path Configuration                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Every script imports from here.                                     ║
║  To change a folder name, edit ONLY this file.                      ║
╚══════════════════════════════════════════════════════════════════════╝

Folder layout
─────────────
  project/
  ├── IMDB-Dataset.csv              ← place dataset here
  ├── main.py
  ├── paths.py                      ← this file
  ├── preprocessing.py
  ├── TF_IDF.py
  ├── word_embeddings.py
  ├── advanced_keyword_extraction.py
  ├── evaluation.py
  │
  ├── generated/                    ← auto-created: intermediate files
  │   ├── IMDB_cleaned.csv
  │   ├── tfidf_keywords_results.csv
  │   ├── tfidf_feature_names.npy
  │   ├── word2vec_model.model
  │   ├── document_embeddings.npy
  │   ├── keyword_embeddings.pkl
  │   ├── advanced_keywords_results.csv
  │   └── global_cluster_topics.csv
  │
  └── evaluation_results/           ← auto-created: charts + final CSV
      ├── evaluation_results.csv
      ├── fig1_summary_bar.png
      ├── fig2_per_doc_f1.png
      ├── fig3_distribution.png
      ├── fig4_per_doc_tp_fp_fn.png
      ├── fig5_tp_fp_fn_total.png
      ├── fig6_manual_table.png
      ├── fig7_radar.png
      ├── fig8_sentiment_f1.png
      └── fig9_confusion_matrix.png
"""

import os

# ── Folder names ──────────────────────────────────────────────────────
GEN_DIR   = "generated"           # intermediate pipeline files
EVAL_DIR  = "evaluation_results"  # charts and final evaluation CSV

# ── Create folders if they don't exist ───────────────────────────────
os.makedirs(GEN_DIR,  exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# INPUT (must be placed in project root by the user)
# ══════════════════════════════════════════════════════════════════════
DATASET = "IMDB-Dataset.csv"


# ══════════════════════════════════════════════════════════════════════
# GENERATED / INTERMEDIATE FILES  →  generated/
# ══════════════════════════════════════════════════════════════════════
CLEANED_CSV         = os.path.join(GEN_DIR, "IMDB_cleaned.csv")
TFIDF_RESULTS_CSV   = os.path.join(GEN_DIR, "tfidf_keywords_results.csv")
TFIDF_FEATURES_NPY  = os.path.join(GEN_DIR, "tfidf_feature_names.npy")
W2V_MODEL           = os.path.join(GEN_DIR, "word2vec_model.model")
DOC_EMBEDDINGS_NPY  = os.path.join(GEN_DIR, "document_embeddings.npy")
KEYWORD_EMB_PKL     = os.path.join(GEN_DIR, "keyword_embeddings.pkl")
ADVANCED_CSV        = os.path.join(GEN_DIR, "advanced_keywords_results.csv")
TOPICS_CSV          = os.path.join(GEN_DIR, "global_cluster_topics.csv")


# ══════════════════════════════════════════════════════════════════════
# EVALUATION OUTPUTS  →  evaluation_results/
# ══════════════════════════════════════════════════════════════════════
EVAL_RESULTS_CSV = os.path.join(EVAL_DIR, "evaluation_results.csv")

EVAL_FIGS = {
    "fig1" : os.path.join(EVAL_DIR, "fig1_summary_bar.png"),
    "fig2" : os.path.join(EVAL_DIR, "fig2_per_doc_f1.png"),
    "fig3" : os.path.join(EVAL_DIR, "fig3_distribution.png"),
    "fig4" : os.path.join(EVAL_DIR, "fig4_per_doc_tp_fp_fn.png"),
    "fig5" : os.path.join(EVAL_DIR, "fig5_tp_fp_fn_total.png"),
    "fig6" : os.path.join(EVAL_DIR, "fig6_manual_table.png"),
    "fig7" : os.path.join(EVAL_DIR, "fig7_radar.png"),
    "fig8" : os.path.join(EVAL_DIR, "fig8_sentiment_f1.png"),
    "fig9" : os.path.join(EVAL_DIR, "fig9_confusion_matrix.png"),
}

# ── All intermediate files in one list (used by main.py for status) ──
ALL_INTERMEDIATE = [
    CLEANED_CSV, TFIDF_RESULTS_CSV, TFIDF_FEATURES_NPY,
    W2V_MODEL, DOC_EMBEDDINGS_NPY, KEYWORD_EMB_PKL,
    ADVANCED_CSV, TOPICS_CSV,
]
ALL_EVAL_OUTPUTS = [EVAL_RESULTS_CSV] + list(EVAL_FIGS.values())