"""
╔══════════════════════════════════════════════════════════════════════╗
║ PROJECT 5 — KEYWORD EXTRACTION SYSTEM                               ║
║ BONUS : TextRank — Graph-Based Keyword Extraction                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ What is TextRank?                                                    ║
║   A graph-based algorithm inspired by Google PageRank.              ║
║   Words are nodes; edges connect words that co-occur in a           ║
║   sliding window. PageRank then scores each word by how             ║
║   many important neighbors it has — no corpus statistics,           ║
║   no embeddings, no labels needed.                                  ║
║                                                                      ║
║ Why is it better than the previous advanced model?                  ║
║   • KMeans / Cosine methods depend on Word2Vec quality              ║
║     (limited by corpus size and OOV words).                         ║
║   • TextRank works on the document itself — it finds words          ║
║     that are structurally central to the text's meaning,            ║
║     not just statistically frequent or embedding-close.             ║
║   • It naturally handles multi-word phrases (collects               ║
║     adjacent top-ranked words into keyphrases).                     ║
║                                                                      ║
║ Inputs  : generated/IMDB_cleaned.csv                                ║
║           generated/advanced_keywords_results.csv  (for comparison) ║
║ Outputs : generated/textrank_keywords_results.csv                   ║
║           evaluation_results/textrank_comparison.png                ║
║           evaluation_results/textrank_before_after.png              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import CLEANED_CSV, ADVANCED_CSV, GEN_DIR, EVAL_DIR

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 4       # co-occurrence window for graph edges
TOP_N         = 5       # keywords per document (same as evaluation.py)
SAMPLE_SIZE   = 25      # must match evaluation.py
RANDOM_STATE  = 42

OUTPUT_CSV    = os.path.join(GEN_DIR, "textrank_keywords_results.csv")
FIG_COMPARE   = os.path.join(EVAL_DIR, "textrank_comparison.png")
FIG_BEFORE_AFTER = os.path.join(EVAL_DIR, "textrank_before_after.png")

os.makedirs(GEN_DIR,  exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Same manual ground truth as evaluation.py
MANUAL_ANNOTATIONS = {
    0:  ["summerslam","luger","yokozuna","wwf","wrestling"],
    1:  ["farscape","television","miniseries","drama","fans"],
    2:  ["seagal","chase","hijacking","action","destruction"],
    3:  ["emma","paltrow","austen","british","gwyneth"],
    4:  ["carell","anchorman","virgin","sunshine","comedy"],
    5:  ["tommy","farley","callahan","comedy","spade"],
    6:  ["favela","rising","documentary","music","brazil"],
    7:  ["purgatory","tony","sopranos","episode","inferno"],
    8:  ["graphics","documentary","series","detail","audience"],
    9:  ["tinseltown","pantoliano","writers","storage","bland"],
    10: ["schoyen","actor","norwegian","english","boring"],
    11: ["sharky","reynolds","narcotics","atlanta","cop"],
    12: ["dinosaur","animation","cgi","raptor","terrible"],
    13: ["anime","japan","graphics","plot","animation"],
    14: ["acting","york","hollywood","culture","seinfeld"],
    15: ["dostoyevsky","punishment","tragedy","philosophy","character"],
    16: ["heist","diner","kidnap","noir","gang"],
    17: ["waters","flamingos","johnson","maryland","trailer"],
    18: ["science","fiction","visuals","music","future"],
    19: ["graves","acting","clark","performance","doctor"],
    20: ["williams","waterbury","england","production","score"],
    21: ["documentary","revelations","channel","letterman","bill"],
    22: ["butchers","flesh","horror","allegory","dark"],
    23: ["kristine","anderson","explicit","career","producers"],
    24: ["moonwalker","jackson","biography","wiz","documentary"],
}

C_TFIDF   = "#4C72B0"
C_COSINE  = "#DD8452"
C_CLUSTER = "#55A868"
C_TEXTRANK= "#9467BD"   # distinct purple for the new model
TSIZE, LSIZE = 14, 12


# ══════════════════════════════════════════════════════════════════════
# TEXTRANK IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════

def build_cooccurrence_graph(tokens: list, window: int = WINDOW_SIZE) -> nx.Graph:
    """
    Build an undirected weighted graph:
      - Nodes  : unique tokens
      - Edges  : tokens that appear within `window` positions of each other
      - Weight : number of times a pair co-occurs (accumulated)
    """
    graph = nx.Graph()
    graph.add_nodes_from(set(tokens))

    for i, token in enumerate(tokens):
        for j in range(i + 1, min(i + window + 1, len(tokens))):
            neighbor = tokens[j]
            if token == neighbor:
                continue
            if graph.has_edge(token, neighbor):
                graph[token][neighbor]["weight"] += 1
            else:
                graph.add_edge(token, neighbor, weight=1)

    return graph


def textrank_keywords(text: str, top_n: int = TOP_N,
                      window: int = WINDOW_SIZE,
                      alpha: float = 0.85) -> list:
    """
    Full TextRank pipeline for a single document:
      1. Tokenise (split on whitespace — text is already pre-cleaned)
      2. Build co-occurrence graph
      3. Run PageRank (alpha = damping factor, same as original paper)
      4. Collect adjacent top-ranked tokens as keyphrases
      5. Return top_n (keyword, score) pairs

    Returns [] for documents that are too short to build a graph.
    """
    tokens = text.strip().split()
    if len(tokens) < 3:
        return []

    graph = build_cooccurrence_graph(tokens, window)

    if len(graph) == 0 or graph.number_of_edges() == 0:
        return []

    try:
        scores = nx.pagerank(graph, alpha=alpha, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(graph, alpha=alpha, weight=None, max_iter=500)

    # ── Keyphrase merging ─────────────────────────────────────────────
    # After ranking individual words, merge consecutive tokens that are
    # BOTH in the top-50% of PageRank scores into a single keyphrase.
    # This is how TextRank produces multi-word terms (e.g. "machine learning").
    score_threshold = np.percentile(list(scores.values()), 50)
    top_tokens = {t for t, s in scores.items() if s >= score_threshold}

    keyphrases: dict = {}
    i = 0
    while i < len(tokens):
        if tokens[i] in top_tokens:
            phrase_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tokens[j] in top_tokens:
                phrase_tokens.append(tokens[j])
                j += 1
            phrase = " ".join(phrase_tokens)
            phrase_score = sum(scores.get(t, 0) for t in phrase_tokens) / len(phrase_tokens)
            # Keep the best score if a phrase appears multiple times
            if phrase not in keyphrases or keyphrases[phrase] < phrase_score:
                keyphrases[phrase] = phrase_score
            i = j
        else:
            i += 1

    if not keyphrases:
        # Fall back to top individual words
        keyphrases = {t: s for t, s in scores.items()}

    ranked = sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print("=" * 65)
print(" BONUS — TextRank Graph-Based Keyword Extraction")
print("=" * 65)

for f in [CLEANED_CSV, ADVANCED_CSV]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"[ERROR] '{f}' not found.\n"
            f" -> Run all 5 pipeline steps first."
        )

df = pd.read_csv(CLEANED_CSV)
df["clean_review"] = df["clean_review"].fillna("")

df_adv = pd.read_csv(ADVANCED_CSV)
for col in ["tfidf_keywords", "cosine_keywords", "cluster_keywords"]:
    df_adv[col] = df_adv[col].apply(ast.literal_eval)

print(f" Loaded {len(df):,} documents")
print(f" Output folder : {GEN_DIR}/  |  {EVAL_DIR}/\n")


# ══════════════════════════════════════════════════════════════════════
# EXTRACT TEXTRANK KEYWORDS FOR ALL DOCUMENTS
# ══════════════════════════════════════════════════════════════════════

print(" Extracting TextRank keywords...")
tr_rows = []
for i, row in df.iterrows():
    pairs = textrank_keywords(row["clean_review"], top_n=TOP_N)
    tr_rows.append({
        "document_id"       : i,
        "textrank_keywords" : [kw for kw, _ in pairs],
        "textrank_scores"   : [round(float(sc), 4) for _, sc in pairs],
    })

df_tr = pd.DataFrame(tr_rows)
df_tr.to_csv(OUTPUT_CSV, index=False)
print(f" Saved -> {OUTPUT_CSV}")
print(f" Sample — Document 0 : {df_tr['textrank_keywords'].iloc[0][:5]}\n")


# ══════════════════════════════════════════════════════════════════════
# EVALUATION  (same 25-doc sample + same ground truth as evaluation.py)
# ══════════════════════════════════════════════════════════════════════

print("=" * 65)
print(" EVALUATION — Comparing all 4 methods")
print("=" * 65)

df_sample = df_adv.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
manual_ref = [MANUAL_ANNOTATIONS[i] for i in range(SAMPLE_SIZE)]

# Pull keywords for all 4 methods
tfidf_kws   = [r[:TOP_N] for r in df_sample["tfidf_keywords"]]
cosine_kws  = [r[:TOP_N] for r in df_sample["cosine_keywords"]]
cluster_kws = [r[:TOP_N] for r in df_sample["cluster_keywords"]]
textrank_kws = [
    df_tr.loc[df_tr["document_id"] == df_sample.iloc[i]["document_id"],
              "textrank_keywords"].values[0][:TOP_N]
    if not df_tr.loc[df_tr["document_id"] == df_sample.iloc[i]["document_id"]].empty
    else []
    for i in range(SAMPLE_SIZE)
]

# ── Metric helpers (identical to evaluation.py) ───────────────────────

def word_set(kw_list):
    return set(" ".join(kw_list).split())

def precision_n(pred, ref):
    pw, rw = word_set(pred), set(ref)
    return len(pw & rw) / len(pw) if pw else 0.0

def recall_n(pred, ref):
    pw, rw = word_set(pred), set(ref)
    return len(pw & rw) / len(rw) if rw else 0.0

def f1_n(pred, ref):
    p, r = precision_n(pred, ref), recall_n(pred, ref)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def compute_metrics(preds, refs, label):
    P = [precision_n(p, r) for p, r in zip(preds, refs)]
    R = [recall_n(p, r)    for p, r in zip(preds, refs)]
    F = [f1_n(p, r)        for p, r in zip(preds, refs)]
    print(f"\n [{label}]")
    print(f"   Avg Precision : {np.mean(P):.4f}")
    print(f"   Avg Recall    : {np.mean(R):.4f}")
    print(f"   Avg F1-Score  : {np.mean(F):.4f}")
    print(f"   Median F1     : {np.median(F):.4f}")
    return np.mean(P), np.mean(R), np.mean(F), P, R, F

def build_cm_arrays(preds, refs):
    y_true, y_pred = [], []
    for pred, ref in zip(preds, refs):
        pw, rw = word_set(pred), set(ref)
        for kw in ref:
            y_true.append(1); y_pred.append(1 if kw in pw else 0)
        for kw in pw:
            if kw not in rw:
                y_true.append(0); y_pred.append(1)
    return np.array(y_true), np.array(y_pred)

def accuracy_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return (tp + tn) / cm.sum() if cm.sum() else 0.0


print("\n PRECISION / RECALL / F1")
print("=" * 65)

t_p,  t_r,  t_f,  tP,  tR,  tF  = compute_metrics(tfidf_kws,    manual_ref, "TF-IDF Baseline      (BEFORE)")
c_p,  c_r,  c_f,  cP,  cR,  cF  = compute_metrics(cosine_kws,   manual_ref, "Cosine Re-ranking    (BEFORE)")
k_p,  k_r,  k_f,  kP,  kR,  kF  = compute_metrics(cluster_kws,  manual_ref, "KMeans Clustering    (BEFORE)")
tr_p, tr_r, tr_f, trP, trR, trF = compute_metrics(textrank_kws, manual_ref, "TextRank Graph-Based (AFTER )")

t_yt,  t_yp  = build_cm_arrays(tfidf_kws,    manual_ref)
c_yt,  c_yp  = build_cm_arrays(cosine_kws,   manual_ref)
k_yt,  k_yp  = build_cm_arrays(cluster_kws,  manual_ref)
tr_yt, tr_yp = build_cm_arrays(textrank_kws, manual_ref)

t_cm  = confusion_matrix(t_yt,  t_yp)
c_cm  = confusion_matrix(c_yt,  c_yp)
k_cm  = confusion_matrix(k_yt,  k_yp)
tr_cm = confusion_matrix(tr_yt, tr_yp)

t_acc  = accuracy_from_cm(t_cm)
c_acc  = accuracy_from_cm(c_cm)
k_acc  = accuracy_from_cm(k_cm)
tr_acc = accuracy_from_cm(tr_cm)

# ── BEFORE / AFTER summary table ──────────────────────────────────────

print("\n" + "=" * 65)
print(" BEFORE vs AFTER — Summary Table")
print("=" * 65)

summary = pd.DataFrame({
    "Model"    : ["TF-IDF Baseline", "Cosine Re-ranking", "KMeans Clustering", "TextRank (NEW)"],
    "Stage"    : ["BEFORE", "BEFORE", "BEFORE", "AFTER"],
    "Accuracy" : [round(t_acc, 4), round(c_acc, 4), round(k_acc, 4), round(tr_acc, 4)],
    "Precision": [round(t_p, 4),   round(c_p, 4),   round(k_p, 4),   round(tr_p, 4)],
    "Recall"   : [round(t_r, 4),   round(c_r, 4),   round(k_r, 4),   round(tr_r, 4)],
    "F1-Score" : [round(t_f, 4),   round(c_f, 4),   round(k_f, 4),   round(tr_f, 4)],
    "Median F1": [round(float(np.median(tF)),  4), round(float(np.median(cF)),  4),
                  round(float(np.median(kF)),  4), round(float(np.median(trF)), 4)],
})

print(summary.to_string(index=False))
best_f1 = summary["F1-Score"].max()
winner  = summary[summary["F1-Score"] == best_f1].iloc[0]["Model"]
print(f"\n Winner by F1 : {winner}  (F1 = {best_f1:.4f})")

# TextRank improvement over the best BEFORE method
before_best_f1 = max(t_f, c_f, k_f)
delta_f1 = tr_f - before_best_f1
sign = "+" if delta_f1 >= 0 else ""
print(f" TextRank improvement over best previous model : {sign}{delta_f1:.4f} F1")


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — All 4 methods side-by-side bar (Precision / Recall / F1)
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 5))
x  = np.arange(3)
bw = 0.18

vals = [
    ([t_p,  t_r,  t_f],  C_TFIDF,    "TF-IDF Baseline (BEFORE)"),
    ([c_p,  c_r,  c_f],  C_COSINE,   "Cosine Re-ranking (BEFORE)"),
    ([k_p,  k_r,  k_f],  C_CLUSTER,  "KMeans Clustering (BEFORE)"),
    ([tr_p, tr_r, tr_f], C_TEXTRANK, "TextRank Graph-Based (AFTER)"),
]

offsets = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]
for (v, color, label), offset in zip(vals, offsets):
    bars = ax.bar(x + offset, v, bw, label=label, color=color, edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")

ax.set_ylim(0, 0.75)
ax.set_xticks(x)
ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontsize=LSIZE)
ax.set_ylabel("Score", fontsize=LSIZE)
ax.set_title("All 4 Methods — Precision / Recall / F1\n"
             "BEFORE (TF-IDF, Cosine, KMeans)  vs  AFTER (+ TextRank)",
             fontsize=TSIZE, fontweight="bold")
ax.legend(fontsize=9, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_COMPARE, dpi=150)
plt.close()
print(f"\n Saved -> {FIG_COMPARE}")


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — BEFORE vs AFTER highlight chart
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: F1 per document — best BEFORE vs TextRank ---
ax = axes[0]
x  = np.arange(SAMPLE_SIZE)
bw = 0.35

# "Best BEFORE" = element-wise max of tF, cF, kF
best_before_per_doc = [max(tF[i], cF[i], kF[i]) for i in range(SAMPLE_SIZE)]

ax.bar(x - bw / 2, best_before_per_doc, bw,
       label="Best BEFORE (per doc)",
       color="#bbbbbb", edgecolor="white", alpha=0.9)
ax.bar(x + bw / 2, trF, bw,
       label="TextRank AFTER",
       color=C_TEXTRANK, edgecolor="white", alpha=0.9)

ax.axhline(np.mean(best_before_per_doc),
           color="#888888", ls="--", lw=1.5,
           label=f"BEFORE mean ({np.mean(best_before_per_doc):.3f})")
ax.axhline(np.mean(trF),
           color=C_TEXTRANK, ls="--", lw=1.5,
           label=f"TextRank mean ({np.mean(trF):.3f})")

ax.set_xticks(x)
ax.set_xticklabels([f"D{i+1}" for i in range(SAMPLE_SIZE)],
                   fontsize=7, rotation=45)
ax.set_ylabel("F1-Score", fontsize=LSIZE)
ax.set_ylim(0, 1.0)
ax.set_title("Per-Document F1\nBest BEFORE vs TextRank AFTER",
             fontsize=TSIZE, fontweight="bold")
ax.legend(fontsize=9, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

# --- Right: Radar chart — all 4 methods ---
ax2 = axes[1]
cats   = ["Precision", "Recall", "F1-Score", "Accuracy"]
angles = [n / 4 * 2 * np.pi for n in range(4)] + [0]
ax2    = fig.add_subplot(122, polar=True)

for v, color, label in [
    ([t_p,  t_r,  t_f,  t_acc],  C_TFIDF,    "TF-IDF (BEFORE)"),
    ([c_p,  c_r,  c_f,  c_acc],  C_COSINE,   "Cosine (BEFORE)"),
    ([k_p,  k_r,  k_f,  k_acc],  C_CLUSTER,  "KMeans (BEFORE)"),
    ([tr_p, tr_r, tr_f, tr_acc], C_TEXTRANK, "TextRank (AFTER)"),
]:
    vals_closed = v + v[:1]
    lw = 3.0 if "AFTER" in label else 1.5
    ax2.plot(angles, vals_closed, color=color, lw=lw, label=label)
    ax2.fill(angles, vals_closed, color=color, alpha=0.08 if "BEFORE" in label else 0.18)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(cats, fontsize=11)
ax2.set_ylim(0, 0.7)
ax2.set_title("Radar: BEFORE vs AFTER\n(TextRank highlighted)",
              fontsize=TSIZE, fontweight="bold", pad=22)
ax2.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15), fontsize=9)

plt.suptitle("BEFORE vs AFTER — Adding TextRank to the System",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_BEFORE_AFTER, dpi=150, bbox_inches="tight")
plt.close()
print(f" Saved -> {FIG_BEFORE_AFTER}")

print("\n TextRank COMPLETE")
print("=" * 65)
