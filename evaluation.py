"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 5 : Evaluation & Comparison                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Input  : generated/advanced_keywords_results.csv                    ║
║                                                                      ║
║  Evaluates 3 methods against manual annotations:                     ║
║    1. TF-IDF Baseline                                                ║
║    2. Cosine Similarity Re-ranking  (Advanced A)                     ║
║    3. KMeans Clustering             (Advanced B)                     ║
║                                                                      ║
║  Metrics:                                                            ║
║    Precision / Recall / F1  |  Confusion Matrix  |  TP/FP/FN        ║
║                                                                      ║
║  Outputs -> evaluation_results/                                      ║
║    evaluation_results.csv  +  fig1 ... fig9 .png                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import ADVANCED_CSV, EVAL_RESULTS_CSV, EVAL_FIGS, EVAL_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
SAMPLE_SIZE  = 25
TOP_N        = 5
RANDOM_STATE = 42
C_TFIDF      = "#4C72B0"
C_COSINE     = "#DD8452"
C_CLUSTER    = "#55A868"
TSIZE, LSIZE = 14, 12

# ══════════════════════════════════════════════════════════════════════
# MANUAL ANNOTATIONS  (ground truth for the 25 sampled documents)
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# LOAD advanced_keywords_results.csv  (built by Step 4)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 5 — Evaluation & Model Comparison")
print("=" * 60)
print(f"  Output folder : {EVAL_DIR}/")

if not os.path.exists(ADVANCED_CSV):
    raise FileNotFoundError(
        f"[ERROR] '{ADVANCED_CSV}' not found.\n"
        "  -> Run advanced_keyword_extraction.py (Step 4) first."
    )

df_adv = pd.read_csv(ADVANCED_CSV)
for col in ["tfidf_keywords","tfidf_scores","cosine_keywords","cosine_scores",
            "cluster_keywords","cluster_scores"]:
    df_adv[col] = df_adv[col].apply(ast.literal_eval)

print(f"  Loaded {ADVANCED_CSV}  ({len(df_adv):,} rows)")

df_sample   = df_adv.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
tfidf_kws   = [r[:TOP_N] for r in df_sample["tfidf_keywords"]]
cosine_kws  = [r[:TOP_N] for r in df_sample["cosine_keywords"]]
cluster_kws = [r[:TOP_N] for r in df_sample["cluster_keywords"]]
manual_ref  = [MANUAL_ANNOTATIONS[i] for i in range(SAMPLE_SIZE)]
sentiments  = df_sample["sentiment"].tolist()

print(f"  Sample size   : {SAMPLE_SIZE} documents")
print(f"  Keywords/doc  : top-{TOP_N}")
print(f"  Methods       : TF-IDF | Cosine Re-ranking | KMeans Clustering")

# ══════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════
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
    return 2*p*r/(p+r) if (p+r) else 0.0

def compute_metrics(preds, refs, label):
    P = [precision_n(p,r) for p,r in zip(preds,refs)]
    R = [recall_n(p,r)    for p,r in zip(preds,refs)]
    F = [f1_n(p,r)        for p,r in zip(preds,refs)]
    print(f"\n  [{label}]")
    print(f"    Avg Precision : {np.mean(P):.4f}")
    print(f"    Avg Recall    : {np.mean(R):.4f}")
    print(f"    Avg F1-Score  : {np.mean(F):.4f}")
    print(f"    Median F1     : {np.median(F):.4f}")
    print(f"    Std F1        : {np.std(F):.4f}")
    return np.mean(P), np.mean(R), np.mean(F), P, R, F

def count_tp_fp_fn(preds, refs):
    TP=FP=FN=0; tp_d,fp_d,fn_d=[],[],[]
    for pred,ref in zip(preds,refs):
        pw,rw = word_set(pred), set(ref)
        tp_d.append(len(pw&rw)); fp_d.append(len(pw-rw)); fn_d.append(len(rw-pw))
        TP+=tp_d[-1]; FP+=fp_d[-1]; FN+=fn_d[-1]
    return TP,FP,FN,tp_d,fp_d,fn_d

def build_cm_arrays(preds, refs):
    y_true,y_pred=[],[]
    for pred,ref in zip(preds,refs):
        pw,rw = word_set(pred), set(ref)
        for kw in ref:
            y_true.append(1); y_pred.append(1 if kw in pw else 0)
        for kw in pw:
            if kw not in rw:
                y_true.append(0); y_pred.append(1)
    return np.array(y_true), np.array(y_pred)

# ══════════════════════════════════════════════════════════════════════
# COMPUTE METRICS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PRECISION / RECALL / F1")
print("=" * 60)

t_p,t_r,t_f,tP,tR,tF = compute_metrics(tfidf_kws,   manual_ref, "TF-IDF Baseline")
c_p,c_r,c_f,cP,cR,cF = compute_metrics(cosine_kws,  manual_ref, "Cosine Re-ranking (Advanced A)")
k_p,k_r,k_f,kP,kR,kF = compute_metrics(cluster_kws, manual_ref, "KMeans Clustering (Advanced B)")

t_TP,t_FP,t_FN,t_tp_d,t_fp_d,t_fn_d = count_tp_fp_fn(tfidf_kws,   manual_ref)
c_TP,c_FP,c_FN,c_tp_d,c_fp_d,c_fn_d = count_tp_fp_fn(cosine_kws,  manual_ref)
k_TP,k_FP,k_FN,k_tp_d,k_fp_d,k_fn_d = count_tp_fp_fn(cluster_kws, manual_ref)

print(f"\n  TF-IDF Baseline   -> TP={t_TP}  FP={t_FP}  FN={t_FN}")
print(f"  Cosine Re-ranking -> TP={c_TP}  FP={c_FP}  FN={c_FN}")
print(f"  KMeans Clustering -> TP={k_TP}  FP={k_FP}  FN={k_FN}")

t_yt,t_yp = build_cm_arrays(tfidf_kws,   manual_ref)
c_yt,c_yp = build_cm_arrays(cosine_kws,  manual_ref)
k_yt,k_yp = build_cm_arrays(cluster_kws, manual_ref)
t_cm = confusion_matrix(t_yt,t_yp)
c_cm = confusion_matrix(c_yt,c_yp)
k_cm = confusion_matrix(k_yt,k_yp)

# ══════════════════════════════════════════════════════════════════════
# MANUAL EVALUATION TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MANUAL EVALUATION TABLE")
print("=" * 60)

rows = []
for i in range(SAMPLE_SIZE):
    rows.append({
        "Doc"             : i+1,
        "Sentiment"       : sentiments[i],
        "Snippet"         : df_sample["review"].iloc[i][:70].replace("\n"," ")+"...",
        "TF-IDF Keywords" : ", ".join(tfidf_kws[i]),
        "Cosine Keywords" : ", ".join(cosine_kws[i]),
        "Cluster Keywords": ", ".join(cluster_kws[i]),
        "Manual Reference": ", ".join(manual_ref[i]),
        "TF-IDF F1"       : round(tF[i],3),
        "Cosine F1"       : round(cF[i],3),
        "Cluster F1"      : round(kF[i],3),
    })

man_df = pd.DataFrame(rows)
print(man_df[["Doc","TF-IDF Keywords","Cosine Keywords",
              "Cluster Keywords","Manual Reference",
              "TF-IDF F1","Cosine F1","Cluster F1"]].to_string(index=False))

man_df.to_csv(EVAL_RESULTS_CSV, index=False)
print(f"\n  Saved -> {EVAL_RESULTS_CSV}")

# ══════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GENERATING CHARTS")
print("=" * 60)

# ── Fig 1 : Summary bar ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
x  = np.arange(3); bw = 0.25
b1 = ax.bar(x-bw,  [t_p,t_r,t_f], bw, label="TF-IDF Baseline",       color=C_TFIDF,   edgecolor="white")
b2 = ax.bar(x,     [c_p,c_r,c_f], bw, label="Cosine Re-ranking (A)",  color=C_COSINE,  edgecolor="white")
b3 = ax.bar(x+bw,  [k_p,k_r,k_f], bw, label="KMeans Clustering (B)",  color=C_CLUSTER, edgecolor="white")
for b in list(b1)+list(b2)+list(b3):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
            f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0,0.65); ax.set_xticks(x)
ax.set_xticklabels(["Precision","Recall","F1-Score"], fontsize=LSIZE)
ax.set_ylabel("Score",fontsize=LSIZE)
ax.set_title("All Methods — Precision / Recall / F1\nGround Truth: Manual Annotation (25 docs)",
             fontsize=TSIZE, fontweight="bold")
ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig(EVAL_FIGS["fig1"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig1']}")

# ── Fig 2 : Per-document F1 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16,5))
x = np.arange(SAMPLE_SIZE); bw = 0.25
ax.bar(x-bw, tF, bw, label="TF-IDF Baseline",       color=C_TFIDF,   alpha=0.85)
ax.bar(x,    cF, bw, label="Cosine Re-ranking (A)",  color=C_COSINE,  alpha=0.85)
ax.bar(x+bw, kF, bw, label="KMeans Clustering (B)",  color=C_CLUSTER, alpha=0.85)
ax.axhline(np.mean(tF), color=C_TFIDF,   ls="--", lw=1.5, label=f"TF-IDF mean ({np.mean(tF):.3f})")
ax.axhline(np.mean(cF), color=C_COSINE,  ls="--", lw=1.5, label=f"Cosine mean ({np.mean(cF):.3f})")
ax.axhline(np.mean(kF), color=C_CLUSTER, ls="--", lw=1.5, label=f"Cluster mean ({np.mean(kF):.3f})")
ax.set_xticks(x); ax.set_xticklabels([f"D{i+1}" for i in range(SAMPLE_SIZE)],fontsize=7,rotation=45)
ax.set_ylabel("F1-Score",fontsize=LSIZE); ax.set_ylim(0,1.0)
ax.set_title("Per-Document F1 — All 3 Methods vs Manual Ground Truth",
             fontsize=TSIZE,fontweight="bold")
ax.legend(fontsize=9,ncol=3); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",alpha=0.25); plt.tight_layout()
plt.savefig(EVAL_FIGS["fig2"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig2']}")

# ── Fig 3 : Box plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,6))
data_box = pd.DataFrame({
    "F1-Score": tF+cF+kF,
    "Model"   : (["TF-IDF Baseline"]*SAMPLE_SIZE +
                 ["Cosine Re-ranking (A)"]*SAMPLE_SIZE +
                 ["KMeans Clustering (B)"]*SAMPLE_SIZE),
})
sns.boxplot(data=data_box,x="Model",y="F1-Score",
            palette=[C_TFIDF,C_COSINE,C_CLUSTER],
            width=0.45,linewidth=1.5,ax=ax,
            flierprops=dict(marker="",linestyle="none"))
sns.stripplot(data=data_box,x="Model",y="F1-Score",
              palette=["#1a3a6b","#8b4513","#1a5c30"],
              size=6,alpha=0.7,jitter=True,ax=ax)
for xi,(vals,c) in enumerate(zip([tF,cF,kF],[C_TFIDF,C_COSINE,C_CLUSTER])):
    ax.plot(xi,np.mean(vals),marker="D",color="white",markersize=10,
            markeredgecolor=c,markeredgewidth=2.5,zorder=5)
    ax.text(xi+0.26,np.mean(vals),f"mean={np.mean(vals):.3f}",
            va="center",fontsize=9,color=c)
ax.set_title("F1-Score Distribution — 3 Methods\n(each dot = 1 document | ◇ = mean)",
             fontsize=TSIZE,fontweight="bold")
ax.set_ylabel("F1-Score",fontsize=LSIZE); ax.set_xlabel("")
ax.set_ylim(-0.05,1.05); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig(EVAL_FIGS["fig3"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig3']}")

# ── Fig 4 : Per-doc TP/FP/FN stacked ──────────────────────────────────
fig, axes = plt.subplots(3,1,figsize=(16,10),sharex=True)
x = np.arange(SAMPLE_SIZE)
for ax,(name,tp_d,fp_d,fn_d) in zip(axes,[
    ("TF-IDF Baseline",       t_tp_d,t_fp_d,t_fn_d),
    ("Cosine Re-ranking (A)", c_tp_d,c_fp_d,c_fn_d),
    ("KMeans Clustering (B)", k_tp_d,k_fp_d,k_fn_d),
]):
    ax.bar(x,tp_d,label="TP",color="#2ecc71",edgecolor="white")
    ax.bar(x,fp_d,bottom=tp_d,label="FP",color="#e74c3c",edgecolor="white")
    ax.bar(x,fn_d,bottom=[t+f for t,f in zip(tp_d,fp_d)],
           label="FN",color="#e67e22",edgecolor="white",alpha=0.85)
    ax.set_title(name,fontsize=TSIZE,fontweight="bold")
    ax.set_ylabel("Count",fontsize=LSIZE); ax.set_ylim(0,12)
    ax.legend(fontsize=9,ncol=3,loc="upper right")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
axes[2].set_xticks(x)
axes[2].set_xticklabels([f"D{i+1}" for i in range(SAMPLE_SIZE)],fontsize=7,rotation=45)
plt.suptitle("Per-Document TP / FP / FN  (top-5 keywords per doc)",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig(EVAL_FIGS["fig4"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig4']}")

# ── Fig 5 : Total TP/FP/FN aggregate ──────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(15,5))
bar_colors = ["#2ecc71","#e74c3c","#e67e22"]
for ax,(name,TP,FP,FN) in zip(axes,[
    ("TF-IDF Baseline",       t_TP,t_FP,t_FN),
    ("Cosine Re-ranking (A)", c_TP,c_FP,c_FN),
    ("KMeans Clustering (B)", k_TP,k_FP,k_FN),
]):
    vals = [TP,FP,FN]
    bars = ax.bar(["TP\n(Correct)","FP\n(Wrong pred)","FN\n(Missed)"],
                  vals,color=bar_colors,edgecolor="white",linewidth=1.2,width=0.5)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                str(val),ha="center",fontsize=13,fontweight="bold")
    prec = TP/(TP+FP) if (TP+FP) else 0
    rec  = TP/(TP+FN) if (TP+FN) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    ax.set_title(f"{name}\nP={prec:.3f}  R={rec:.3f}  F1={f1:.3f}",
                 fontsize=11,fontweight="bold")
    ax.set_ylabel("Count (all 25 docs)",fontsize=LSIZE)
    ax.set_ylim(0,max(vals)*1.3)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.3)
plt.suptitle("Total TP / FP / FN across all 25 Documents",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig(EVAL_FIGS["fig5"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig5']}")

# ── Fig 6 : Manual table image ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22,10)); ax.axis("off")
cols = ["Doc","Sentiment","TF-IDF Keywords","Cosine Keywords",
        "Cluster Keywords","Manual Reference","TF-IDF F1","Cosine F1","Cluster F1"]
col_tf,col_co,col_cl = cols.index("TF-IDF F1"),cols.index("Cosine F1"),cols.index("Cluster F1")
tbl = ax.table(cellText=man_df[cols].values,colLabels=cols,
               cellLoc="center",loc="center",colColours=["#d0d8e8"]*len(cols))
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1,1.8)
for i in range(SAMPLE_SIZE):
    scores = {col_tf:man_df["TF-IDF F1"].iloc[i],
              col_co:man_df["Cosine F1"].iloc[i],
              col_cl:man_df["Cluster F1"].iloc[i]}
    winner = max(scores,key=scores.get)
    tbl[i+1,winner].set_facecolor(
        {col_tf:"#d4e6f1",col_co:"#fde8d8",col_cl:"#d5f5e3"}[winner])
ax.set_title("Manual Evaluation — All 25 Documents\n"
             "Blue=TF-IDF wins | Orange=Cosine wins | Green=Cluster wins",
             fontsize=TSIZE,fontweight="bold",pad=20)
plt.tight_layout()
plt.savefig(EVAL_FIGS["fig6"],dpi=150,bbox_inches="tight"); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig6']}")

# ── Fig 7 : Radar chart ────────────────────────────────────────────────
cats   = ["Precision","Recall","F1-Score"]
angles = [n/3*2*np.pi for n in range(3)]+[0]
fig, ax = plt.subplots(figsize=(6,6),subplot_kw={"polar":True})
for vals,color,label in [
    ([t_p,t_r,t_f],C_TFIDF,  "TF-IDF Baseline"),
    ([c_p,c_r,c_f],C_COSINE, "Cosine Re-ranking (A)"),
    ([k_p,k_r,k_f],C_CLUSTER,"KMeans Clustering (B)"),
]:
    v = vals+vals[:1]
    ax.plot(angles,v,color=color,lw=2.2,label=label)
    ax.fill(angles,v,color=color,alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=12)
ax.set_ylim(0,0.6)
ax.set_title("Radar Chart — All 3 Methods",fontsize=TSIZE,fontweight="bold",pad=22)
ax.legend(loc="upper right",bbox_to_anchor=(1.5,1.15),fontsize=9)
plt.tight_layout()
plt.savefig(EVAL_FIGS["fig7"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig7']}")

# ── Fig 8 : Sentiment split ────────────────────────────────────────────
pos_idx = [i for i,s in enumerate(sentiments) if s=="positive"]
neg_idx = [i for i,s in enumerate(sentiments) if s=="negative"]
def smean(scores,idx): return np.mean([scores[i] for i in idx]) if idx else 0

fig, ax = plt.subplots(figsize=(9,5))
x=np.arange(2); bw=0.25
cats_s=[f"Positive\n(n={len(pos_idx)})",f"Negative\n(n={len(neg_idx)})"]
b1=ax.bar(x-bw,[smean(tF,pos_idx),smean(tF,neg_idx)],bw,label="TF-IDF Baseline",      color=C_TFIDF,  edgecolor="white")
b2=ax.bar(x,   [smean(cF,pos_idx),smean(cF,neg_idx)],bw,label="Cosine Re-ranking (A)", color=C_COSINE, edgecolor="white")
b3=ax.bar(x+bw,[smean(kF,pos_idx),smean(kF,neg_idx)],bw,label="KMeans Clustering (B)", color=C_CLUSTER,edgecolor="white")
for b in list(b1)+list(b2)+list(b3):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,
            f"{b.get_height():.3f}",ha="center",fontsize=10,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats_s,fontsize=LSIZE)
all_vals=[smean(tF,pos_idx),smean(tF,neg_idx),smean(cF,pos_idx),
          smean(cF,neg_idx),smean(kF,pos_idx),smean(kF,neg_idx)]
ax.set_ylim(0,max(all_vals)*1.5)
ax.set_ylabel("Avg F1-Score",fontsize=LSIZE)
ax.set_title("Average F1 by Sentiment (Positive vs Negative)",
             fontsize=TSIZE,fontweight="bold")
ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig(EVAL_FIGS["fig8"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig8']}")

# ── Fig 9 : Confusion matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(18,5))
for ax,(name,cm) in zip(axes,[
    ("TF-IDF Baseline",       t_cm),
    ("Cosine Re-ranking (A)", c_cm),
    ("KMeans Clustering (B)", k_cm),
]):
    im = ax.imshow(cm,cmap="Blues",vmin=0)
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    tick_labels=["Not Keyword (0)","Keyword (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(tick_labels,fontsize=9)
    ax.set_yticks([0,1]); ax.set_yticklabels(tick_labels,fontsize=9)
    ax.set_xlabel("Predicted label",fontsize=11,fontweight="bold")
    ax.set_ylabel("Actual label",   fontsize=11,fontweight="bold")
    for i in range(2):
        for j in range(2):
            val=cm[i,j]; clr="white" if val>cm.max()*0.55 else "black"
            ax.text(j,i,[["TN","FP"],["FN","TP"]][i][j]+f"\n{val}",
                    ha="center",va="center",fontsize=13,fontweight="bold",color=clr)
    tn,fp,fn,tp_v=cm.ravel()
    prec=tp_v/(tp_v+fp) if (tp_v+fp) else 0
    rec =tp_v/(tp_v+fn) if (tp_v+fn) else 0
    f1  =2*prec*rec/(prec+rec) if (prec+rec) else 0
    ax.set_title(f"{name}\nP={prec:.3f}  R={rec:.3f}  F1={f1:.3f}",
                 fontsize=11,fontweight="bold")
plt.suptitle("Confusion Matrix — Keyword Extraction (25 docs)",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig(EVAL_FIGS["fig9"],dpi=150); plt.close()
print(f"  Saved -> {EVAL_FIGS['fig9']}")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL COMPARISON SUMMARY")
print("=" * 60)

summary = pd.DataFrame({
    "Model"    : ["TF-IDF Baseline","Cosine Re-ranking (A)","KMeans Clustering (B)"],
    "Accuracy" : [
                 round((t_cm[0,0] + t_cm[1,1]) / t_cm.sum(), 4),
                 round((c_cm[0,0] + c_cm[1,1]) / c_cm.sum(), 4),
                 round((k_cm[0,0] + k_cm[1,1]) / k_cm.sum(), 4)],
    "Precision": [round(t_p,4), round(c_p,4), round(k_p,4)],
    "Recall"   : [round(t_r,4), round(c_r,4), round(k_r,4)],
    "F1-Score" : [round(t_f,4), round(c_f,4), round(k_f,4)],
    "Median F1": [round(float(np.median(tF)),4),
                  round(float(np.median(cF)),4),
                  round(float(np.median(kF)),4)],
    "Std F1"   : [round(float(np.std(tF)),4),
                  round(float(np.std(cF)),4),
                  round(float(np.std(kF)),4)],
    "TP/FP/FN" : [f"{t_TP}/{t_FP}/{t_FN}",
                  f"{c_TP}/{c_FP}/{c_FN}",
                  f"{k_TP}/{k_FP}/{k_FN}"],
})
print(summary.to_string(index=False))

best   = summary["F1-Score"].max()
winner = summary[summary["F1-Score"]==best].iloc[0]["Model"]
print(f"\n  Winner by F1 : {winner}  (F1 = {best:.4f})")
print(f"  Evaluated on : {SAMPLE_SIZE} manually annotated documents")
print(f"  Output folder: ./{EVAL_DIR}/")
print("\n  STEP 5 COMPLETE")
print("=" * 60)