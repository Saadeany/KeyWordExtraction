# Keyword Extraction System
> Project 5 — Natural Language Processing  
> Automatically extract the most important keywords and phrases from text using statistical, semantic, and graph-based methods.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [GUI](#gui)
- [Bonus — TextRank](#bonus--textrank)
- [Requirements](#requirements)

---

## Overview

This system extracts the most important keywords from text without reading the full document. Instead of returning common words like `"is"`, `"the"`, or `"by"`, it identifies meaningful terms like `"machine learning"`, `"healthcare"`, or `"diagnosis"` that represent the main ideas.

**Example:**

| Input Text | Bad Output | Good Output (this system) |
|---|---|---|
| *"Artificial Intelligence is transforming healthcare by improving diagnosis and patient care using advanced machine learning models."* | `is, the, by, using` | `Artificial Intelligence`, `healthcare`, `diagnosis`, `machine learning` |

The system implements four models across two categories:

- **Baseline** — TF-IDF statistical extraction
- **Advanced** — Word embedding-based extraction (Cosine Re-ranking + KMeans Clustering)
- **Bonus** — TextRank graph-based extraction

---

## Dataset

**IMDB Movie Reviews Dataset** — 50,000 movie reviews (positive and negative sentiment)

The dataset was chosen because:
- Reviews are rich in domain-specific vocabulary
- Varied writing styles test robustness of keyword extraction
- Large enough for meaningful TF-IDF and Word2Vec training

Each review is used as a standalone document for keyword extraction.

---

## Project Structure

```
KeyWordExtraction/
│
├── preprocessing.py                  # Step 1 — text cleaning pipeline
├── TF_IDF.py                         # Step 2 — TF-IDF feature extraction
├── word_embeddings.py                # Step 3 — Word2Vec training
├── advanced_keyword_extraction.py    # Step 4 — Cosine + KMeans models
├── evaluation.py                     # Step 5 — metrics and charts
├── bonus_model.py                    # Bonus — TextRank model
├── gui.py                            # GUI (tkinter, 4 tabs)
├── main.py                           # Run full pipeline from terminal
├── paths.py                          # Shared file paths
│
├── generated/                        # Auto-created — pipeline outputs
│   ├── IMDB_cleaned.csv
│   ├── tfidf_results.csv
│   ├── word2vec_model.bin
│   ├── advanced_keywords_results.csv
│   └── textrank_keywords_results.csv
│
└── evaluation_results/               # Auto-created — charts and metrics
    ├── evaluation_results.csv
    ├── fig1_summary_bar.png
    ├── fig2_per_doc_f1.png
    ├── ...
    ├── textrank_comparison.png
    └── textrank_before_after.png
```

---

## Models

### Model 1 — TF-IDF Baseline
**File:** `TF_IDF.py`  
**Approach:** Statistical

TF-IDF (Term Frequency–Inverse Document Frequency) scores each word based on how often it appears in a document versus how common it is across all documents. Rare but frequent words in a document score highest.

- Uses `ngram_range=(1, 2)` to capture multi-word phrases (bigrams)
- `max_features=5000` vocabulary
- Stopwords filtered in preprocessing
- Top-N keywords selected by score

**Strength:** Fast, interpretable, no training required  
**Weakness:** Ignores word meaning — "good" and "great" treated as unrelated

---

### Model 2 — Cosine Similarity Re-ranking
**File:** `advanced_keyword_extraction.py` (Method A)  
**Approach:** Semantic — Word2Vec embeddings

Takes the TF-IDF keyword candidates and re-ranks them by measuring the cosine similarity between each keyword's Word2Vec vector and the full document vector. Keywords most semantically aligned with the document's overall meaning rank highest.

- Builds document embedding by averaging all word vectors
- Re-ranks using `cosine_similarity(keyword_vector, document_vector)`
- Fixes TF-IDF's inability to understand meaning

**Strength:** Understands semantic relatedness  
**Weakness:** Fails on out-of-vocabulary (OOV) words not seen during Word2Vec training

---

### Model 3 — KMeans Clustering
**File:** `advanced_keyword_extraction.py` (Method B)  
**Approach:** Semantic — clustering for diversity

Groups keyword embeddings into K clusters (K = number of desired keywords). One representative keyword is picked from each cluster. This forces the output to cover different semantic areas of the document rather than returning near-synonyms.

- Uses `sklearn KMeans` on Word2Vec keyword vectors
- Picks the keyword closest to each cluster centroid
- Maximises diversity across extracted keywords

**Strength:** Diverse, non-redundant keyword set  
**Weakness:** Cluster quality depends on Word2Vec embedding quality

---

### Model 4 — TextRank (Bonus)
**File:** `bonus_model.py`  
**Approach:** Graph-based — PageRank on word co-occurrence

Builds a graph where words are nodes and edges connect words that appear within a sliding window of each other. PageRank then scores each word by how many important neighbors it has. No corpus statistics or pre-trained embeddings are needed — it works entirely from the document itself.

- Window size = 4 tokens
- Damping factor α = 0.85 (same as original TextRank paper)
- Adjacent top-ranked words are merged into keyphrases automatically
- Completely independent of TF-IDF and Word2Vec

**Strength:** Works on any document without corpus or embeddings; naturally produces multi-word keyphrases  
**Weakness:** Less effective on very short texts

---

## How to Run

### Option A — Terminal (recommended for grading)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (Steps 1–5)
python main.py

# Run bonus model after pipeline completes
python bonus_model.py
```

### Option B — GUI

```bash
python gui.py
```

Then inside the GUI:

1. Go to the **Pipeline Runner** tab
2. Click **Step 1** through **Step 5** in order
3. Watch live console output for each step
4. Go to the **Results Viewer** tab to see all charts
5. After all steps complete, run `python bonus_model.py` from terminal for the bonus

> **Important:** Always run Steps 1–5 before running `bonus_model.py`. The bonus model reads the output files generated by the pipeline. Running it before the pipeline will cause a `FileNotFoundError`.

---

## Evaluation Metrics

Evaluation is performed against **25 manually annotated documents** used as ground truth. Each document has a reference set of expected keywords.

| Metric | Description |
|---|---|
| **Accuracy** | `(TP + TN) / total` — overall correct predictions |
| **Precision** | `TP / (TP + FP)` — how many extracted keywords are relevant |
| **Recall** | `TP / (TP + FN)` — how many relevant keywords were found |
| **F1-Score** | `2 × (P × R) / (P + R)` — harmonic mean of precision and recall |
| **Confusion Matrix** | Visual breakdown of TP / TN / FP / FN per model |

All four models are compared on the same 25 documents so results are directly comparable.

**Output charts (9 total from `evaluation.py`):**
- Summary bar chart (Precision / Recall / F1 per model)
- Per-document F1 scores
- Boxplot of F1 distributions
- Radar chart across all metrics
- TP / FP / FN breakdown
- Sentiment-split analysis (positive vs negative reviews)
- Confusion matrices (one per model)

---

## Results

Results are saved to `evaluation_results/evaluation_results.csv` after running Step 5.

| Model | Stage | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| TF-IDF Baseline | Before | — | — | — | — |
| Cosine Re-ranking | Before | — | — | — | — |
| KMeans Clustering | Before | — | — | — | — |
| TextRank | After (Bonus) | — | — | — | — |

> Run the pipeline to populate the table with your actual scores.

---

## GUI

The GUI (`gui.py`) is built with `tkinter` and has 4 tabs:

**Pipeline Runner** — run each of the 5 steps with a button click and see live console output.

**Keyword Extractor** — enter any text and instantly extract keywords using both TF-IDF and Word2Vec. No need to run the full pipeline first for this tab.

**Results Viewer** — browse all 9 evaluation charts and the results table without leaving the app.

**About** — project description and model overview.

---

## Bonus — TextRank

TextRank is a completely different approach from the other three models. While TF-IDF uses corpus statistics and Word2Vec uses learned embeddings, TextRank builds a co-occurrence graph from the document itself and applies the PageRank algorithm.

**Before vs After comparison** generated by `bonus_model.py`:

- `evaluation_results/textrank_comparison.png` — all 4 models side by side (Precision / Recall / F1)
- `evaluation_results/textrank_before_after.png` — per-document F1 improvement + radar chart with TextRank highlighted

The before/after story:

| | TF-IDF | Cosine | KMeans | TextRank |
|---|---|---|---|---|
| Needs corpus? | Yes | Yes | Yes | No |
| Understands meaning? | No | Yes | Yes | Partial |
| Multi-word phrases? | Bigrams only | No | No | Yes (natural) |
| Works on OOV words? | No | No | No | Yes |

---

## Requirements

```
pandas
numpy
scikit-learn
nltk
gensim
networkx
matplotlib
seaborn
```

Install all at once:

```bash
pip install pandas numpy scikit-learn nltk gensim networkx matplotlib seaborn
```

Also download NLTK data (done automatically on first run, or manually):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
