"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          ENHANCED : KeyBERT - Transformer-Based Extraction          ║
╠══════════════════════════════════════════════════════════════════════╣
║ What is KeyBERT?                                                    ║
║   A transformer-based keyword extraction method that uses BERT     ║
║   embeddings to identify the most relevant keywords/keyphrases.     ║
║   It combines sentence transformers with cosine similarity for      ║
║   state-of-the-art performance.                                     ║
║                                                                      ║
║ Why is it better than previous methods?                            ║
║   • Uses contextual embeddings (BERT) vs static (Word2Vec)         ║
║   • Handles out-of-vocabulary words naturally                       ║
║   • Better semantic understanding of document context               ║
║   • Produces higher quality, more diverse keywords                  ║
║                                                                      ║
║ Inputs  : generated/IMDB_cleaned.csv                                ║
║           generated/advanced_keywords_results.csv  (for comparison) ║
║ Outputs : generated/keybert_results.csv                             ║
║           evaluation_results/keybert_comparison.png                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import CLEANED_CSV, ADVANCED_CSV, GEN_DIR, EVAL_DIR

warnings.filterwarnings("ignore")

# ── Try to import KeyBERT components ───────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("  [WARNING] KeyBERT not available. Install with:")
    print("    pip install keybert sentence-transformers torch")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
TOP_N_KEYWORDS = 10
MIN_NGRAM = 1
MAX_NGRAM = 3
USE_MMR = True  # Maximal Marginal Relevance for diversity
DIVERSITY = 0.3

def extract_keybert_keywords(text, model, top_n=TOP_N_KEYWORDS, use_mmr=USE_MMR, diversity=DIVERSITY):
    """
    Extract keywords using KeyBERT with optional MMR for diversity.
    
    Args:
        text: Input text
        model: KeyBERT model instance
        top_n: Number of keywords to extract
        use_mmr: Whether to use Maximal Marginal Relevance
        diversity: Diversity parameter for MMR (0-1)
    
    Returns:
        List of (keyword, score) tuples
    """
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        if use_mmr:
            # Use MMR for diverse keyword selection
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=(MIN_NGRAM, MAX_NGRAM),
                stop_words='english',
                use_mmr=True,
                diversity=diversity,
                top_n=top_n
            )
        else:
            # Standard cosine similarity
            keywords = model.extract_keywords(
                text,
                keyphrase_ngram_range=(MIN_NGRAM, MAX_NGRAM),
                stop_words='english',
                top_n=top_n
            )
        
        return keywords
    except Exception as e:
        print(f"    Error extracting keywords: {e}")
        return []

def run_keybert_extraction():
    """Main function to run KeyBERT keyword extraction."""
    
    if not KEYBERT_AVAILABLE:
        print("=" * 65)
        print("  KEYBERT EXTRACTION - SKIPPED (LIBRARIES NOT INSTALLED)")
        print("=" * 65)
        print("  To enable KeyBERT, run:")
        print("    pip install keybert sentence-transformers torch")
        return
    
    print("=" * 65)
    print("  ENHANCED KEYWORD EXTRACTION — KeyBERT (Transformer-Based)")
    print("=" * 65)
    print(f"  Output folder : {GEN_DIR}/")
    
    # Load data
    df = pd.read_csv(CLEANED_CSV)
    df["clean_review"] = df["clean_review"].fillna("")
    
    print(f"  Documents loaded : {len(df):,}")
    
    # Initialize KeyBERT model
    print("  Loading KeyBERT model (this may take a moment)...")
    try:
        # Use a lightweight but effective model
        model = KeyBERT('all-MiniLM-L6-v2')
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return
    
    # Extract keywords for all documents
    print(f"  Extracting top-{TOP_N_KEYWORDS} keywords per document...")
    
    keybert_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  KeyBERT extraction"):
        # Use original review for better context (cleaned for other models)
        text = row["review"]
        
        # Extract keywords
        keywords = extract_keybert_keywords(
            text, model, 
            top_n=TOP_N_KEYWORDS, 
            use_mmr=USE_MMR, 
            diversity=DIVERSITY
        )
        
        keybert_results.append({
            "document_id": idx,
            "review": text,
            "clean_review": row["clean_review"],
            "sentiment": row["sentiment"],
            "keybert_keywords": [kw for kw, score in keywords],
            "keybert_scores": [round(score, 4) for kw, score in keywords]
        })
    
    # Save results
    df_keybert = pd.DataFrame(keybert_results)
    keybert_csv = os.path.join(GEN_DIR, "keybert_results.csv")
    df_keybert.to_csv(keybert_csv, index=False)
    
    print(f"  Saved -> {keybert_csv}")
    
    # Sample results
    print(f"\n  Sample results:")
    for i in range(min(3, len(df_keybert))):
        row = df_keybert.iloc[i]
        print(f"    Doc {i} ({row['sentiment']}):")
        print(f"      Keywords: {row['keybert_keywords'][:5]}")
        print(f"      Scores:   {row['keybert_scores'][:5]}")
    
    # Compare with existing results if available
    if os.path.exists(ADVANCED_CSV):
        print("\n  Comparison with existing models:")
        compare_with_existing_models(df_keybert)
    
    print("\n  KeyBERT extraction complete!")
    print("=" * 65)

def compare_with_existing_models(df_keybert):
    """Compare KeyBERT results with existing models."""
    
    try:
        df_existing = pd.read_csv(ADVANCED_CSV)
        
        # Sample comparison for first 5 documents
        print("    Document  | TF-IDF     | Cosine    | KMeans    | KeyBERT")
        print("    --------- | ---------- | ---------- | ---------- | ----------")
        
        for i in range(min(5, len(df_keybert))):
            keybert_kws = df_keybert.iloc[i]['keybert_keywords'][:3]
            
            if i < len(df_existing):
                row = df_existing.iloc[i]
                tfidf_kws = row['tfidf_keywords'][:3] if 'tfidf_keywords' in row else []
                cosine_kws = row['cosine_keywords'][:3] if 'cosine_keywords' in row else []
                cluster_kws = row['cluster_keywords'][:3] if 'cluster_keywords' in row else []
                
                print(f"    {i:8d} | {str(tfidf_kws[:2]):10s} | {str(cosine_kws[:2]):10s} | {str(cluster_kws[:2]):10s} | {str(keybert_kws[:2]):10s}")
            else:
                print(f"    {i:8d} | {'N/A':10s} | {'N/A':10s} | {'N/A':10s} | {str(keybert_kws[:2]):10s}")
    
    except Exception as e:
        print(f"    Error in comparison: {e}")

def create_comparison_chart():
    """Create comparison chart if evaluation data is available."""
    
    if not os.path.exists("evaluation_results/evaluation_results.csv"):
        return
    
    try:
        df_eval = pd.read_csv("evaluation_results/evaluation_results.csv")
        
        # This would be enhanced after running evaluation with KeyBERT
        print("  Comparison chart will be generated after evaluation update")
        
    except Exception as e:
        print(f"  Error creating comparison chart: {e}")

if __name__ == "__main__":
    run_keybert_extraction()
