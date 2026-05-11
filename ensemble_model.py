"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          ENSEMBLE : Multi-Model Voting System                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ Combines multiple keyword extraction methods using:                    ║
║   • Weighted voting based on model performance                       ║
║   • Diversity promotion to avoid redundancy                         ║
║   • Consensus filtering for high-confidence keywords                 ║
║   • Adaptive weighting based on document characteristics             ║
║                                                                      ║
║ Models combined:                                                     ║
║   1. TF-IDF Baseline                                                ║
║   2. Cosine Similarity Re-ranking                                   ║
║   3. KMeans Clustering                                             ║
║   4. KeyBERT (if available)                                        ║
║   5. TextRank (if available)                                       ║
║                                                                      ║
║ Outputs -> generated/ensemble_results.csv                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import ADVANCED_CSV, GEN_DIR, EVAL_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
TOP_N_KEYWORDS = 10
MIN_VOTES = 2  # Minimum votes for keyword to be considered
DIVERSITY_THRESHOLD = 0.7  # Cosine similarity threshold for diversity

# Model weights (can be adjusted based on performance)
DEFAULT_WEIGHTS = {
    'tfidf': 0.15,
    'cosine': 0.25,
    'cluster': 0.20,
    'keybert': 0.30,  # Higher weight for transformer model
    'textrank': 0.10
}

def safe_parse_keywords(keyword_list):
    """Safely parse keyword lists from CSV."""
    if isinstance(keyword_list, list):
        return keyword_list
    try:
        return ast.literal_eval(keyword_list)
    except:
        return []

def safe_parse_scores(score_list):
    """Safely parse score lists from CSV."""
    if isinstance(score_list, list):
        return score_list
    try:
        return ast.literal_eval(score_list)
    except:
        return []

def calculate_keyword_votes(keywords_list, scores_list, weights):
    """Calculate weighted votes for each keyword."""
    keyword_votes = defaultdict(float)
    
    for keywords, scores, (model, weight) in zip(keywords_list, scores_list, weights.items()):
        if not keywords:
            continue
            
        # Normalize scores to sum to 1
        if scores and len(scores) == len(keywords):
            total_score = sum(scores)
            if total_score > 0:
                normalized_scores = [s / total_score for s in scores]
            else:
                normalized_scores = [1.0 / len(scores)] * len(scores)
        else:
            normalized_scores = [1.0 / len(keywords)] * len(keywords)
        
        # Add weighted votes
        for keyword, score in zip(keywords, normalized_scores):
            keyword_votes[keyword] += score * weight
    
    return keyword_votes

def promote_diversity(keywords_with_votes, embeddings_dict, threshold=DIVERSITY_THRESHOLD):
    """Promote diversity by filtering out similar keywords."""
    if not embeddings_dict:
        return keywords_with_votes
    
    diverse_keywords = []
    selected_vectors = []
    
    for keyword, votes in keywords_with_votes:
        if keyword not in embeddings_dict:
            # Keep keywords without embeddings
            diverse_keywords.append((keyword, votes))
            continue
        
        keyword_vector = embeddings_dict[keyword]
        
        # Check similarity with already selected keywords
        is_diverse = True
        for selected_vector in selected_vectors:
            similarity = cosine_similarity(
                keyword_vector.reshape(1, -1),
                selected_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > threshold:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_keywords.append((keyword, votes))
            selected_vectors.append(keyword_vector)
    
    return diverse_keywords

def adaptive_weighting(document_length, model_performance):
    """Adapt model weights based on document characteristics."""
    weights = DEFAULT_WEIGHTS.copy()
    
    # For short documents, prefer TF-IDF and TextRank
    if document_length < 100:
        weights['tfidf'] *= 1.5
        weights['textrank'] *= 1.3
        weights['keybert'] *= 0.8
    
    # For long documents, prefer embedding-based methods
    elif document_length > 500:
        weights['cosine'] *= 1.3
        weights['cluster'] *= 1.2
        weights['keybert'] *= 1.2
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights

def consensus_filtering(keywords_votes, min_votes=MIN_VOTES):
    """Filter keywords based on consensus across models."""
    consensus_keywords = []
    
    for keyword, votes in keywords_votes.items():
        if votes >= min_votes:
            consensus_keywords.append((keyword, votes))
    
    return sorted(consensus_keywords, key=lambda x: x[1], reverse=True)

def create_ensemble_keywords(df_row, weights, embeddings_dict=None):
    """Create ensemble keywords for a single document."""
    
    # Collect keywords and scores from all available models
    model_data = {
        'tfidf': (safe_parse_keywords(df_row.get('tfidf_keywords', [])),
                 safe_parse_scores(df_row.get('tfidf_scores', []))),
        'cosine': (safe_parse_keywords(df_row.get('cosine_keywords', [])),
                  safe_parse_scores(df_row.get('cosine_scores', []))),
        'cluster': (safe_parse_keywords(df_row.get('cluster_keywords', [])),
                   safe_parse_scores(df_row.get('cluster_scores', [])))
    }
    
    # Add KeyBERT if available
    if 'keybert_keywords' in df_row:
        model_data['keybert'] = (safe_parse_keywords(df_row.get('keybert_keywords', [])),
                                safe_parse_scores(df_row.get('keybert_scores', [])))
    
    # Add TextRank if available
    if 'textrank_keywords' in df_row:
        model_data['textrank'] = (safe_parse_keywords(df_row.get('textrank_keywords', [])),
                                 safe_parse_scores(df_row.get('textrank_scores', [])))
    
    # Filter weights to only include available models
    available_weights = {k: v for k, v in weights.items() if k in model_data}
    
    # Calculate weighted votes
    keyword_votes = calculate_keyword_votes(
        [v[0] for v in model_data.values()],
        [v[1] for v in model_data.values()],
        available_weights
    )
    
    # Convert to list and sort by votes
    keywords_with_votes = sorted(keyword_votes.items(), key=lambda x: x[1], reverse=True)
    
    # Apply consensus filtering
    consensus_keywords = consensus_filtering(dict(keywords_with_votes))
    
    # Promote diversity if embeddings available
    if embeddings_dict:
        diverse_keywords = promote_diversity(consensus_keywords, embeddings_dict)
    else:
        diverse_keywords = consensus_keywords
    
    # Return top-N keywords
    top_keywords = diverse_keywords[:TOP_N_KEYWORDS]
    
    return [kw for kw, _ in top_keywords], [round(vote, 4) for _, vote in top_keywords]

def load_embeddings_for_diversity():
    """Load word embeddings for diversity promotion."""
    embeddings_dict = {}
    
    # Try to load Word2Vec embeddings
    w2v_model_path = os.path.join(GEN_DIR, "word2vec_model.model")
    if os.path.exists(w2v_model_path):
        try:
            from gensim.models import Word2Vec
            model = Word2Vec.load(w2v_model_path)
            
            # Create embeddings dictionary
            for word in model.wv.index_to_key:
                embeddings_dict[word] = model.wv[word]
            
            print(f"  Loaded {len(embeddings_dict)} Word2Vec embeddings")
        except Exception as e:
            print(f"  Could not load Word2Vec embeddings: {e}")
    
    return embeddings_dict

def run_ensemble_extraction():
    """Main function to run ensemble keyword extraction."""
    
    print("=" * 70)
    print("  ENSEMBLE KEYWORD EXTRACTION — Multi-Model Voting System")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(ADVANCED_CSV):
        print(f"  [ERROR] '{ADVANCED_CSV}' not found. Run pipeline first.")
        return
    
    df = pd.read_csv(ADVANCED_CSV)
    print(f"  Loaded {len(df):,} documents")
    
    # Check for KeyBERT results
    keybert_file = os.path.join(GEN_DIR, "keybert_results.csv")
    if os.path.exists(keybert_file):
        df_keybert = pd.read_csv(keybert_file)
        df = df.merge(df_keybert[['document_id', 'keybert_keywords', 'keybert_scores']], 
                     on='document_id', how='left')
        print("  KeyBERT results merged")
    
    # Check for TextRank results
    textrank_file = os.path.join(GEN_DIR, "textrank_keywords_results.csv")
    if os.path.exists(textrank_file):
        df_textrank = pd.read_csv(textrank_file)
        df = df.merge(df_textrank[['document_id', 'textrank_keywords', 'textrank_scores']], 
                     on='document_id', how='left')
        print("  TextRank results merged")
    
    # Load embeddings for diversity promotion
    embeddings_dict = load_embeddings_for_diversity()
    
    # Process each document
    ensemble_results = []
    
    print(f"  Creating ensemble keywords for all documents...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Ensemble"):
        # Get document length for adaptive weighting
        document_length = len(row.get('clean_review', '').split())
        
        # Get adaptive weights
        weights = adaptive_weighting(document_length, None)
        
        # Create ensemble keywords
        ensemble_keywords, ensemble_scores = create_ensemble_keywords(
            row, weights, embeddings_dict
        )
        
        ensemble_results.append({
            'document_id': idx,
            'review': row.get('review', ''),
            'clean_review': row.get('clean_review', ''),
            'sentiment': row.get('sentiment', ''),
            'ensemble_keywords': ensemble_keywords,
            'ensemble_scores': ensemble_scores,
            'num_models_used': len([k for k in ['tfidf', 'cosine', 'cluster', 'keybert', 'textrank'] 
                                   if f'{k}_keywords' in row and row[f'{k}_keywords']])
        })
    
    # Save ensemble results
    df_ensemble = pd.DataFrame(ensemble_results)
    ensemble_csv = os.path.join(GEN_DIR, "ensemble_results.csv")
    df_ensemble.to_csv(ensemble_csv, index=False)
    
    print(f"  Saved -> {ensemble_csv}")
    
    # Analysis and visualization
    analyze_ensemble_performance(df_ensemble, df)
    
    # Sample results
    print(f"\n  Sample ensemble results:")
    for i in range(min(3, len(df_ensemble))):
        row = df_ensemble.iloc[i]
        print(f"    Doc {i} ({row['sentiment']}, {row['num_models_used']} models):")
        print(f"      Keywords: {row['ensemble_keywords'][:5]}")
        print(f"      Scores:   {row['ensemble_scores'][:5]}")
    
    print(f"\n  Ensemble extraction complete!")
    print("=" * 70)
    
    return df_ensemble

def analyze_ensemble_performance(df_ensemble, df_original):
    """Analyze ensemble performance and create visualizations."""
    
    print(f"\n  Analyzing ensemble performance...")
    
    # Create analysis directory
    analysis_dir = "ensemble_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Model usage statistics
    model_usage = df_ensemble['num_models_used'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    model_usage.plot(kind='bar', color='#4C72B0')
    plt.title('Number of Models Used per Document', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Models', fontsize=12)
    plt.ylabel('Document Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    chart1_path = os.path.join(analysis_dir, "model_usage.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Score distribution analysis
    all_scores = []
    for scores in df_ensemble['ensemble_scores']:
        all_scores.extend(scores)
    
    plt.figure(figsize=(12, 6))
    plt.hist(all_scores, bins=30, alpha=0.7, color='#DD8452', edgecolor='black')
    plt.title('Ensemble Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Ensemble Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart2_path = os.path.join(analysis_dir, "score_distribution.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Keyword diversity analysis
    keyword_diversity = []
    for keywords in df_ensemble['ensemble_keywords']:
        unique_keywords = set(keywords)
        diversity = len(unique_keywords) / len(keywords) if keywords else 0
        keyword_diversity.append(diversity)
    
    plt.figure(figsize=(10, 6))
    plt.hist(keyword_diversity, bins=20, alpha=0.7, color='#55A868', edgecolor='black')
    plt.title('Keyword Diversity Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Diversity Ratio (Unique/Total)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart3_path = os.path.join(analysis_dir, "diversity_analysis.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comparison with individual models (if references available)
    if 'Manual Reference' in df_original.columns:
        create_model_comparison_chart(df_ensemble, df_original, analysis_dir)
    
    # Save analysis summary
    analysis_summary = {
        'total_documents': len(df_ensemble),
        'avg_models_per_doc': df_ensemble['num_models_used'].mean(),
        'avg_keywords_per_doc': df_ensemble['ensemble_keywords'].apply(len).mean(),
        'avg_ensemble_score': np.mean(all_scores),
        'keyword_diversity_avg': np.mean(keyword_diversity)
    }
    
    summary_df = pd.DataFrame([analysis_summary])
    summary_csv = os.path.join(analysis_dir, "ensemble_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"    Analysis saved to {analysis_dir}/")
    print(f"    Average models per document: {analysis_summary['avg_models_per_doc']:.1f}")
    print(f"    Average keywords per document: {analysis_summary['avg_keywords_per_doc']:.1f}")
    print(f"    Average ensemble score: {analysis_summary['avg_ensemble_score']:.3f}")

def create_model_comparison_chart(df_ensemble, df_original, analysis_dir):
    """Create comparison chart between ensemble and individual models."""
    
    try:
        # Calculate F1 scores for ensemble and individual models
        models_to_compare = ['tfidf', 'cosine', 'cluster']
        if 'keybert_keywords' in df_original.columns:
            models_to_compare.append('keybert')
        
        f1_scores = {}
        
        # Ensemble F1
        ensemble_f1_scores = []
        for i, row in df_ensemble.iterrows():
            if i < len(df_original):
                ref_keywords = safe_parse_keywords(df_original.iloc[i].get('Manual Reference', []))
                pred_keywords = row['ensemble_keywords']
                
                if pred_keywords and ref_keywords:
                    pred_set = set(pred_keywords)
                    ref_set = set(ref_keywords)
                    
                    tp = len(pred_set & ref_set)
                    fp = len(pred_set - ref_set)
                    fn = len(ref_set - pred_set)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    ensemble_f1_scores.append(f1)
        
        f1_scores['Ensemble'] = np.mean(ensemble_f1_scores) if ensemble_f1_scores else 0
        
        # Individual model F1 scores (simplified calculation)
        for model in models_to_compare:
            model_f1_scores = []
            for i, row in df_original.iterrows():
                if i < len(df_ensemble):
                    ref_keywords = safe_parse_keywords(row.get('Manual Reference', []))
                    pred_keywords = safe_parse_keywords(row.get(f'{model}_keywords', []))
                    
                    if pred_keywords and ref_keywords:
                        pred_set = set(pred_keywords)
                        ref_set = set(ref_keywords)
                        
                        tp = len(pred_set & ref_set)
                        fp = len(pred_set - ref_set)
                        fn = len(ref_set - pred_set)
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        model_f1_scores.append(f1)
            
            f1_scores[model.upper()] = np.mean(model_f1_scores) if model_f1_scores else 0
        
        # Create comparison chart
        plt.figure(figsize=(12, 6))
        models = list(f1_scores.keys())
        scores = list(f1_scores.values())
        colors = ['#C44E52' if model == 'Ensemble' else '#4C72B0' for model in models]
        
        bars = plt.bar(models, scores, color=colors)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Performance Comparison (F1 Score)', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.ylim(0, max(scores) * 1.2)
        plt.tight_layout()
        
        chart_path = os.path.join(analysis_dir, "model_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Error creating comparison chart: {e}")

if __name__ == "__main__":
    run_ensemble_extraction()
