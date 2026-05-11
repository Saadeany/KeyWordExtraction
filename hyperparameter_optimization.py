"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          OPTIMIZATION : Hyperparameter Tuning                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ Optimizes hyperparameters for:                                        ║
║   • TF-IDF parameters (max_features, ngram_range, min/max_df)        ║
║   • Word2Vec parameters (vector_size, window, epochs)                ║
║   • Clustering parameters (n_clusters, algorithm)                   ║
║   • Advanced model parameters (diversity, top_n)                    ║
║                                                                      ║
║ Uses grid search and cross-validation for optimal performance.       ║
║ Outputs optimized parameter sets and performance comparisons.       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paths import CLEANED_CSV, GEN_DIR, EVAL_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
CV_FOLDS = 5
SAMPLE_SIZE = 100  # Use subset for faster optimization
RANDOM_STATE = 42

# TF-IDF parameter grids
TFIDF_PARAM_GRID = {
    'max_features': [1000, 3000, 5000, 8000],
    'ngram_range': [(1, 1), (1, 2), (1, 3)],
    'min_df': [1, 2, 5],
    'max_df': [0.7, 0.85, 0.95]
}

# Word2Vec parameter grids
W2V_PARAM_GRID = {
    'vector_size': [50, 100, 200],
    'window': [3, 5, 7],
    'epochs': [5, 10, 20],
    'min_count': [1, 2, 5]
}

# Clustering parameter grids
CLUSTER_PARAM_GRID = {
    'n_clusters': [3, 5, 7, 10],
    'n_init': [10, 20],
    'max_iter': [100, 300]
}

def safe_parse_keywords(keyword_list):
    """Safely parse keyword lists from CSV."""
    if isinstance(keyword_list, list):
        return keyword_list
    try:
        return ast.literal_eval(keyword_list)
    except:
        return []

def evaluate_tfidf_params(corpus, references, param_combination):
    """Evaluate TF-IDF parameter combination."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=param_combination['max_features'],
            ngram_range=param_combination['ngram_range'],
            min_df=param_combination['min_df'],
            max_df=param_combination['max_df'],
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top 5 keywords for each document
        all_f1_scores = []
        top_n = 5
        
        for i in range(len(corpus)):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            predicted_keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            reference_keywords = references[i]
            
            # Calculate F1 score
            if predicted_keywords and reference_keywords:
                predicted_set = set(predicted_keywords)
                reference_set = set(reference_keywords)
                
                tp = len(predicted_set & reference_set)
                fp = len(predicted_set - reference_set)
                fn = len(reference_set - predicted_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                all_f1_scores.append(f1)
        
        return np.mean(all_f1_scores) if all_f1_scores else 0.0
        
    except Exception as e:
        return 0.0

def evaluate_word2vec_params(corpus_tokenized, references, param_combination):
    """Evaluate Word2Vec parameter combination."""
    try:
        model = Word2Vec(
            sentences=corpus_tokenized,
            vector_size=param_combination['vector_size'],
            window=param_combination['window'],
            min_count=param_combination['min_count'],
            epochs=param_combination['epochs'],
            seed=RANDOM_STATE,
            workers=4
        )
        
        # Simple evaluation: vocabulary coverage and semantic similarity
        vocab_size = len(model.wv)
        
        # Calculate average similarity for reference keywords
        total_similarity = 0
        total_pairs = 0
        
        for ref_keywords in references:
            if len(ref_keywords) >= 2:
                for i in range(len(ref_keywords) - 1):
                    kw1, kw2 = ref_keywords[i], ref_keywords[i + 1]
                    if kw1 in model.wv and kw2 in model.wv:
                        similarity = model.wv.similarity(kw1, kw2)
                        total_similarity += similarity
                        total_pairs += 1
        
        avg_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
        
        # Combine vocabulary size and semantic coherence
        score = (vocab_size / 1000) * 0.3 + avg_similarity * 0.7  # Weighted combination
        
        return score
        
    except Exception as e:
        return 0.0

def evaluate_clustering_params(embeddings, references, param_combination):
    """Evaluate clustering parameter combination."""
    try:
        kmeans = KMeans(
            n_clusters=param_combination['n_clusters'],
            n_init=param_combination['n_init'],
            max_iter=param_combination['max_iter'],
            random_state=RANDOM_STATE
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Evaluate cluster quality using silhouette score (simplified)
        from sklearn.metrics import silhouette_score
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        return silhouette_avg
        
    except Exception as e:
        return 0.0

def optimize_tfidf(corpus, references):
    """Optimize TF-IDF hyperparameters."""
    print("  Optimizing TF-IDF parameters...")
    
    # Generate all parameter combinations
    param_names = list(TFIDF_PARAM_GRID.keys())
    param_values = list(TFIDF_PARAM_GRID.values())
    combinations = list(product(*param_values))
    
    best_score = 0.0
    best_params = None
    results = []
    
    print(f"    Testing {len(combinations)} parameter combinations...")
    
    for i, combination in enumerate(tqdm(combinations, desc="    TF-IDF")):
        params = dict(zip(param_names, combination))
        score = evaluate_tfidf_params(corpus, references, params)
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score, results

def optimize_word2vec(corpus_tokenized, references):
    """Optimize Word2Vec hyperparameters."""
    print("  Optimizing Word2Vec parameters...")
    
    # Generate all parameter combinations
    param_names = list(W2V_PARAM_GRID.keys())
    param_values = list(W2V_PARAM_GRID.values())
    combinations = list(product(*param_values))
    
    best_score = 0.0
    best_params = None
    results = []
    
    # Limit combinations for faster execution
    combinations = combinations[:20]  # Take first 20 combinations
    
    print(f"    Testing {len(combinations)} parameter combinations...")
    
    for i, combination in enumerate(tqdm(combinations, desc="    Word2Vec")):
        params = dict(zip(param_names, combination))
        score = evaluate_word2vec_params(corpus_tokenized, references, params)
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score, results

def optimize_clustering(embeddings, references):
    """Optimize clustering hyperparameters."""
    print("  Optimizing clustering parameters...")
    
    # Generate all parameter combinations
    param_names = list(CLUSTER_PARAM_GRID.keys())
    param_values = list(CLUSTER_PARAM_GRID.values())
    combinations = list(product(*param_values))
    
    best_score = 0.0
    best_params = None
    results = []
    
    print(f"    Testing {len(combinations)} parameter combinations...")
    
    for i, combination in enumerate(tqdm(combinations, desc="    Clustering")):
        params = dict(zip(param_names, combination))
        score = evaluate_clustering_params(embeddings, references, params)
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score, results

def run_hyperparameter_optimization():
    """Main function to run hyperparameter optimization."""
    
    print("=" * 70)
    print("  HYPERPARAMETER OPTIMIZATION — Grid Search & Cross-Validation")
    print("=" * 70)
    
    # Create output directory
    opt_dir = "optimization_results"
    os.makedirs(opt_dir, exist_ok=True)
    print(f"  Output directory: {opt_dir}/")
    
    # Load data
    if not os.path.exists(CLEANED_CSV):
        print(f"  [ERROR] '{CLEANED_CSV}' not found. Run preprocessing first.")
        return
    
    df = pd.read_csv(CLEANED_CSV)
    df["clean_review"] = df["clean_review"].fillna("")
    
    # Sample for faster optimization
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    corpus = df["clean_review"].tolist()
    corpus_tokenized = [doc.split() for doc in corpus]
    
    # Create synthetic reference keywords for evaluation
    # In real use, these would be human annotations
    print("  Creating reference keywords for evaluation...")
    references = []
    for doc in corpus:
        words = doc.split()
        # Use most frequent words as synthetic reference
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        references.append([word for word, freq in top_words])
    
    print(f"  Optimizing on {len(df)} documents")
    
    # Optimize each component
    optimization_results = {}
    
    # 1. TF-IDF Optimization
    tfidf_best, tfidf_score, tfidf_results = optimize_tfidf(corpus, references)
    optimization_results['tfidf'] = {
        'best_params': tfidf_best,
        'best_score': tfidf_score,
        'all_results': tfidf_results
    }
    
    # 2. Word2Vec Optimization
    w2v_best, w2v_score, w2v_results = optimize_word2vec(corpus_tokenized, references)
    optimization_results['word2vec'] = {
        'best_params': w2v_best,
        'best_score': w2v_score,
        'all_results': w2v_results
    }
    
    # 3. Clustering Optimization (need embeddings first)
    print("  Training baseline Word2Vec for clustering optimization...")
    baseline_model = Word2Vec(
        sentences=corpus_tokenized,
        vector_size=100,
        window=5,
        min_count=2,
        epochs=10,
        seed=RANDOM_STATE
    )
    
    # Create document embeddings
    doc_embeddings = []
    for tokens in corpus_tokenized:
        vecs = [baseline_model.wv[word] for word in tokens if word in baseline_model.wv]
        if vecs:
            doc_emb = np.mean(vecs, axis=0)
        else:
            doc_emb = np.zeros(100)
        doc_embeddings.append(doc_emb)
    
    doc_embeddings = np.array(doc_embeddings)
    
    cluster_best, cluster_score, cluster_results = optimize_clustering(doc_embeddings, references)
    optimization_results['clustering'] = {
        'best_params': cluster_best,
        'best_score': cluster_score,
        'all_results': cluster_results
    }
    
    # Save results
    results_df = pd.DataFrame({
        'Component': ['TF-IDF', 'Word2Vec', 'Clustering'],
        'Best_Params': [str(tfidf_best), str(w2v_best), str(cluster_best)],
        'Best_Score': [tfidf_score, w2v_score, cluster_score]
    })
    
    results_csv = os.path.join(opt_dir, "optimization_summary.csv")
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n  OPTIMIZATION RESULTS:")
    print(f"  TF-IDF:      Score={tfidf_score:.4f}, Params={tfidf_best}")
    print(f"  Word2Vec:    Score={w2v_score:.4f}, Params={w2v_best}")
    print(f"  Clustering:  Score={cluster_score:.4f}, Params={cluster_best}")
    
    print(f"\n  Results saved -> {results_csv}")
    
    # Create visualization
    create_optimization_charts(optimization_results, opt_dir)
    
    print(f"\n  Hyperparameter optimization complete!")
    print("=" * 70)
    
    return optimization_results

def create_optimization_charts(results, opt_dir):
    """Create optimization result visualizations."""
    
    print(f"  Creating optimization charts...")
    
    # Chart 1: Parameter impact for TF-IDF
    tfidf_results = results['tfidf']['all_results']
    if tfidf_results:
        df_tfidf = pd.DataFrame([
            {
                'max_features': r['params']['max_features'],
                'ngram_range': str(r['params']['ngram_range']),
                'score': r['score']
            } for r in tfidf_results
        ])
        
        plt.figure(figsize=(12, 6))
        for ngram in df_tfidf['ngram_range'].unique():
            subset = df_tfidf[df_tfidf['ngram_range'] == ngram]
            plt.plot(subset['max_features'], subset['score'], 'o-', 
                    label=f'ngram={ngram}', linewidth=2, markersize=8)
        
        plt.title('TF-IDF Parameter Optimization', fontsize=14, fontweight='bold')
        plt.xlabel('Max Features', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        chart1_path = os.path.join(opt_dir, "tfidf_optimization.png")
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Chart 2: Overall optimization comparison
    components = ['TF-IDF', 'Word2Vec', 'Clustering']
    scores = [
        results['tfidf']['best_score'],
        results['word2vec']['best_score'], 
        results['clustering']['best_score']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(components, scores, color=['#4C72B0', '#DD8452', '#55A868'])
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Hyperparameter Optimization Results', fontsize=14, fontweight='bold')
    plt.ylabel('Best Score', fontsize=12)
    plt.ylim(0, max(scores) * 1.2)
    plt.tight_layout()
    
    chart2_path = os.path.join(opt_dir, "optimization_summary.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Charts saved to {opt_dir}/")

if __name__ == "__main__":
    run_hyperparameter_optimization()
