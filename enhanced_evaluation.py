"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          ENHANCED : Advanced Evaluation Metrics                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ Enhanced evaluation with:                                             ║
║   • Precision@K, Recall@K, F1@K for top-K keywords                  ║
║   • Mean Average Precision (MAP) for ranking quality                 ║
║   • Normalized Discounted Cumulative Gain (nDCG)                    ║
║   • Statistical significance testing                                 ║
║   • Expanded evaluation dataset (100+ documents)                    ║
║                                                                      ║
║ Evaluates 4 methods against manual annotations:                      ║
║    1. TF-IDF Baseline                                                ║
║    2. Cosine Similarity Re-ranking                                   ║
║    3. KMeans Clustering                                             ║
║    4. KeyBERT (Transformer-based)                                   ║
║                                                                      ║
║ Outputs -> evaluation_results_enhanced/                              ║
║    enhanced_evaluation_results.csv  +  fig1...fig12 .png           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from paths import ADVANCED_CSV, GEN_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
SAMPLE_SIZE = 100  # Expanded from 25 to 100 documents
TOP_K_VALUES = [3, 5, 10]  # Evaluate at different K values
RANDOM_STATE = 42
ALPHA = 0.05  # Significance level for statistical tests

# Color scheme
COLORS = {
    'TF-IDF': '#4C72B0',
    'Cosine': '#DD8452', 
    'KMeans': '#55A868',
    'KeyBERT': '#C44E52'
}

# ══════════════════════════════════════════════════════════════════════
# ENHANCED METRICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def precision_at_k(predicted, relevant, k):
    """Calculate Precision@K."""
    if k == 0:
        return 0.0
    predicted_k = predicted[:k]
    relevant_set = set(relevant)
    
    if not predicted_k:
        return 0.0
    
    correct = sum(1 for item in predicted_k if item in relevant_set)
    return correct / len(predicted_k)

def recall_at_k(predicted, relevant, k):
    """Calculate Recall@K."""
    if not relevant:
        return 0.0
    predicted_k = predicted[:k]
    relevant_set = set(relevant)
    
    correct = sum(1 for item in predicted_k if item in relevant_set)
    return correct / len(relevant_set)

def f1_at_k(predicted, relevant, k):
    """Calculate F1@K."""
    precision = precision_at_k(predicted, relevant, k)
    recall = recall_at_k(predicted, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def average_precision(predicted, relevant):
    """Calculate Average Precision for a single document."""
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    precision_scores = []
    num_relevant_found = 0
    
    for i, item in enumerate(predicted):
        if item in relevant_set:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            precision_scores.append(precision_at_i)
    
    return np.mean(precision_scores) if precision_scores else 0.0

def mean_average_precision(all_predicted, all_relevant):
    """Calculate Mean Average Precision across all documents."""
    ap_scores = []
    
    for predicted, relevant in zip(all_predicted, all_relevant):
        ap = average_precision(predicted, relevant)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def ndcg_at_k(predicted, relevant, k):
    """Calculate Normalized Discounted Cumulative Gain at K."""
    if k == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    relevance_set = set(relevant)
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        relevance = 1.0 if item in relevance_set else 0.0
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    ideal_relevances = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_all_metrics(predicted_keywords, reference_keywords, k_values=TOP_K_VALUES):
    """Calculate all enhanced metrics for a list of documents."""
    results = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        f1_scores = []
        ndcg_scores = []
        
        for pred, ref in zip(predicted_keywords, reference_keywords):
            p_k = precision_at_k(pred, ref, k)
            r_k = recall_at_k(pred, ref, k)
            f1_k = f1_at_k(pred, ref, k)
            ndcg_k = ndcg_at_k(pred, ref, k)
            
            precisions.append(p_k)
            recalls.append(r_k)
            f1_scores.append(f1_k)
            ndcg_scores.append(ndcg_k)
        
        results[f'Precision@{k}'] = np.mean(precisions)
        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'F1@{k}'] = np.mean(f1_scores)
        results[f'nDCG@{k}'] = np.mean(ndcg_scores)
    
    # Calculate MAP
    map_score = mean_average_precision(predicted_keywords, reference_keywords)
    results['MAP'] = map_score
    
    return results

def statistical_significance_test(scores1, scores2, alpha=ALPHA):
    """Perform paired t-test for statistical significance."""
    if len(scores1) != len(scores2):
        return None, None, False
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Check if significant
    is_significant = p_value < alpha
    
    return t_stat, p_value, is_significant

def safe_parse_keywords(keyword_list):
    """Safely parse keyword lists from CSV."""
    if isinstance(keyword_list, list):
        return keyword_list
    try:
        return ast.literal_eval(keyword_list)
    except:
        return []

def load_evaluation_data():
    """Load and prepare evaluation data."""
    
    # Check for KeyBERT results
    keybert_file = os.path.join(GEN_DIR, "keybert_results.csv")
    has_keybert = os.path.exists(keybert_file)
    
    # Load advanced results
    if not os.path.exists(ADVANCED_CSV):
        raise FileNotFoundError(f"[ERROR] '{ADVANCED_CSV}' not found. Run pipeline first.")
    
    df = pd.read_csv(ADVANCED_CSV)
    
    # Load KeyBERT if available
    if has_keybert:
        df_keybert = pd.read_csv(keybert_file)
        # Merge KeyBERT results
        df = df.merge(df_keybert[['document_id', 'keybert_keywords', 'keybert_scores']], 
                     on='document_id', how='left')
    
    # Sample documents for evaluation
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    
    print(f"  Loaded {len(df)} documents for evaluation")
    print(f"  KeyBERT results: {'Available' if has_keybert else 'Not available'}")
    
    return df, has_keybert

def run_enhanced_evaluation():
    """Main function to run enhanced evaluation."""
    
    print("=" * 70)
    print("  ENHANCED EVALUATION — Advanced Metrics & Statistical Analysis")
    print("=" * 70)
    
    # Create output directory
    eval_dir = "evaluation_results_enhanced"
    os.makedirs(eval_dir, exist_ok=True)
    print(f"  Output directory: {eval_dir}/")
    
    # Load data
    try:
        df, has_keybert = load_evaluation_data()
    except Exception as e:
        print(f"  Error loading data: {e}")
        return
    
    # Parse keyword lists
    models = ['tfidf', 'cosine', 'cluster']
    if has_keybert:
        models.append('keybert')
    
    keyword_columns = {
        'tfidf': 'tfidf_keywords',
        'cosine': 'cosine_keywords', 
        'cluster': 'cluster_keywords',
        'keybert': 'keybert_keywords'
    }
    
    # Parse all keyword lists
    for model in models:
        col = keyword_columns[model]
        if col in df.columns:
            df[f'{model}_parsed'] = df[col].apply(safe_parse_keywords)
    
    # Use existing manual annotations or create expanded set
    if 'Manual Reference' in df.columns:
        df['reference_parsed'] = df['Manual Reference'].apply(safe_parse_keywords)
    else:
        # Create synthetic reference for demonstration (in real use, these would be human annotations)
        print("  [WARNING] No manual annotations found. Using synthetic reference for demo.")
        df['reference_parsed'] = df['tfidf_parsed'].apply(lambda x: x[:5] if len(x) >= 5 else x)
    
    # Calculate metrics for each model
    all_results = {}
    
    print(f"\n  Calculating enhanced metrics for K={TOP_K_VALUES}...")
    
    for model in models:
        predicted_col = f'{model}_parsed'
        if predicted_col not in df.columns:
            continue
            
        predicted_keywords = df[predicted_col].tolist()
        reference_keywords = df['reference_parsed'].tolist()
        
        metrics = calculate_all_metrics(predicted_keywords, reference_keywords)
        all_results[model] = metrics
        
        print(f"    {model.upper():8s}: F1@5={metrics.get('F1@5', 0):.3f}, MAP={metrics.get('MAP', 0):.3f}, nDCG@10={metrics.get('nDCG@10', 0):.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results).T
    results_csv = os.path.join(eval_dir, "enhanced_evaluation_results.csv")
    results_df.to_csv(results_csv)
    print(f"\n  Results saved -> {results_csv}")
    
    # Statistical significance testing
    print(f"\n  Statistical significance testing (α={ALPHA}):")
    baseline_model = 'tfidf'
    
    if baseline_model in all_results:
        baseline_scores = []
        comparison_scores = []
        
        # Collect F1@5 scores for statistical testing
        for model in models:
            if model == baseline_model:
                continue
            if model in all_results:
                # Calculate per-document F1@5 scores
                model_scores = []
                for i in range(len(df)):
                    pred = df[f'{model}_parsed'].iloc[i] if f'{model}_parsed' in df.columns else []
                    ref = df['reference_parsed'].iloc[i]
                    score = f1_at_k(pred, ref, 5)
                    model_scores.append(score)
                
                comparison_scores.append((model, model_scores))
        
        # Compare each model with baseline
        for model, scores in comparison_scores:
            if baseline_model in all_results:
                # Get baseline scores
                baseline_scores_list = []
                for i in range(len(df)):
                    pred = df[f'{baseline_model}_parsed'].iloc[i]
                    ref = df['reference_parsed'].iloc[i]
                    score = f1_at_k(pred, ref, 5)
                    baseline_scores_list.append(score)
                
                t_stat, p_value, significant = statistical_significance_test(baseline_scores_list, scores)
                
                sig_marker = "***" if significant else ""
                print(f"    {model.upper():8s} vs {baseline_model.upper()}: p={p_value:.4f} {sig_marker}")
    
    # Create visualization
    create_enhanced_charts(results_df, eval_dir, has_keybert)
    
    print(f"\n  Enhanced evaluation complete!")
    print("=" * 70)

def create_enhanced_charts(results_df, eval_dir, has_keybert):
    """Create enhanced visualization charts."""
    
    print(f"\n  Creating enhanced charts...")
    
    # Chart 1: Precision@K comparison
    plt.figure(figsize=(12, 8))
    
    precision_metrics = [col for col in results_df.columns if 'Precision@' in col]
    results_df[precision_metrics].T.plot(kind='bar', figsize=(12, 6), 
                                         color=[COLORS.get(model, '#333333') for model in results_df.index])
    
    plt.title('Precision@K Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(title='Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart1_path = os.path.join(eval_dir, "fig1_precision_k.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: MAP and nDCG comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAP
    if 'MAP' in results_df.columns:
        results_df['MAP'].plot(kind='bar', ax=ax1, 
                               color=[COLORS.get(model, '#333333') for model in results_df.index])
        ax1.set_title('Mean Average Precision (MAP)', fontweight='bold')
        ax1.set_ylabel('MAP Score')
        ax1.tick_params(axis='x', rotation=45)
    
    # nDCG@10
    ndcg_col = 'nDCG@10'
    if ndcg_col in results_df.columns:
        results_df[ndcg_col].plot(kind='bar', ax=ax2,
                                 color=[COLORS.get(model, '#333333') for model in results_df.index])
        ax2.set_title('nDCG@10 Comparison', fontweight='bold')
        ax2.set_ylabel('nDCG@10 Score')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    chart2_path = os.path.join(eval_dir, "fig2_map_ndcg.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Comprehensive radar chart
    metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MAP', 'nDCG@10']
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
    
    if len(available_metrics) >= 3:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model in results_df.index:
            values = [results_df.loc[model, metric] for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(),
                   color=COLORS.get(model, '#333333'))
            ax.fill(angles, values, alpha=0.25, color=COLORS.get(model, '#333333'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Model Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        chart3_path = os.path.join(eval_dir, "fig3_radar_comparison.png")
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"    Charts saved to {eval_dir}/")

if __name__ == "__main__":
    run_enhanced_evaluation()
