"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          PHRASES : Advanced Multi-Word Phrase Extraction             ║
╠══════════════════════════════════════════════════════════════════════╣
║ Advanced phrase extraction techniques:                                ║
║   • POS-based noun phrase chunking                                   ║
║   • Named Entity Recognition (NER)                                    ║
║   • Collocation detection using statistical methods                  ║
║   • Graph-based phrase extraction                                     ║
║   • Semantic phrase validation                                        ║
║                                                                      ║
║ Phrase types extracted:                                               ║
║   • Noun phrases (e.g., "machine learning")                          ║
║   • Named entities (e.g., "New York City")                          ║
║   • Technical terms (e.g., "artificial intelligence")               ║
║   • Verb phrases (e.g., "deep learning")                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# NLP imports
try:
    import nltk
    from nltk.chunk import ne_chunk
    from nltk.chunk import conlltags2tree
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try spaCy for advanced NER and phrase extraction
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = None  # Will be loaded on demand
except ImportError:
    SPACY_AVAILABLE = False

from paths import CLEANED_CSV, GEN_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
MIN_PHRASE_LENGTH = 2
MAX_PHRASE_LENGTH = 4
MIN_PHRASE_FREQ = 2
TOP_N_PHRASES = 15

# POS patterns for different phrase types
PHRASE_PATTERNS = {
    'noun_phrase': [('JJ', 'NN'), ('JJ', 'NNS'), ('NN', 'NN'), ('NN', 'NNS'), 
                    ('DT', 'JJ', 'NN'), ('DT', 'JJ', 'NNS')],
    'verb_phrase': [('VB', 'RB'), ('VB', 'DT'), ('VBG', 'NN'), ('VBN', 'NN')],
    'technical': [('JJ', 'JJ', 'NN'), ('NN', 'IN', 'NN'), ('JJ', 'NN', 'NN')]
}

def download_nltk_data():
    """Download required NLTK data."""
    if NLTK_AVAILABLE:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)

def load_spacy_model():
    """Load spaCy model for advanced NER."""
    global nlp
    if SPACY_AVAILABLE and nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            print("  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return False
    return SPACY_AVAILABLE and nlp is not None

def extract_noun_phrases(text):
    """Extract noun phrases using POS patterns."""
    if not NLTK_AVAILABLE:
        return []
    
    try:
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        phrases = []
        
        # Extract phrases based on patterns
        for pattern_name, patterns in PHRASE_PATTERNS.items():
            for pattern in patterns:
                for i in range(len(pos_tags) - len(pattern) + 1):
                    window = pos_tags[i:i+len(pattern)]
                    window_pos = [tag for word, tag in window]
                    window_words = [word for word, tag in window]
                    
                    if window_pos == list(pattern):
                        phrase = ' '.join(window_words)
                        if len(phrase.split()) >= MIN_PHRASE_LENGTH:
                            phrases.append(phrase)
        
        return phrases
    except Exception as e:
        return []

def extract_named_entities(text):
    """Extract named entities using NLTK NER."""
    if not NLTK_AVAILABLE:
        return []
    
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Named entity recognition
        tree = ne_chunk(pos_tags, binary=False)
        
        entities = []
        current_entity = []
        
        for subtree in tree:
            if hasattr(subtree, 'label'):
                # It's a named entity
                entity_words = [word for word, tag in subtree.leaves()]
                entity = ' '.join(entity_words)
                if len(entity.split()) >= MIN_PHRASE_LENGTH:
                    entities.append(entity)
        
        return entities
    except Exception as e:
        return []

def extract_spacy_entities(text):
    """Extract entities using spaCy for better accuracy."""
    if not load_spacy_model():
        return []
    
    try:
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if len(ent.text.split()) >= MIN_PHRASE_LENGTH:
                entities.append(ent.text)
        
        return entities
    except Exception as e:
        return []

def extract_collocations(text, n=2):
    """Extract collocations using statistical measures."""
    if not NLTK_AVAILABLE:
        return []
    
    try:
        tokens = [word.lower() for word in word_tokenize(text) 
                if word.isalpha() and word not in stopwords.words('english')]
        
        if n == 2:
            finder = BigramCollocationFinder.from_words(tokens)
            measures = BigramAssocMeasures()
        elif n == 3:
            finder = TrigramCollocationFinder.from_words(tokens)
            measures = TrigramAssocMeasures()
        else:
            return []
        
        # Filter by frequency
        finder.apply_freq_filter(MIN_PHRASE_FREQ)
        
        # Get top collocations by PMI
        collocations = finder.nbest(measures.pmi, TOP_N_PHRASES)
        
        return [' '.join(collocation) for collocation in collocations]
    except Exception as e:
        return []

def extract_graph_phrases(text):
    """Extract phrases using graph-based approach."""
    try:
        sentences = sent_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and len(w) > 2]
            
            # Create co-occurrence graph
            word_freq = Counter(words)
            
            # Find frequent word pairs
            for i in range(len(words) - 1):
                pair = (words[i], words[i+1])
                if word_freq[pair[0]] >= MIN_PHRASE_FREQ and word_freq[pair[1]] >= MIN_PHRASE_FREQ:
                    phrases.append(' '.join(pair))
        
        return list(set(phrases))
    except Exception as e:
        return []

def validate_phrases_semantically(phrases, context_text):
    """Validate phrases based on semantic coherence."""
    if not phrases:
        return []
    
    validated_phrases = []
    
    for phrase in phrases:
        # Check if phrase appears in context
        if phrase.lower() in context_text.lower():
            # Additional validation could be added here
            validated_phrases.append(phrase)
    
    return validated_phrases

def extract_all_phrases(text):
    """Extract all types of phrases from text."""
    if not text or len(text.strip()) < 20:
        return []
    
    all_phrases = []
    
    # 1. Noun phrases
    noun_phrases = extract_noun_phrases(text)
    all_phrases.extend(noun_phrases)
    
    # 2. Named entities (NLTK)
    ner_phrases = extract_named_entities(text)
    all_phrases.extend(ner_phrases)
    
    # 3. Named entities (spaCy)
    spacy_entities = extract_spacy_entities(text)
    all_phrases.extend(spacy_entities)
    
    # 4. Collocations
    bigram_collocations = extract_collocations(text, n=2)
    all_phrases.extend(bigram_collocations)
    
    trigram_collocations = extract_collocations(text, n=3)
    all_phrases.extend(trigram_collocations)
    
    # 5. Graph-based phrases
    graph_phrases = extract_graph_phrases(text)
    all_phrases.extend(graph_phrases)
    
    # Remove duplicates and validate
    unique_phrases = list(set(all_phrases))
    validated_phrases = validate_phrases_semantically(unique_phrases, text)
    
    # Filter by length and frequency
    filtered_phrases = []
    phrase_freq = Counter(validated_phrases)
    
    for phrase in validated_phrases:
        phrase_words = phrase.split()
        if (MIN_PHRASE_LENGTH <= len(phrase_words) <= MAX_PHRASE_LENGTH and
            phrase_freq[phrase] >= MIN_PHRASE_FREQ):
            filtered_phrases.append(phrase)
    
    return filtered_phrases

def score_phrases(phrases, text):
    """Score phrases based on multiple criteria."""
    if not phrases:
        return []
    
    scored_phrases = []
    word_freq = Counter(word.lower() for word in word_tokenize(text) if word.isalpha())
    
    for phrase in phrases:
        score = 0.0
        
        # 1. Length score (prefer medium-length phrases)
        length = len(phrase.split())
        if 2 <= length <= 3:
            score += 1.0
        elif length == 4:
            score += 0.8
        
        # 2. Frequency score
        phrase_freq = text.lower().count(phrase.lower())
        score += min(phrase_freq / 10.0, 1.0)
        
        # 3. Word importance score
        words = phrase.split()
        word_importance = sum(word_freq.get(word.lower(), 0) for word in words)
        score += min(word_importance / len(words) / 5.0, 1.0)
        
        # 4. Capitalization score (proper nouns)
        if any(word[0].isupper() for word in words):
            score += 0.5
        
        scored_phrases.append((phrase, score))
    
    # Sort by score
    scored_phrases.sort(key=lambda x: x[1], reverse=True)
    
    return scored_phrases

def run_phrase_extraction():
    """Main function to run phrase extraction on the dataset."""
    
    print("=" * 70)
    print("  ADVANCED PHRASE EXTRACTION — Multi-Word Keyword Detection")
    print("=" * 70)
    
    # Download required data
    download_nltk_data()
    
    # Load data
    if not os.path.exists(CLEANED_CSV):
        print(f"  [ERROR] '{CLEANED_CSV}' not found. Run preprocessing first.")
        return
    
    df = pd.read_csv(CLEANED_CSV)
    df["clean_review"] = df["clean_review"].fillna("")
    
    # Sample for faster processing
    sample_size = min(1000, len(df))
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  Processing {len(df)} documents")
    print(f"  spaCy available: {SPACY_AVAILABLE}")
    print(f"  NLTK available: {NLTK_AVAILABLE}")
    
    # Extract phrases for each document
    phrase_results = []
    phrase_stats = defaultdict(int)
    
    print(f"  Extracting multi-word phrases...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Phrase extraction"):
        text = row['review']
        
        # Extract all phrases
        phrases = extract_all_phrases(text)
        
        # Score phrases
        scored_phrases = score_phrases(phrases, text)
        
        # Get top phrases
        top_phrases = [phrase for phrase, score in scored_phrases[:TOP_N_PHRASES]]
        top_scores = [round(score, 4) for phrase, score in scored_phrases[:TOP_N_PHRASES]]
        
        phrase_results.append({
            'document_id': idx,
            'review': text,
            'clean_review': row['clean_review'],
            'sentiment': row['sentiment'],
            'extracted_phrases': top_phrases,
            'phrase_scores': top_scores,
            'total_phrases_found': len(phrases)
        })
        
        # Update statistics
        for phrase in phrases:
            phrase_stats[phrase] += 1
    
    # Save results
    df_phrases = pd.DataFrame(phrase_results)
    phrases_csv = os.path.join(GEN_DIR, "phrase_extraction_results.csv")
    df_phrases.to_csv(phrases_csv, index=False)
    
    print(f"  Saved -> {phrases_csv}")
    
    # Analysis and visualization
    analyze_phrase_extraction(df_phrases, phrase_stats)
    
    # Sample results
    print(f"\n  Sample phrase extraction results:")
    for i in range(min(3, len(df_phrases))):
        row = df_phrases.iloc[i]
        print(f"    Doc {i} ({row['sentiment']}):")
        print(f"      Phrases: {row['extracted_phrases'][:5]}")
        print(f"      Scores:  {row['phrase_scores'][:5]}")
    
    print(f"\n  Phrase extraction complete!")
    print("=" * 70)
    
    return df_phrases

def analyze_phrase_extraction(df_phrases, phrase_stats):
    """Analyze phrase extraction results and create visualizations."""
    
    print(f"\n  Analyzing phrase extraction...")
    
    # Create analysis directory
    analysis_dir = "phrase_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Phrase length distribution
    phrase_lengths = []
    for phrases in df_phrases['extracted_phrases']:
        for phrase in phrases:
            phrase_lengths.append(len(phrase.split()))
    
    plt.figure(figsize=(10, 6))
    plt.hist(phrase_lengths, bins=range(1, max(phrase_lengths)+2), 
             alpha=0.7, color='#4C72B0', edgecolor='black')
    plt.title('Phrase Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words in Phrase', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart1_path = os.path.join(analysis_dir, "phrase_length_distribution.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top phrases across all documents
    top_phrases = sorted(phrase_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    
    plt.figure(figsize=(12, 8))
    phrases, frequencies = zip(*top_phrases)
    plt.barh(range(len(phrases)), frequencies, color='#DD8452')
    plt.yticks(range(len(phrases)), phrases)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top 20 Multi-Word Phrases', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    chart2_path = os.path.join(analysis_dir, "top_phrases.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Phrases per document distribution
    phrases_per_doc = df_phrases['total_phrases_found']
    
    plt.figure(figsize=(10, 6))
    plt.hist(phrases_per_doc, bins=30, alpha=0.7, color='#55A868', edgecolor='black')
    plt.title('Phrases Found per Document', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Phrases', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart3_path = os.path.join(analysis_dir, "phrases_per_document.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Score distribution
    all_scores = []
    for scores in df_phrases['phrase_scores']:
        all_scores.extend(scores)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=30, alpha=0.7, color='#C44E52', edgecolor='black')
    plt.title('Phrase Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Phrase Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart4_path = os.path.join(analysis_dir, "score_distribution.png")
    plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis summary
    analysis_summary = {
        'total_documents': len(df_phrases),
        'total_unique_phrases': len(phrase_stats),
        'avg_phrases_per_doc': np.mean(phrases_per_doc),
        'avg_phrase_length': np.mean(phrase_lengths),
        'most_common_phrase': top_phrases[0][0] if top_phrases else None,
        'most_common_frequency': top_phrases[0][1] if top_phrases else 0
    }
    
    summary_df = pd.DataFrame([analysis_summary])
    summary_csv = os.path.join(analysis_dir, "phrase_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"    Analysis saved to {analysis_dir}/")
    print(f"    Total unique phrases: {analysis_summary['total_unique_phrases']}")
    print(f"    Average phrases per document: {analysis_summary['avg_phrases_per_doc']:.1f}")
    print(f"    Average phrase length: {analysis_summary['avg_phrase_length']:.1f} words")
    print(f"    Most common phrase: '{analysis_summary['most_common_phrase']}' ({analysis_summary['most_common_frequency']} times)")

if __name__ == "__main__":
    run_phrase_extraction()
