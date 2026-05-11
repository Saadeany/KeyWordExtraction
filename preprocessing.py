"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          STEP 1 : Text Preprocessing                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Input  : IMDB-Dataset.csv                                           ║
║  Output : generated/IMDB_cleaned.csv                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import nltk
import pandas as pd
from paths import DATASET, CLEANED_CSV, GEN_DIR

# ── Download required NLTK data (safe to run multiple times) ──────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4',  quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize

print("=" * 60)
print("  STEP 1 — Text Preprocessing")
print("=" * 60)

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"\n[ERROR] '{DATASET}' not found.\n"
        f"  -> Place 'IMDB-Dataset.csv' in the same folder as this script."
    )

df = pd.read_csv(DATASET)
print(f"  Loaded dataset   : {len(df):,} rows")
print(f"  Columns          : {list(df.columns)}")
print(f"  Output folder    : {GEN_DIR}/")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Enhanced stopwords for movie reviews
movie_stopwords = {
    'movie', 'film', 'story', 'character', 'characters', 'scene', 'scenes',
    'plot', 'ending', 'beginning', 'start', 'end', 'way', 'thing', 'things',
    'something', 'anything', 'nothing', 'everything', 'someone', 'anyone',
    'everyone', 'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went',
    'get', 'got', 'see', 'saw', 'look', 'looked', 'seem', 'seemed'
}
enhanced_stopwords = stop_words.union(movie_stopwords)

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to wordnet tags for better lemmatization."""
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

def extract_phrases(tokens, pos_tags):
    """Extract meaningful phrases using POS patterns."""
    phrases = []
    current_phrase = []
    
    # Pattern for adjective-noun and noun-noun phrases
    for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
        # Start new phrase with adjective or noun
        if pos.startswith(('JJ', 'NN')) and not current_phrase:
            current_phrase.append(token)
        # Continue phrase with adjective or noun
        elif pos.startswith(('JJ', 'NN')) and current_phrase:
            current_phrase.append(token)
        # End phrase and start new one
        elif pos.startswith(('JJ', 'NN')) and current_phrase:
            if len(current_phrase) > 1:
                phrases.append(' '.join(current_phrase))
            current_phrase = [token]
        # End phrase
        else:
            if len(current_phrase) > 1:
                phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    # Add final phrase
    if len(current_phrase) > 1:
        phrases.append(' '.join(current_phrase))
    
    return phrases

def clean_text(text: str) -> str:
    """
    Enhanced preprocessing pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove punctuation and numbers
      4. Tokenize
      5. POS tagging
      6. Lemmatization with POS information
      7. Remove enhanced stopwords and short tokens
      8. Extract meaningful phrases
      9. Re-join
    """
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize and POS tag
    tokens = nltk.word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Lemmatize with POS information and filter
    lemmatized_tokens = []
    for token, pos in zip(tokens, pos_tags):
        if (token not in enhanced_stopwords and 
            len(token) > 2 and 
            not token.isdigit()):
            
            wordnet_pos = get_wordnet_pos(pos)
            lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
            lemmatized_tokens.append(lemma)
    
    # Extract phrases from original tokens
    phrases = extract_phrases(tokens, pos_tags)
    
    # Combine tokens and phrases
    all_tokens = lemmatized_tokens + [p.replace(' ', '_') for p in phrases]
    
    return " ".join(all_tokens)

print("\n  Cleaning reviews — may take ~1 min on 50k rows...")
df['clean_review'] = df['review'].apply(clean_text)

before = len(df)
df = df[df['clean_review'].str.strip() != ""].reset_index(drop=True)
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped} empty documents after cleaning.")

df.to_csv(CLEANED_CSV, index=False)

print(f"\n  Clean documents saved : {len(df):,}")
print(f"  Saved -> {CLEANED_CSV}")
print(f"\n  Sample:")
print(f"  RAW  : {df['review'].iloc[0][:80]}...")
print(f"  CLEAN: {df['clean_review'].iloc[0][:80]}...")
print("\n  STEP 1 COMPLETE")
print("=" * 60)