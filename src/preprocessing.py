"""
Text Preprocessing Module for SMS Spam Detection.

This module implements principled NLP text normalization for SMS security analysis:
1. Lowercasing: Ensures uniform casing across all vocabulary.
2. Threat Indicator Token Normalization:
   - URLs / Web links -> 'url_token' (preserves critical phishing/smishing hook signals).
   - Email addresses -> 'email_token' (preserves contact spoofing lures).
   - Currency symbols ($, £, €, ₹) -> 'currency_token' (preserves monetary lure triggers).
3. Punctuation & Special Character Removal: Retains alphanumeric tokens and whitespace.
4. Whitespace Tokenization: Splits normalized messages into individual tokens.
5. Selective Stopword Removal with Negation Preservation:
   - Filters out non-informative grammatical noise ('the', 'at', 'is').
   - Explicitly PRESERVES meaningful negation operators ('not', 'no', 'never', 'neither',
     'without', 'against', 'cannot') to preserve sentiment and directive context.
6. Morphological Stemming: Reduces words to root forms using PorterStemmer ('running' -> 'run').
7. Whitespace Normalization: Eliminates redundant spacing and line breaks.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Critical negation terms preserved in anti-spam and sentiment analysis
NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "without", "against",
    "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent"
}

# Multi-layer stopword initialization with offline fallback
try:
    _raw_stopwords = set(stopwords.words('english'))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _raw_stopwords = set(ENGLISH_STOP_WORDS)

# Stopwords with negations preserved
STOPWORDS = _raw_stopwords - NEGATION_WORDS

STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Applies consistent NLP preprocessing rules across all experimental pipelines.

    Transformation Pipeline:
    1. Type validation and lowercase conversion.
    2. URL normalization: 'https?://...' or 'www....' -> 'url_token'.
    3. Email normalization: 'user@domain.com' -> 'email_token'.
    4. Currency normalization: '£', '$', '€', '₹' -> 'currency_token'.
    5. Punctuation removal: strips non-alphanumeric noise while retaining letters, numbers, and spaces.
    6. Tokenization: splits on whitespace.
    7. Stopword filtering with negation preservation.
    8. Porter Stemming.
    9. Whitespace rejoining.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase conversion
    text = text.lower()

    # Step 2: URL token replacement (Preserves smishing indicators)
    text = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', text)

    # Step 3: Email token replacement
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)

    # Step 4: Currency symbol replacement (Preserves financial prize lures)
    text = re.sub(r'[$£€₹]', ' currencytoken ', text)

    # Step 5: Remove non-alphanumeric punctuation (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Step 6: Whitespace tokenization
    tokens = text.split()

    # Step 7 & 8: Stopword filtering (with negations preserved) and Porter Stemming
    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]

    # Step 9: Normalized single-space rejoining
    return " ".join(cleaned_tokens)


def compute_transformation_stats(df, raw_col='message', processed_col='processed_message'):
    """
    Calculates corpus-level statistics before and after text transformation:
    - Character volume
    - Word token count
    - Unique vocabulary size
    """
    chars_before = df[raw_col].apply(len).sum()
    words_before = df[raw_col].apply(lambda x: len(str(x).split())).sum()
    vocab_before = len(set(" ".join(df[raw_col]).lower().split()))

    chars_after = df[processed_col].apply(len).sum()
    words_after = df[processed_col].apply(lambda x: len(str(x).split())).sum()
    vocab_after = len(set(" ".join(df[processed_col]).split()))

    stats = {
        "Characters": {
            "Before": int(chars_before),
            "After": int(chars_after),
            "Reduction %": round((1 - chars_after / chars_before) * 100, 2)
        },
        "Words": {
            "Before": int(words_before),
            "After": int(words_after),
            "Reduction %": round((1 - words_after / words_before) * 100, 2)
        },
        "Unique Vocab": {
            "Before": int(vocab_before),
            "After": int(vocab_after),
            "Reduction %": round((1 - vocab_after / vocab_before) * 100, 2)
        },
    }
    return stats


if __name__ == "__main__":
    sample = "USPS Notice: Your package could not be delivered today! Pay £1.99 fee at http://usps-track.com immediately."
    print("Original text :", sample)
    print("Processed text:", preprocess_text(sample))
