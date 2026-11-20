"""
Text Preprocessing Module for SMS Spam Detection.

This module implements natural language processing (NLP) text normalization:
1. Lowercasing: Ensures uniform casing so 'Free', 'FREE', and 'free' are treated as identical tokens.
2. Special Character & Punctuation Removal: Strips symbols (!, $, %, @, etc.) that introduce noise.
3. Tokenization: Splits text messages into individual word tokens.
4. Stopword Removal: Filters out high-frequency grammatical words ('the', 'is', 'at') with low semantic value.
5. Stemming: Reduces words to their morphological root/base form using PorterStemmer (e.g., 'running' -> 'run').
6. Whitespace Normalization: Eliminates redundant spaces, tabs, and line breaks.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK resources are available
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Applies sequential NLP preprocessing steps on a raw text string.

    Steps:
    1. Check for null or non-string input.
    2. Convert to lowercase.
    3. Remove punctuation and non-alphanumeric characters (keeping spaces).
    4. Tokenize into individual words by whitespace.
    5. Remove English stopwords.
    6. Apply Porter Stemmer to reduce words to root form.
    7. Rejoin into a single normalized space-separated string.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase conversion
    text = text.lower()

    # Step 2: Remove URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Step 3: Remove punctuation and non-alphanumeric characters
    # Keep alphanumeric characters and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Step 4: Tokenization by whitespace
    tokens = text.split()

    # Step 5 & 6: Stopword removal and Stemming
    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]

    # Step 7: Whitespace normalization and rejoining
    return " ".join(cleaned_tokens)


def compute_transformation_stats(df, raw_col='message', processed_col='processed_message'):
    """
    Calculates corpus-level statistics before and after text transformation:
    - Character count
    - Word count
    - Vocabulary size (unique words)
    """
    chars_before = df[raw_col].apply(len).sum()
    words_before = df[raw_col].apply(lambda x: len(str(x).split())).sum()
    vocab_before = len(set(" ".join(df[raw_col]).lower().split()))

    chars_after = df[processed_col].apply(len).sum()
    words_after = df[processed_col].apply(lambda x: len(str(x).split())).sum()
    vocab_after = len(set(" ".join(df[processed_col]).split()))

    stats = {
        "Characters": {"Before": int(chars_before), "After": int(chars_after), "Reduction %": round((1 - chars_after / chars_before) * 100, 2)},
        "Words": {"Before": int(words_before), "After": int(words_after), "Reduction %": round((1 - words_after / words_before) * 100, 2)},
        "Unique Vocab": {"Before": int(vocab_before), "After": int(vocab_after), "Reduction %": round((1 - vocab_after / vocab_before) * 100, 2)},
    }
    return stats


if __name__ == "__main__":
    sample = "WINNER!! As a valued customer, you have been selected to win £900 cash prize! Text CLAIM to 87121."
    print("Original text :", sample)
    print("Processed text:", preprocess_text(sample))
