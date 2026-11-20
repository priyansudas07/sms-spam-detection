"""
Feature Extraction Module for SMS Spam Detection.

This module handles:
1. Stratified 80:20 Train-Test splitting to preserve class balance without leakage.
2. Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).
3. Custom scikit-learn compatible preprocessor transformer.
4. Separate vectorizer configurations tailored to model families:
   - Standard TF-IDF for Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
   - Binarized TF-IDF / Binary Vectorizer for Bernoulli Naive Bayes (modeling presence/absence).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import preprocess_text


class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that applies text preprocessing
    consistently across train and test sets inside a Pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(preprocess_text)
        return [preprocess_text(text) for text in X]


def split_data(df, text_col='message', target_col='label_num', test_size=0.20, random_state=42):
    """
    Splits dataset into stratified train and test partitions.
    Preserves the 87.6% Ham / 12.4% Spam proportion across both folds.
    """
    X = df[text_col]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"[INFO] Train class balance: Ham = {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%), Spam = {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"[INFO] Test class balance:  Ham = {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%), Spam = {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def get_tfidf_vectorizer(max_features=3000, binary=False):
    """
    Returns a TF-IDF vectorizer configured for numerical text representation.
    - binary=False: standard continuous TF-IDF weights (used for MNB, LR, SVM)
    - binary=True: binarized TF-IDF (1 for presence, 0 for absence, used for BNB)
    """
    return TfidfVectorizer(
        max_features=max_features,
        binary=binary,
        sublinear_tf=True,
        ngram_range=(1, 2)  # unigrams and bigrams to capture phrases like 'free entry', 'claim now'
    )


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data

    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    vectorizer = get_tfidf_vectorizer(max_features=3000)
    transformer = TextPreprocessorTransformer()

    X_train_clean = transformer.transform(X_train)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    print(f"[INFO] TF-IDF Matrix shape: {X_train_tfidf.shape}")
    print(f"[INFO] Sample vocabulary features: {list(vectorizer.get_feature_names_out()[:15])}")
