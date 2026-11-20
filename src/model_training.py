"""
Model Training Module for SMS Spam Detection.

This module implements end-to-end training for four classification algorithms:
1. Multinomial Naive Bayes (MultinomialNB)
2. Bernoulli Naive Bayes (BernoulliNB)
3. Logistic Regression (LogisticRegression)
4. Linear Support Vector Machine (Linear SVM)

Key Design Choices:
- Scikit-learn Pipeline architecture encapsulating:
  [Text Preprocessing] -> [TF-IDF Vectorization] -> [Classifier]
- Prevents data leakage by ensuring all fitting occurs strictly on training partitions.
- Stratified 5-Fold Cross Validation for hyperparameter tuning using GridSearchCV.
- Model persistence for inference in the Streamlit application.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from feature_extraction import (
    TextPreprocessorTransformer,
    get_tfidf_vectorizer,
    split_data
)
from data_loader import load_raw_data, clean_data


def get_baseline_pipelines():
    """
    Constructs baseline scikit-learn Pipelines for all four algorithms.
    Notice BernoulliNB is paired with binary=True vectorization.
    """
    pipelines = {
        "Multinomial Naive Bayes": Pipeline([
            ('preprocessor', TextPreprocessorTransformer()),
            ('vectorizer', get_tfidf_vectorizer(max_features=3000, binary=False)),
            ('classifier', MultinomialNB(alpha=1.0))
        ]),
        "Bernoulli Naive Bayes": Pipeline([
            ('preprocessor', TextPreprocessorTransformer()),
            ('vectorizer', get_tfidf_vectorizer(max_features=3000, binary=True)),
            ('classifier', BernoulliNB(alpha=1.0))
        ]),
        "Logistic Regression": Pipeline([
            ('preprocessor', TextPreprocessorTransformer()),
            ('vectorizer', get_tfidf_vectorizer(max_features=3000, binary=False)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ('preprocessor', TextPreprocessorTransformer()),
            ('vectorizer', get_tfidf_vectorizer(max_features=3000, binary=False)),
            ('classifier', SVC(kernel='linear', probability=True, random_state=42))
        ])
    }
    return pipelines


def train_baseline_models(X_train, y_train):
    """
    Fits baseline pipelines on the training set.
    """
    pipelines = get_baseline_pipelines()
    trained_models = {}

    print("\n--- Training Baseline Models ---")
    for name, pipeline in pipelines.items():
        print(f"[INFO] Training {name} baseline...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print(f"[SUCCESS] {name} baseline trained successfully.")

    return trained_models


def tune_models(X_train, y_train):
    """
    Performs Stratified 5-Fold Cross Validation hyperparameter tuning
    via GridSearchCV on the training set for all four models.
    """
    param_grids = {
        "Multinomial Naive Bayes": {
            'vectorizer__max_features': [2500, 3000, 4000],
            'classifier__alpha': [0.1, 0.5, 1.0, 1.5]
        },
        "Bernoulli Naive Bayes": {
            'vectorizer__max_features': [2500, 3000, 4000],
            'classifier__alpha': [0.05, 0.1, 0.5, 1.0]
        },
        "Logistic Regression": {
            'vectorizer__max_features': [2500, 3000, 4000],
            'classifier__C': [0.5, 1.0, 5.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        "Linear SVM": {
            'vectorizer__max_features': [2500, 3000, 4000],
            'classifier__C': [0.1, 0.5, 1.0, 2.0]
        }
    }

    base_pipelines = get_baseline_pipelines()
    tuned_models = {}
    best_params = {}

    print("\n--- Hyperparameter Tuning (5-Fold CV) ---")
    for name, pipeline in base_pipelines.items():
        print(f"[INFO] Running GridSearchCV for {name}...")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring='f1',  # Optimize for F1 score given class imbalance
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        tuned_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        print(f"[BEST] {name}: Best CV F1 = {grid_search.best_score_:.4f}")
        print(f"       Parameters: {grid_search.best_params_}")

    return tuned_models, best_params


def save_model(model, filepath="models/best_model.pkl"):
    """
    Serializes a trained pipeline using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model successfully saved to {filepath}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    baseline_models = train_baseline_models(X_train, y_train)
    tuned_models, best_params = tune_models(X_train, y_train)
