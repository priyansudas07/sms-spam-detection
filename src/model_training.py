"""
Model Training Module for SMS Spam Detection.

Trains all eight experimental combinations (2 Feature Extractors x 4 Classifiers):
1. TF-IDF + Multinomial Naive Bayes (MNB)
2. TF-IDF + Bernoulli Naive Bayes (BNB) [Binarized]
3. TF-IDF + Logistic Regression (LR)
4. TF-IDF + Linear SVM
5. CountVectorizer + Multinomial Naive Bayes (MNB)
6. CountVectorizer + Bernoulli Naive Bayes (BNB) [Binarized]
7. CountVectorizer + Logistic Regression (LR)
8. CountVectorizer + Linear SVM

Key Methodological Guarantees:
- Strict pipeline encapsulation preventing leakage.
- Stratified 5-Fold Cross Validation via GridSearchCV strictly on X_train.
- Real execution time tracking for each experiment.
- Serialization of trained pipelines into models/ directory.
"""

import os
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from feature_extraction import (
    TextPreprocessorTransformer,
    get_vectorizer,
    split_data
)
from data_loader import load_raw_data, clean_data


def build_pipeline(feature_tech: str, algorithm_name: str, max_features: int = 4000):
    """
    Constructs an end-to-end Pipeline:
    [TextPreprocessorTransformer] -> [Vectorizer (TF-IDF or Count)] -> [Classifier]
    """
    is_bernoulli = (algorithm_name == "Bernoulli Naive Bayes")
    vectorizer = get_vectorizer(
        technique=feature_tech,
        max_features=max_features,
        binary=is_bernoulli
    )

    if algorithm_name == "Multinomial Naive Bayes":
        classifier = MultinomialNB()
    elif algorithm_name == "Bernoulli Naive Bayes":
        classifier = BernoulliNB()
    elif algorithm_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm_name == "Linear SVM":
        classifier = SVC(kernel='linear', probability=True, random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return Pipeline([
        ('preprocessor', TextPreprocessorTransformer()),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])


def get_all_8_pipelines():
    """
    Instantiates the 8 experimental baseline pipelines.
    """
    feature_techniques = [
        ("TF-IDF", "tfidf"),
        ("CountVectorizer", "count")
    ]
    algorithms = [
        "Multinomial Naive Bayes",
        "Bernoulli Naive Bayes",
        "Logistic Regression",
        "Linear SVM"
    ]

    pipelines = {}
    for tech_display, tech_code in feature_techniques:
        for algo in algorithms:
            exp_key = f"{tech_display} + {algo}"
            pipelines[exp_key] = {
                "pipeline": build_pipeline(tech_code, algo),
                "feature_tech": tech_code,
                "algorithm": algo
            }
    return pipelines


def get_tuning_param_grid(algorithm_name: str):
    """
    Returns parameter search space for GridSearchCV.
    """
    if algorithm_name == "Multinomial Naive Bayes":
        return {
            'vectorizer__max_features': [3000, 4000],
            'classifier__alpha': [0.05, 0.1, 0.5, 1.0]
        }
    elif algorithm_name == "Bernoulli Naive Bayes":
        return {
            'vectorizer__max_features': [3000, 4000],
            'classifier__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    elif algorithm_name == "Logistic Regression":
        return {
            'vectorizer__max_features': [3000, 4000],
            'classifier__C': [0.5, 1.0, 5.0, 10.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    elif algorithm_name == "Linear SVM":
        return {
            'vectorizer__max_features': [3000, 4000],
            'classifier__C': [0.5, 1.0, 2.0]
        }
    return {}


def train_and_tune_all_8(X_train, y_train, models_dir="models"):
    """
    Executes 5-fold cross-validation GridSearchCV for all 8 experimental combinations.
    Measures and logs training time, best CV F1 score, and chosen parameters.
    """
    os.makedirs(models_dir, exist_ok=True)
    all_pipelines = get_all_8_pipelines()

    trained_models = {}
    training_records = []

    print(f"\n{'='*75}")
    print(f"TRAINING 8 EXPERIMENTAL COMBINATIONS (Stratified 5-Fold Cross Validation)")
    print(f"{'='*75}\n")

    for idx, (exp_name, info) in enumerate(all_pipelines.items(), 1):
        algo = info["algorithm"]
        tech_code = info["feature_tech"]
        base_pipe = info["pipeline"]
        param_grid = get_tuning_param_grid(algo)

        print(f"[{idx}/8] Optimizing: {exp_name}...")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        elapsed_sec = time.time() - start_time

        best_pipe = grid_search.best_estimator_
        trained_models[exp_name] = best_pipe

        # Serialize model
        file_slug = exp_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "")
        model_filepath = os.path.join(models_dir, f"{file_slug}.pkl")
        joblib.dump(best_pipe, model_filepath)

        record = {
            "Experiment": exp_name,
            "Feature_Extraction": "TF-IDF" if tech_code == "tfidf" else "CountVectorizer",
            "Algorithm": algo,
            "Best_CV_F1": round(float(grid_search.best_score_), 4),
            "Best_Params": str(grid_search.best_params_),
            "Train_Time_Sec": round(elapsed_sec, 2),
            "Model_Path": model_filepath
        }
        training_records.append(record)

        print(f"      ✓ Completed in {elapsed_sec:.2f}s | Best 5-Fold CV F1: {grid_search.best_score_:.4f}")
        print(f"      ✓ Parameters: {grid_search.best_params_}\n")

    df_train_summary = pd.DataFrame(training_records)
    print(f"{'='*75}")
    print("ALL 8 COMBINATIONS TRAINED AND OPTIMIZED SUCCESSFULLY")
    print(f"{'='*75}")

    return trained_models, df_train_summary


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    trained_models, train_summary = train_and_tune_all_8(X_train, y_train)
    print("\n--- Training & Cross-Validation Summary ---")
    print(train_summary[["Experiment", "Best_CV_F1", "Train_Time_Sec"]].to_string(index=False))
