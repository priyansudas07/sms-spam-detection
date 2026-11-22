"""
Model Training & Validation-Based Selection Module for SMS Spam Detection.

Trains, tunes, and validates all eight experimental combinations (2 Feature Extractors x 4 Classifiers):
1. TF-IDF + Multinomial Naive Bayes (MNB)
2. TF-IDF + Bernoulli Naive Bayes (BNB) [Binarized]
3. TF-IDF + Logistic Regression (LR)
4. TF-IDF + Linear SVM (Platt Calibrated)
5. CountVectorizer + Multinomial Naive Bayes (MNB)
6. CountVectorizer + Bernoulli Naive Bayes (BNB) [Binarized]
7. CountVectorizer + Logistic Regression (LR)
8. CountVectorizer + Linear SVM (Platt Calibrated)

Key Methodological Guarantees:
- Strict pipeline encapsulation preventing data leakage across folds.
- Stratified 5-Fold Cross Validation via GridSearchCV strictly on X_train.
- SELECTION PROTOCOL: Model selection is determined strictly on the training partition
  using Mean 5-Fold CV F1-Score as the selection criterion. The test set remains untouched.
- Full provenance recording: Saves complete CV results and champion model metadata.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Resolve internal src modules
sys.path.append(os.path.dirname(__file__))
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
        classifier = BernoulliNB(binarize=0.0)
    elif algorithm_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm_name == "Linear SVM":
        # probability=True enables Platt scaling for calibrated posterior outputs
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
                "feature_display": tech_display,
                "algorithm": algo
            }
    return pipelines


def get_tuning_param_grid(algorithm_name: str):
    """
    Returns the parameter search space for GridSearchCV.
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


def train_tune_and_select(X_train, y_train, models_dir="models", results_dir="reports/results"):
    """
    1. Executes Stratified 5-Fold Cross Validation GridSearchCV across all 8 combinations.
    2. Measures training runtime, CV F1-score mean, and CV F1-score standard deviation.
    3. SELECTS the preferred model strictly using Training Cross-Validation results (Peak Mean CV F1).
    4. Serializes all 8 pipelines and persists the selected model metadata.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_pipelines = get_all_8_pipelines()
    trained_models = {}
    cv_records = []

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 80)
    print("TRAINING & TUNING 8 EXPERIMENTAL PIPELINES (Stratified 5-Fold Cross Validation)")
    print("=" * 80 + "\n")

    for idx, (exp_name, info) in enumerate(all_pipelines.items(), 1):
        algo = info["algorithm"]
        tech_code = info["feature_tech"]
        tech_display = info["feature_display"]
        base_pipe = info["pipeline"]
        param_grid = get_tuning_param_grid(algo)

        print(f"[{idx}/8] Optimizing: {exp_name}...")
        start_time = time.perf_counter()

        grid_search = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='f1',
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        grid_search.fit(X_train, y_train)
        elapsed_sec = time.perf_counter() - start_time

        best_pipe = grid_search.best_estimator_
        trained_models[exp_name] = best_pipe

        best_idx = grid_search.best_index_
        mean_cv_f1 = float(grid_search.best_score_)
        std_cv_f1 = float(grid_search.cv_results_['std_test_score'][best_idx])

        # Serialize individual pipeline
        file_slug = exp_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "")
        model_filepath = os.path.join(models_dir, f"{file_slug}.pkl")
        joblib.dump(best_pipe, model_filepath)

        record = {
            "Experiment": exp_name,
            "Feature_Extraction": tech_display,
            "Algorithm": algo,
            "Mean_CV_F1": round(mean_cv_f1, 4),
            "Std_CV_F1": round(std_cv_f1, 4),
            "Best_Params": str(grid_search.best_params_),
            "Train_Time_Sec": round(elapsed_sec, 2),
            "Model_File": f"{file_slug}.pkl"
        }
        cv_records.append(record)

        print(f"      ✓ Mean 5-Fold CV F1: {mean_cv_f1*100:.2f}% (±{std_cv_f1*100:.2f}%) in {elapsed_sec:.2f}s")
        print(f"      ✓ Optimal Hyperparameters: {grid_search.best_params_}\n")

    df_cv_summary = pd.DataFrame(cv_records)

    # Export cross-validation summary
    cv_csv_path = os.path.join(results_dir, "cv_results_summary.csv")
    cv_json_path = os.path.join(results_dir, "cv_results_summary.json")
    df_cv_summary.to_csv(cv_csv_path, index=False)
    with open(cv_json_path, "w", encoding="utf-8") as f:
        json.dump(cv_records, f, indent=2)

    # --- METHODOLOGICAL MODEL SELECTION ON TRAINING CV ONLY ---
    champion_idx = df_cv_summary["Mean_CV_F1"].idxmax()
    champion_row = df_cv_summary.iloc[champion_idx]
    champion_exp_name = champion_row["Experiment"]
    champion_pipeline = trained_models[champion_exp_name]

    # Save champion model
    champion_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(champion_pipeline, champion_model_path)

    selection_metadata = {
        "selected_model_name": champion_exp_name,
        "selection_metric": "Mean Stratified 5-Fold Cross-Validation F1-Score on X_train",
        "selection_rationale": (
            "Model selection was conducted strictly using training cross-validation performance "
            "to guarantee test-set isolation and eliminate data snooping bias. "
            f"{champion_exp_name} attained the peak Mean CV F1-Score ({champion_row['Mean_CV_F1']*100:.2f}%)."
        ),
        "mean_cv_f1": champion_row["Mean_CV_F1"],
        "std_cv_f1": champion_row["Std_CV_F1"],
        "optimal_hyperparameters": champion_row["Best_Params"],
        "model_file": champion_row["Model_File"],
        "persisted_path": champion_model_path,
        "selection_timestamp": datetime.now().isoformat()
    }

    metadata_path = os.path.join(models_dir, "selected_model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(selection_metadata, f, indent=2)

    print("=" * 80)
    print("MODEL SELECTION PROTOCOL COMPLETE (TRAINING CV DRIVEN)")
    print(f"★ SELECTED CHAMPION MODEL: {champion_exp_name}")
    print(f"  Selection Metric: Mean 5-Fold CV F1 = {champion_row['Mean_CV_F1']*100:.2f}% (±{champion_row['Std_CV_F1']*100:.2f}%)")
    print(f"  Saved metadata to: {metadata_path}")
    print(f"  Serialized champion to: {champion_model_path}")
    print("=" * 80 + "\n")

    return trained_models, df_cv_summary, selection_metadata


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, report = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    trained_models, cv_summary, metadata = train_tune_and_select(X_train, y_train)
    print("\n--- Cross-Validation Training Leaderboard ---")
    print(cv_summary[["Experiment", "Mean_CV_F1", "Std_CV_F1", "Train_Time_Sec"]].to_string(index=False))
