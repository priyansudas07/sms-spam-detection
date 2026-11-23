"""
Statistical Significance & Confidence Interval Engine.
Part of SIGNAL — SMS Intelligence Lab.

Computes:
1. Exact McNemar's paired non-parametric tests across all model pairs.
2. 95% Wilson Score Confidence Intervals for Accuracy, Precision, and Recall.
3. 95% Non-parametric Bootstrap Confidence Intervals for F1-scores (B=1,000 resamples).
4. Exports structured tables and academic reporting payloads.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure src modules are resolvable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_raw_data, clean_data
from feature_extraction import TextPreprocessorTransformer  # noqa: F401


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> dict:
    """Computes exact Wilson Score interval for a proportion k/n."""
    if n == 0:
        return {"proportion": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    res = stats.binomtest(k, n)
    ci = res.proportion_ci(confidence_level=confidence, method="wilson")
    return {
        "proportion": round(float(k / n), 4),
        "ci_lower": round(float(ci.low), 4),
        "ci_upper": round(float(ci.high), 4)
    }


def compute_bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> dict:
    """Computes non-parametric percentile bootstrap CI for F1-score."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    f1_bootstraps = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y_true = y_true[indices]
        sample_y_pred = y_pred[indices]
        
        # Guard against zero-positive samples in resample
        if np.sum(sample_y_true) == 0 or np.sum(sample_y_pred) == 0:
            continue
        score = f1_score(sample_y_true, sample_y_pred, pos_label=1, zero_division=0)
        f1_bootstraps.append(score)

    if not f1_bootstraps:
        base_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        return {"f1": round(float(base_f1), 4), "ci_lower": round(float(base_f1), 4), "ci_upper": round(float(base_f1), 4)}

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(f1_bootstraps, lower_pct))
    ci_upper = float(np.percentile(f1_bootstraps, upper_pct))
    base_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    return {
        "f1": round(base_f1, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4)
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str,
    name_b: str
) -> dict:
    """
    Computes exact McNemar's test for paired binary classification models.
    Contingency table:
      b = Model A correct, Model B wrong
      c = Model B correct, Model A wrong
    Under H0: b and c are equally likely (p=0.5).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n_both_correct = int(np.sum(correct_a & correct_b))
    n_a_only = int(np.sum(correct_a & (~correct_b)))
    n_b_only = int(np.sum((~correct_a) & correct_b))
    n_both_wrong = int(np.sum((~correct_a) & (~correct_b)))

    total_discordant = n_a_only + n_b_only

    if total_discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        # Exact binomial test for discordant pairs
        binom_res = stats.binomtest(n_a_only, total_discordant, 0.5)
        p_value = float(binom_res.pvalue)
        # Asymptotic chi-squared statistic with Edwards continuity correction
        statistic = float(((abs(n_a_only - n_b_only) - 1.0) ** 2) / total_discordant) if total_discordant > 1 else 0.0

    is_significant = p_value < 0.05

    return {
        "model_a": name_a,
        "model_b": name_b,
        "contingency_table": {
            "both_correct": n_both_correct,
            "model_a_only_correct": n_a_only,
            "model_b_only_correct": n_b_only,
            "both_wrong": n_both_wrong
        },
        "discordant_pairs": total_discordant,
        "mcnemar_chi2_statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "is_statistically_significant": is_significant,
        "interpretation": (
            f"Statistically significant difference (p = {p_value:.4f} < 0.05)"
            if is_significant
            else f"No statistically significant difference (p = {p_value:.4f} >= 0.05); differences are within random sample variation"
        )
    }


def run_statistical_analysis(
    models_dir: str = os.path.join(PROJECT_ROOT, "models"),
    results_dir: str = os.path.join(PROJECT_ROOT, "reports", "results")
):
    """Executes full statistical hypothesis testing and CI derivation."""
    print("=" * 70)
    print("SIGNAL: STATISTICAL SIGNIFICANCE & CONFIDENCE INTERVAL ENGINE")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X = clean_df["message"]
    y = clean_df["label_num"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    y_test_arr = np.array(y_test)

    # 2. Pipeline mapping
    pipelines = {
        "TF-IDF + Bernoulli NB (Champion)": "tfidf_plus_bernoulli_naive_bayes.pkl",
        "CountVectorizer + Bernoulli NB": "countvectorizer_plus_bernoulli_naive_bayes.pkl",
        "CountVectorizer + Logistic Regression (Peak)": "countvectorizer_plus_logistic_regression.pkl",
        "TF-IDF + Logistic Regression": "tfidf_plus_logistic_regression.pkl",
        "CountVectorizer + Linear SVM": "countvectorizer_plus_linear_svm.pkl",
        "TF-IDF + Linear SVM": "tfidf_plus_linear_svm.pkl",
        "CountVectorizer + Multinomial NB": "countvectorizer_plus_multinomial_naive_bayes.pkl",
        "TF-IDF + Multinomial NB": "tfidf_plus_multinomial_naive_bayes.pkl",
    }

    preds_dict = {}
    print(f"\nEvaluating {len(pipelines)} pipelines on N = {len(y_test_arr)} test samples...")

    for name, pkl_file in pipelines.items():
        model_path = os.path.join(models_dir, pkl_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")
        model = joblib.load(model_path)
        preds_dict[name] = np.array(model.predict(X_test))

    # 3. Compute 95% Confidence Intervals
    ci_records = []
    total_test = len(y_test_arr)
    total_pos = int(np.sum(y_test_arr == 1))

    for name, y_pred in preds_dict.items():
        k_acc = int(np.sum(y_pred == y_test_arr))
        acc_ci = compute_wilson_ci(k_acc, total_test)

        # TP, FP, FN
        tp = int(np.sum((y_pred == 1) & (y_test_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_test_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_test_arr == 1)))

        pred_pos = tp + fp
        prec_ci = compute_wilson_ci(tp, pred_pos) if pred_pos > 0 else {"proportion": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        rec_ci = compute_wilson_ci(tp, total_pos)
        f1_ci = compute_bootstrap_f1_ci(y_test_arr, y_pred, n_bootstrap=1000, random_state=42)

        ci_records.append({
            "Pipeline Name": name,
            "Accuracy": acc_ci["proportion"],
            "Accuracy 95% CI": f"[{acc_ci['ci_lower']*100:.2f}%, {acc_ci['ci_upper']*100:.2f}%]",
            "Precision": prec_ci["proportion"],
            "Precision 95% CI": f"[{prec_ci['ci_lower']*100:.2f}%, {prec_ci['ci_upper']*100:.2f}%]",
            "Recall": rec_ci["proportion"],
            "Recall 95% CI": f"[{rec_ci['ci_lower']*100:.2f}%, {rec_ci['ci_upper']*100:.2f}%]",
            "F1-Score": f1_ci["f1"],
            "F1-Score 95% CI": f"[{f1_ci['ci_lower']*100:.2f}%, {f1_ci['ci_upper']*100:.2f}%]",
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    df_ci = pd.DataFrame(ci_records)
    print("\n95% CONFIDENCE INTERVALS TABLE:")
    print(df_ci[["Pipeline Name", "Accuracy 95% CI", "Precision 95% CI", "Recall 95% CI", "F1-Score 95% CI"]].to_string(index=False))

    # 4. Compute McNemar's Tests
    champion_name = "TF-IDF + Bernoulli NB (Champion)"
    champ_preds = preds_dict[champion_name]

    mcnemar_comparisons = []
    key_pairs = [
        ("TF-IDF + Bernoulli NB (Champion)", "CountVectorizer + Logistic Regression (Peak)"),
        ("TF-IDF + Bernoulli NB (Champion)", "CountVectorizer + Linear SVM"),
        ("TF-IDF + Bernoulli NB (Champion)", "TF-IDF + Linear SVM"),
        ("TF-IDF + Bernoulli NB (Champion)", "TF-IDF + Logistic Regression"),
        ("TF-IDF + Bernoulli NB (Champion)", "CountVectorizer + Bernoulli NB"),
        ("CountVectorizer + Logistic Regression (Peak)", "TF-IDF + Logistic Regression"),
        ("CountVectorizer + Linear SVM", "TF-IDF + Linear SVM"),
        ("TF-IDF + Multinomial NB", "TF-IDF + Bernoulli NB (Champion)"),
        ("CountVectorizer + Multinomial NB", "CountVectorizer + Bernoulli NB")
    ]

    for name_a, name_b in key_pairs:
        res = compute_mcnemar_test(y_test_arr, preds_dict[name_a], preds_dict[name_b], name_a, name_b)
        mcnemar_comparisons.append(res)
        print(f"\nMcNemar Test: {name_a} vs. {name_b}")
        print(f"  Discordant Pairs: {res['discordant_pairs']} | Chi2: {res['mcnemar_chi2_statistic']} | p-value: {res['p_value']:.4f}")
        print(f"  Result: {res['interpretation']}")

    # 5. Export to results
    os.makedirs(results_dir, exist_ok=True)
    ci_csv_path = os.path.join(results_dir, "confidence_intervals.csv")
    df_ci.to_csv(ci_csv_path, index=False)
    print(f"\n[SAVED] Confidence intervals exported to: {ci_csv_path}")

    mcnemar_json_path = os.path.join(results_dir, "mcnemar_significance_tests.json")
    with open(mcnemar_json_path, "w", encoding="utf-8") as f:
        json.dump(mcnemar_comparisons, f, indent=2)
    print(f"[SAVED] McNemar test results exported to: {mcnemar_json_path}")

    # Unified statistical payload
    unified_payload = {
        "confidence_intervals": ci_records,
        "mcnemar_comparisons": mcnemar_comparisons,
        "sample_size": total_test,
        "ham_test_samples": total_test - total_pos,
        "spam_test_samples": total_pos,
        "notes": (
            "Wilson score method applied for Accuracy, Precision, and Recall proportions. "
            "Non-parametric percentile bootstrap (B=1,000) applied for F1-score. "
            "Exact binomial McNemar's test applied for paired classifier discordance."
        )
    }
    stats_json_path = os.path.join(results_dir, "statistical_significance_results.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(unified_payload, f, indent=2)
    print(f"[SAVED] Unified statistical payload exported to: {stats_json_path}")

    return df_ci, mcnemar_comparisons


if __name__ == "__main__":
    run_statistical_analysis()
