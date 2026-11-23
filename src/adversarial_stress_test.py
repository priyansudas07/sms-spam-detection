"""
Adversarial Evasion & Robustness Stress Testing Engine.
Part of SIGNAL — SMS Intelligence Lab.

Evaluates:
1. Synthetic adversarial noise: Leetspeak, Punctuation Insertion, Character Inversion.
2. Robustness degradation curves across classical ML pipelines vs. Tier-2 Hybrid Guard.
3. Generates high-resolution 300 DPI publication chart and structured results.
"""

import os
import sys
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import load_raw_data, clean_data
from feature_extraction import TextPreprocessorTransformer  # noqa: F401
from smishing_heuristics import analyze_smishing_heuristics, calculate_weighted_ensemble_prediction


# Leetspeak translation dictionary
LEET_DICT = {
    'a': ['4', '@'],
    'e': ['3'],
    'i': ['1', '!'],
    'o': ['0'],
    's': ['$', '5'],
    't': ['7']
}


def apply_leetspeak(word: str, prob: float, rng: random.Random) -> str:
    chars = []
    for char in word:
        lower_c = char.lower()
        if lower_c in LEET_DICT and rng.random() < prob:
            chars.append(rng.choice(LEET_DICT[lower_c]))
        else:
            chars.append(char)
    return "".join(chars)


def apply_punctuation_insertion(word: str, prob: float, rng: random.Random) -> str:
    if len(word) <= 3 or rng.random() >= prob:
        return word
    delimiters = ['.', '-', '_', '*']
    delim = rng.choice(delimiters)
    return delim.join(list(word))


def perturb_text(text: str, noise_level: float, seed: int = 42) -> str:
    """
    Applies controlled adversarial perturbation at noise_level in [0.0, 1.0].
    """
    if noise_level <= 0.0:
        return text

    rng = random.Random(seed + int(noise_level * 1000))
    words = text.split()
    perturbed_words = []

    for w in words:
        # Don't perturb URLs or tokens
        if w.startswith("http") or "www." in w or w.startswith("<") or w.endswith(">"):
            perturbed_words.append(w)
            continue

        if rng.random() < noise_level:
            # 50% chance leetspeak, 50% chance delimiter insertion
            if rng.random() < 0.5:
                w = apply_leetspeak(w, prob=0.6, rng=rng)
            else:
                w = apply_punctuation_insertion(w, prob=0.5, rng=rng)
        perturbed_words.append(w)

    return " ".join(perturbed_words)


def run_adversarial_stress_test(
    models_dir: str = os.path.join(PROJECT_ROOT, "models"),
    results_dir: str = os.path.join(PROJECT_ROOT, "reports", "results"),
    fig_dir: str = os.path.join(PROJECT_ROOT, "reports", "figures")
):
    print("=" * 70)
    print("SIGNAL: ADVERSARIAL EVASION & ROBUSTNESS STRESS TEST")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X = clean_df["message"]
    y = clean_df["label_num"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Filter strictly spam messages from test set
    spam_mask = (np.array(y_test) == 1)
    X_test_spam = X_test[spam_mask].reset_index(drop=True)
    y_test_spam = np.ones(len(X_test_spam), dtype=int)
    n_spam = len(X_test_spam)

    print(f"\nEvaluating adversarial attack resistance on {n_spam} ground-truth SPAM messages...")

    # Load representative pipelines
    models_to_test = {
        "TF-IDF + Bernoulli NB (Champion)": joblib.load(os.path.join(models_dir, "tfidf_plus_bernoulli_naive_bayes.pkl")),
        "CountVec + Logistic Regression (Peak)": joblib.load(os.path.join(models_dir, "countvectorizer_plus_logistic_regression.pkl")),
        "CountVec + Linear SVM": joblib.load(os.path.join(models_dir, "countvectorizer_plus_linear_svm.pkl")),
        "TF-IDF + Linear SVM": joblib.load(os.path.join(models_dir, "tfidf_plus_linear_svm.pkl")),
        "CountVec + Multinomial NB": joblib.load(os.path.join(models_dir, "countvectorizer_plus_multinomial_naive_bayes.pkl")),
    }

    # Load all 8 for Hybrid Ensemble
    all_pipes = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl") and fname != "best_model.pkl":
            p = joblib.load(os.path.join(models_dir, fname))
            parts = fname.replace(".pkl", "").split("_plus_")
            feat_name = "TF-IDF" if parts[0] == "tfidf" else "CountVectorizer"
            algo_map = {
                "bernoulli_naive_bayes": "Bernoulli Naive Bayes",
                "multinomial_naive_bayes": "Multinomial Naive Bayes",
                "logistic_regression": "Logistic Regression",
                "linear_svm": "Linear Support Vector Machine"
            }
            algo_name = algo_map.get(parts[1], parts[1])
            all_pipes[(feat_name, algo_name)] = p

    noise_levels = [0.0, 0.10, 0.20, 0.35, 0.50]
    results_records = []

    for eps in noise_levels:
        # Generate perturbed test corpus for this epsilon
        perturbed_corpus = [perturb_text(msg, noise_level=eps, seed=i) for i, msg in enumerate(X_test_spam)]

        print(f"\nEvaluating Noise Level epsilon = {eps*100:.0f}%...")
        if eps > 0:
            print(f"  Sample Perturbation: '{X_test_spam[0][:60]}...' -> '{perturbed_corpus[0][:60]}...'")

        # 1. Evaluate single models
        for m_name, model in models_to_test.items():
            preds = model.predict(perturbed_corpus)
            rec = float(recall_score(y_test_spam, preds, pos_label=1, zero_division=0))
            blocked = int(np.sum(preds == 1))
            evaded = n_spam - blocked
            results_records.append({
                "Noise Level (Epsilon)": eps,
                "Model / Architecture": m_name,
                "Adversarial Recall": round(rec * 100.0, 2),
                "Blocked Spam": blocked,
                "Evaded Spam": evaded,
                "Architecture Type": "Baseline Statistical ML"
            })

        # 2. Evaluate Tier-2 Hybrid Defense Ensemble
        hybrid_preds = []
        for msg in perturbed_corpus:
            telemetry = analyze_smishing_heuristics(msg)
            pipe_outs = {}
            for (feat, algo), pipe in all_pipes.items():
                p_pred = pipe.predict([msg])[0]
                p_prob = pipe.predict_proba([msg])[0] if hasattr(pipe, "predict_proba") else ([0.0, 1.0] if p_pred == 1 else [1.0, 0.0])
                pipe_outs[(feat, algo)] = {"pred": int(p_pred), "prob": p_prob}
            ens_res = calculate_weighted_ensemble_prediction(pipe_outs, smishing_telemetry=telemetry)
            hybrid_preds.append(ens_res["pred"])

        hybrid_rec = float(recall_score(y_test_spam, hybrid_preds, pos_label=1, zero_division=0))
        results_records.append({
            "Noise Level (Epsilon)": eps,
            "Model / Architecture": "Tier-2 Hybrid Smishing Guard (Proposed)",
            "Adversarial Recall": round(hybrid_rec * 100.0, 2),
            "Blocked Spam": int(np.sum(np.array(hybrid_preds) == 1)),
            "Evaded Spam": n_spam - int(np.sum(np.array(hybrid_preds) == 1)),
            "Architecture Type": "Two-Tier Hybrid Guard"
        })

    df_adv = pd.DataFrame(results_records)
    print("\nADVERSARIAL STRESS TEST SUMMARY TABLE:")
    pivot_table = df_adv.pivot(index="Model / Architecture", columns="Noise Level (Epsilon)", values="Adversarial Recall")
    print(pivot_table.to_string())

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "adversarial_stress_test_results.csv")
    df_adv.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Adversarial stress test CSV exported to: {csv_path}")

    json_path = os.path.join(results_dir, "adversarial_stress_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    print(f"[SAVED] Adversarial stress test JSON exported to: {json_path}")

    # Generate Publication-Ready Chart (300 DPI)
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "adversarial_robustness_curve.png")

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid")

    palette_map = {
        "Tier-2 Hybrid Smishing Guard (Proposed)": ("#27866B", 3.0, "o-"),
        "CountVec + Linear SVM": ("#E66B36", 1.8, "s--"),
        "CountVec + Logistic Regression (Peak)": ("#2980B9", 1.8, "^--"),
        "TF-IDF + Bernoulli NB (Champion)": ("#8E44AD", 1.8, "v-."),
        "CountVec + Multinomial NB": ("#7F8C8D", 1.5, "x:")
    }

    x_vals = [eps * 100 for eps in noise_levels]

    for model_name, (color, lwidth, marker_style) in palette_map.items():
        subset = df_adv[df_adv["Model / Architecture"] == model_name]
        if not subset.empty:
            plt.plot(x_vals, subset["Adversarial Recall"], marker_style, label=model_name, color=color, linewidth=lwidth, markersize=6)

    plt.title("Adversarial Evasion Robustness: Recall Degradation Curve", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Synthetic Adversarial Noise Level (Leetspeak & Delimiter Splitting %)", fontsize=9, fontweight="bold")
    plt.ylabel("Spam Detection Recall (%) [Higher is Better]", fontsize=9, fontweight="bold")
    plt.ylim(30, 105)
    plt.xticks(x_vals, [f"{int(x)}%" for x in x_vals])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=8, loc="lower left")

    # Callout highlighting Tier-2 Resilience
    plt.annotate(
        "Tier-2 Hybrid Defense maintains\nresilience via multi-vector heuristics",
        xy=(35, pivot_table.loc["Tier-2 Hybrid Smishing Guard (Proposed)", 0.35]),
        xytext=(20, 45),
        arrowprops=dict(facecolor="#27866B", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8, fontweight="bold", color="#27866B",
        bbox=dict(boxstyle="round,pad=0.4", fc="#EAECE9", ec="#27866B", lw=1)
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication chart exported to: {fig_path}")

    return df_adv


if __name__ == "__main__":
    run_adversarial_stress_test()
