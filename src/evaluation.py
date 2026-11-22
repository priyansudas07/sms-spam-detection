"""
Evaluation & Visualization Module for SMS Spam Detection.

Evaluates all eight experimental pipelines on the reserved untouched test set (1,032 samples).
Methodological Distinctions:
- Identifies the Validation-Selected Model (Champion) chosen strictly on training CV.
- Reports full comparative test-set metrics for all 8 pipelines without data snooping.
- Generates:
  1. reports/results/results_summary.csv & results_summary.json
  2. reports/results/experiment_config.json
  3. Visual comparison figures:
     - accuracy_comparison.png
     - precision_comparison.png
     - recall_comparison.png
     - f1_comparison.png
     - feature_extraction_comparison.png
     - confusion_matrices_all8.png (8-panel annotated heatmap grid)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

sys.path.append(os.path.dirname(__file__))
from data_loader import load_raw_data, clean_data
from feature_extraction import split_data
from model_training import train_tune_and_select


def evaluate_pipeline(model, X_test, y_test):
    """
    Evaluates a single model pipeline on the unseen test set.
    Positive label = 1 (Spam).
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Confusion_Matrix": cm.tolist()
    }


def run_comprehensive_evaluation(models_dict, X_test, y_test, results_dir="reports/results", fig_dir="reports/figures", models_dir="models"):
    """
    Evaluates all 8 pipelines on untouched test data, exports machine-readable results,
    generates publication charts, and verifies the validation-selected model.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load validation selection metadata if available
    metadata_path = os.path.join(models_dir, "selected_model_metadata.json")
    champion_name = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            champion_name = meta.get("selected_model_name")

    records = []
    cm_dict = {}

    for exp_name, model in models_dict.items():
        metrics = evaluate_pipeline(model, X_test, y_test)
        tech = "TF-IDF" if "TF-IDF" in exp_name else "CountVectorizer"
        algo = exp_name.replace("TF-IDF + ", "").replace("CountVectorizer + ", "")

        is_champion = (exp_name == champion_name)

        rec = {
            "Experiment": exp_name,
            "Feature_Extraction": tech,
            "Algorithm": algo,
            "Is_Validation_Champion": is_champion,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "TN": metrics["TN"],
            "FP (False Alarm)": metrics["FP"],
            "FN (Missed Spam)": metrics["FN"],
            "TP (Blocked Spam)": metrics["TP"]
        }
        records.append(rec)
        cm_dict[exp_name] = np.array(metrics["Confusion_Matrix"])

    df_results = pd.DataFrame(records)

    # Export machine-readable evaluation results
    csv_path = os.path.join(results_dir, "results_summary.csv")
    json_path = os.path.join(results_dir, "results_summary.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Test evaluation exported to {csv_path} and {json_path}")

    # Set aesthetic plotting style
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # --- 1. Metric-Specific Comparison Charts ---
    metrics_to_plot = [
        ("Accuracy", "accuracy_comparison.png", "Test Accuracy across 8 Factorial Experiments", (94, 99)),
        ("Precision", "precision_comparison.png", "Spam Precision across 8 Experiments (Minimizing False Alarms)", (90, 100)),
        ("Recall", "recall_comparison.png", "Spam Recall across 8 Experiments (Spam Sensitivity)", (80, 95)),
        ("F1-Score", "f1_comparison.png", "Spam F1-Score across 8 Experiments (Harmonic Mean)", (85, 95))
    ]

    for metric_col, filename, title, ylim in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        palette = {"TF-IDF": "#2b5c8f", "CountVectorizer": "#d9534f"}
        ax = sns.barplot(
            data=df_results,
            x="Algorithm",
            y=df_results[metric_col] * 100,
            hue="Feature_Extraction",
            palette=palette
        )
        plt.title(title, fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Classifier Algorithm", fontweight='semibold')
        plt.ylabel(f"{metric_col} (%)", fontweight='semibold')
        plt.ylim(ylim)
        plt.legend(title="Feature Extraction", loc="lower right")

        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h + 0.1),
                            ha='center', va='bottom', fontsize=8.5, fontweight='semibold')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, filename), dpi=300)
        plt.close()

    # --- 2. Macro Feature Extraction Comparison Chart ---
    df_melt = pd.melt(
        df_results,
        id_vars=["Experiment", "Feature_Extraction", "Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    df_melt["Score (%)"] = df_melt["Score"] * 100

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=df_melt,
        x="Metric",
        y="Score (%)",
        hue="Feature_Extraction",
        palette=["#2b5c8f", "#d9534f"],
        errorbar=None
    )
    plt.title("Aggregated Performance: TF-IDF vs. CountVectorizer (Averaged across 4 Classifiers)",
              fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Score (%)", fontweight='semibold')
    plt.ylim(85, 100)
    plt.legend(title="Technique", loc="lower right")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h + 0.2),
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_extraction_comparison.png"), dpi=300)
    plt.close()

    # --- 3. 8-Panel Annotated Confusion Matrix Grid ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for idx, (exp_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        total = cm.sum()
        cm_pct = (cm / total) * 100

        labels = [
            [f"True Ham (TN)\n{cm[0,0]}\n({cm_pct[0,0]:.1f}%)", f"False Spam (FP)\n{cm[0,1]}\n({cm_pct[0,1]:.1f}%)"],
            [f"Missed Spam (FN)\n{cm[1,0]}\n({cm_pct[1,0]:.1f}%)", f"True Spam (TP)\n{cm[1,1]}\n({cm_pct[1,1]:.1f}%)"]
        ]
        cmap = "Blues" if "TF-IDF" in exp_name else "Reds"
        sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=False, ax=ax,
                    annot_kws={"fontsize": 10, "fontweight": "semibold"})

        row_data = df_results[df_results["Experiment"] == exp_name].iloc[0]
        champ_tag = " ★ [CV CHAMPION]" if exp_name == champion_name else ""
        ax.set_title(f"{exp_name}{champ_tag}\n(Acc: {row_data['Accuracy']*100:.2f}% | F1: {row_data['F1-Score']*100:.2f}%)",
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Predicted Label", fontweight='semibold')
        ax.set_ylabel("True Label", fontweight='semibold')
        ax.set_xticklabels(["Ham", "Spam"])
        ax.set_yticklabels(["Ham", "Spam"])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices_all8.png"), dpi=300)
    plt.close()

    # --- 4. Report Model Selection vs. Test Set Generalization ---
    best_test_idx = df_results["F1-Score"].idxmax()
    best_test_row = df_results.iloc[best_test_idx]
    best_test_name = best_test_row["Experiment"]

    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION SUMMARY (UNTOUCHED TEST SET: N=1,032)")
    print("=" * 80)
    if champion_name:
        champ_test = df_results[df_results["Experiment"] == champion_name].iloc[0]
        print(f"★ VALIDATION-SELECTED MODEL (CHAMPION): {champion_name}")
        print(f"    Selected Via: Mean 5-Fold Stratified CV F1 on X_train (Zero Test Leakage)")
        print(f"    Test Accuracy : {champ_test['Accuracy']*100:.2f}%")
        print(f"    Test Precision: {champ_test['Precision']*100:.2f}% (FP: {champ_test['FP (False Alarm)']}/904)")
        print(f"    Test Recall   : {champ_test['Recall']*100:.2f}% (FN: {champ_test['FN (Missed Spam)']}/128)")
        print(f"    Test F1-Score : {champ_test['F1-Score']*100:.2f}%\n")

    print(f"PEAK TEST SET SCORE: {best_test_name}")
    print(f"    Test Accuracy : {best_test_row['Accuracy']*100:.2f}%")
    print(f"    Test F1-Score : {best_test_row['F1-Score']*100:.2f}%")
    if champion_name == best_test_name:
        print("    ✓ Complete Alignment: Training CV selection precisely matched the best test set performer!")
    else:
        print("    ℹ Variance Note: High test score variance observed between CV selection and test partition.")
    print("=" * 80 + "\n")

    return df_results, champion_name, best_test_name


def save_experiment_config(results_dir="reports/results", random_state=42, test_size=0.20):
    """
    Saves the complete reproducible configuration of the experiment.
    """
    os.makedirs(results_dir, exist_ok=True)
    config = {
        "project_title": "Comparative Analysis of ML Algorithms for SMS Spam Detection",
        "dataset_source": "UCI Machine Learning Repository (SMS Spam Collection ID: 228)",
        "random_state": random_state,
        "test_partition_size": test_size,
        "cv_method": "StratifiedKFold",
        "cv_folds": 5,
        "cv_scoring_metric": "f1 (positive class = spam)",
        "model_selection_protocol": "Peak Mean 5-Fold Cross-Validation F1-Score on Training Partition",
        "feature_extraction_methods": ["TF-IDF (Sublinear)", "CountVectorizer (Bag-of-Words)"],
        "classifiers_evaluated": [
            "Multinomial Naive Bayes",
            "Bernoulli Naive Bayes (Binarized Presence)",
            "Logistic Regression",
            "Linear Support Vector Machine (Platt Calibrated)"
        ],
        "preprocessing_rules": {
            "lowercasing": True,
            "url_tokenization": "urltoken",
            "email_tokenization": "emailtoken",
            "currency_tokenization": "currencytoken",
            "punctuation_removal": True,
            "negation_words_preserved": [
                "no", "not", "nor", "neither", "never", "without", "against",
                "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent"
            ],
            "stemmer": "PorterStemmer"
        }
    }
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Experiment configuration saved to: {config_path}")


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, report = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    trained_models, cv_summary, metadata = train_tune_and_select(X_train, y_train)
    df_eval, champ, best_test = run_comprehensive_evaluation(trained_models, X_test, y_test)
    save_experiment_config()

    print("\n=================== FINAL TEST SET EVALUATION TABLE ===================")
    table_view = df_eval.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        table_view[col] = (table_view[col] * 100).round(2).astype(str) + "%"
    print(table_view[["Experiment", "Is_Validation_Champion", "Accuracy", "Precision", "Recall", "F1-Score", "FP (False Alarm)", "FN (Missed Spam)"]].to_string(index=False))
    print("=======================================================================")
