"""
Evaluation and Visualization Module for SMS Spam Detection.

Evaluates all eight experimental combinations on the untouched test set (1,032 samples).
Generates:
1. reports/results/results_summary.csv & results_summary.json
2. Visual comparison charts:
   - reports/figures/accuracy_comparison.png
   - reports/figures/precision_comparison.png
   - reports/figures/recall_comparison.png
   - reports/figures/f1_comparison.png
   - reports/figures/feature_extraction_comparison.png (TF-IDF vs. CountVectorizer)
   - reports/figures/confusion_matrices_all8.png (8-panel annotated heatmap grid)
   - reports/figures/best_model_comparison.png
3. Identifies and saves the overall best performing pipeline to models/best_model.pkl.
"""

import os
import json
import joblib
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

from data_loader import load_raw_data, clean_data
from feature_extraction import split_data
from model_training import train_and_tune_all_8


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


def run_comprehensive_evaluation(models_dict, X_test, y_test, results_dir="reports/results", fig_dir="reports/figures"):
    """
    Evaluates all 8 pipelines, compiles tabular summaries, exports json/csv,
    and generates visual charts.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    records = []
    cm_dict = {}

    for exp_name, model in models_dict.items():
        metrics = evaluate_pipeline(model, X_test, y_test)
        tech = "TF-IDF" if "TF-IDF" in exp_name else "CountVectorizer"
        algo = exp_name.replace("TF-IDF + ", "").replace("CountVectorizer + ", "")

        rec = {
            "Experiment": exp_name,
            "Feature_Extraction": tech,
            "Algorithm": algo,
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

    # Export machine-readable results
    csv_path = os.path.join(results_dir, "results_summary.csv")
    json_path = os.path.join(results_dir, "results_summary.json")
    df_results.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Results exported to {csv_path} and {json_path}")

    # Set aesthetic plotting theme
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # --- 1. Metric-Specific Bar Charts (Accuracy, Precision, Recall, F1) ---
    metrics_to_plot = [
        ("Accuracy", "accuracy_comparison.png", "Test Accuracy across 8 Experiments", (94, 99)),
        ("Precision", "precision_comparison.png", "Spam Precision across 8 Experiments (Minimizing False Alarms)", (90, 100)),
        ("Recall", "recall_comparison.png", "Spam Recall across 8 Experiments (Sensitivity to Spam)", (80, 95)),
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

    # --- 2. Grouped Feature Extraction Comparison Chart ---
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
    plt.title("Aggregated Performance: TF-IDF vs. CountVectorizer (Averaged across Classifiers)",
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

    # --- 3. 8-Panel Confusion Matrix Grid ---
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
        ax.set_title(f"{exp_name}\n(Acc: {row_data['Accuracy']*100:.2f}% | F1: {row_data['F1-Score']*100:.2f}%)",
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Predicted Label", fontweight='semibold')
        ax.set_ylabel("True Label", fontweight='semibold')
        ax.set_xticklabels(["Ham", "Spam"])
        ax.set_yticklabels(["Ham", "Spam"])

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices_all8.png"), dpi=300)
    plt.close()

    # --- 4. Best Model Selection & Persistence ---
    best_idx = df_results["F1-Score"].idxmax()
    best_row = df_results.iloc[best_idx]
    best_exp_name = best_row["Experiment"]
    best_pipeline = models_dict[best_exp_name]

    best_model_path = "models/best_model.pkl"
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[BEST MODEL IDENTIFIED] {best_exp_name}")
    print(f"       Accuracy: {best_row['Accuracy']*100:.2f}% | F1-Score: {best_row['F1-Score']*100:.2f}% | Precision: {best_row['Precision']*100:.2f}% | Recall: {best_row['Recall']*100:.2f}%")
    print(f"[INFO] Best model serialized to {best_model_path}")

    return df_results, best_exp_name


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    trained_models, train_summary = train_and_tune_all_8(X_train, y_train)
    df_eval_results, best_model_name = run_comprehensive_evaluation(trained_models, X_test, y_test)

    print("\n=================== FINAL TEST SET EVALUATION TABLE ===================")
    table_view = df_eval_results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        table_view[col] = (table_view[col] * 100).round(2).astype(str) + "%"
    print(table_view[["Experiment", "Accuracy", "Precision", "Recall", "F1-Score", "FP (False Alarm)", "FN (Missed Spam)"]].to_string(index=False))
    print("=======================================================================")
