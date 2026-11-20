"""
Model Evaluation and Visualization Module for SMS Spam Detection.

This module handles:
1. Evaluating all four trained ML models on the unseen test set (1,032 samples).
2. Calculating metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
3. Generating and saving publication-quality comparison visualizations:
   - reports/figures/confusion_matrices.png
   - reports/figures/model_comparison_barchart.png
4. Exporting the best performing model pipeline to models/best_model.pkl.
"""

import os
import joblib
import pandas as pd
import numpy as np
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
from model_training import train_baseline_models, tune_models, save_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a single model pipeline on the test set.
    Metrics focus on the minority positive class (Spam = 1).
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "Confusion_Matrix": cm,
        "y_pred": y_pred
    }


def evaluate_all_models(models_dict, X_test, y_test):
    """
    Evaluates multiple model pipelines and compiles results into a DataFrame.
    """
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(model, X_test, y_test)
    return results


def plot_confusion_matrices(results, output_path="reports/figures/confusion_matrices.png"):
    """
    Plots a 2x2 grid of confusion matrices with annotations for Ham vs Spam.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    model_names = list(results.keys())

    for idx, name in enumerate(model_names):
        cm = results[name]["Confusion_Matrix"]
        ax = axes[idx]

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Annotations text
        labels = [
            [f"True Ham\n{cm[0,0]}\n({cm_percent[0,0]:.1f}%)", f"False Spam (FP)\n{cm[0,1]}\n({cm_percent[0,1]:.1f}%)"],
            [f"Missed Spam (FN)\n{cm[1,0]}\n({cm_percent[1,0]:.1f}%)", f"True Spam (TP)\n{cm[1,1]}\n({cm_percent[1,1]:.1f}%)"]
        ]

        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=False,
            ax=ax, annot_kws={"fontsize": 11, "fontweight": "semibold"}
        )
        ax.set_title(f"{name}\n(Acc: {results[name]['Accuracy']*100:.2f}% | F1: {results[name]['F1-Score']*100:.2f}%)",
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Label", fontweight='semibold')
        ax.set_ylabel("True Label", fontweight='semibold')
        ax.set_xticklabels(["Ham", "Spam"])
        ax.set_yticklabels(["Ham", "Spam"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrices plot saved to: {output_path}")


def plot_model_comparison(df_metrics, output_path="reports/figures/model_comparison_barchart.png"):
    """
    Plots a comparative grouped bar chart for Accuracy, Precision, Recall, and F1-score.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Melt DataFrame for Seaborn grouped bar plot
    df_plot = df_metrics.reset_index().rename(columns={"index": "Algorithm"})
    df_melted = pd.melt(
        df_plot,
        id_vars=["Algorithm"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )
    df_melted["Score (%)"] = df_melted["Score"] * 100

    plt.figure(figsize=(13, 6))
    palette = ["#2b5c8f", "#3c9d9b", "#f4a261", "#e76f51"]
    ax = sns.barplot(
        data=df_melted,
        x="Algorithm",
        y="Score (%)",
        hue="Metric",
        palette=palette
    )

    plt.title("Comparative Performance of ML Algorithms for SMS Spam Detection", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Machine Learning Algorithm", fontweight='semibold', labelpad=10)
    plt.ylabel("Score (%)", fontweight='semibold')
    plt.ylim(75, 103)
    plt.legend(loc="lower right", frameon=True)

    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height + 0.5),
                        ha='center', va='bottom', fontsize=8.5, rotation=0, fontweight='semibold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Model comparison bar chart saved to: {output_path}")


def run_full_evaluation():
    """
    Executes the complete evaluation workflow:
    1. Loads cleaned data and train-test splits.
    2. Trains tuned models.
    3. Evaluates on test data.
    4. Generates visual charts.
    5. Saves the best model pipeline to models/best_model.pkl.
    """
    raw_df = load_raw_data()
    clean_df, _ = clean_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)

    # Train tuned models
    tuned_models, best_params = tune_models(X_train, y_train)

    # Evaluate on test set
    results = evaluate_all_models(tuned_models, X_test, y_test)

    # Summary table
    table_data = []
    for name, m in results.items():
        table_data.append({
            "Algorithm": name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1-Score": m["F1-Score"]
        })
    df_summary = pd.DataFrame(table_data).set_index("Algorithm")

    print("\n=================== TEST SET EVALUATION RESULTS ===================")
    print((df_summary * 100).round(2).to_string())
    print("===================================================================\n")

    # Save visual charts
    plot_confusion_matrices(results)
    plot_model_comparison(df_summary)

    # Identify best model based on F1-Score & Accuracy
    best_model_name = df_summary["F1-Score"].idxmax()
    best_pipeline = tuned_models[best_model_name]
    print(f"[TOP PERFORMER] Best Algorithm: {best_model_name} with F1-Score: {df_summary.loc[best_model_name, 'F1-Score']*100:.2f}%")

    # Save to disk
    save_model(best_pipeline, "models/best_model.pkl")

    return df_summary, results, best_model_name


if __name__ == "__main__":
    run_full_evaluation()
