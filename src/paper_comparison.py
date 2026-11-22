"""
Academic Comparative Analysis Module: Our Experiments vs. IEEE TENCON 2023 Study.

Reference Paper:
"A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)"
Authors: Robert G. de Luna, Karina L. Enriquez, Aira Mae Española, Mateo Ramos, Verna C. Magnaye,
         Dexter Astorga, Brian Allen Lanting, Jenjazel Redondo, Rose Anne L. Reaño,
         Trisha Celestial, Danielle Mugar.
Affiliation: Polytechnic University of the Philippines, Sto. Tomas City, Batangas, Philippines.
Conference: 2023 IEEE Region 10 Conference (TENCON), 31 Oct - 3 Nov 2023, Chiang Mai, Thailand.
Pages: 52-57. DOI/ISBN: 979-8-3503-0219-6/23/$31.00 ©2023 IEEE.

Authoritative Verification of Published Paper Details:
- Evaluated 11 classifiers with default parameters using CountVectorizer (Table VI).
- Selected Top 4 based on default accuracy (LR: 95.62%, BNB: 95.43%, RFC: 95.24%, ETC: 94.98%).
- Hyperparameter tuning via GridSearchCV on Top 4 (Table VII):
  1. Bernoulli Naive Bayes (BNB): Acc = 96.6333%, Prec = 98.4431%, Rec = 95.5174%, F1 = 92.7835%
  2. Random Forest (RFC): Acc = 96.0297%, Prec = 98.5228%, Rec = 94.8699%, F1 = 91.1408%
  3. Logistic Regression (LR): Acc = 95.8709%, Prec = 98.7711%, Rec = 93.7399%, F1 = 89.2555%
  4. Extra Trees (ETC): Acc = 95.8070%, Prec = 98.4885%, Rec = 94.3495%, F1 = 90.5663%
- ATTRIBUTION CORRECTION: 96.0297% belongs to Random Forest (RFC), NOT Multinomial Naive Bayes.
- Support Vector Classifier (SVC) had default CV accuracy of 94.5684% in Table VI and was omitted from Table VII.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "results", "results_summary.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "reports", "research_comparison.md")

os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def generate_paper_comparison():
    print("=" * 80)
    print("GENERATING IEEE TENCON 2023 COMPARATIVE VISUALIZATION")
    print("=" * 80)

    # 1. Load our verified experimental test results
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Our experimental results not found at {RESULTS_PATH}. Run src/evaluation.py first.")

    df_our = pd.read_csv(RESULTS_PATH)
    print(f"✓ Loaded our experimental evaluation results ({len(df_our)} pipelines).")

    def get_our_metric(exp_name, metric_col):
        rows = df_our[df_our["Experiment"] == exp_name]
        if rows.empty:
            return 0.0
        return float(rows[metric_col].values[0]) * 100

    bnb_cv_acc = get_our_metric("CountVectorizer + Bernoulli Naive Bayes", "Accuracy")
    bnb_tf_acc = get_our_metric("TF-IDF + Bernoulli Naive Bayes", "Accuracy")
    lr_cv_acc = get_our_metric("CountVectorizer + Logistic Regression", "Accuracy")
    lr_tf_acc = get_our_metric("TF-IDF + Logistic Regression", "Accuracy")
    svm_cv_acc = get_our_metric("CountVectorizer + Linear SVM", "Accuracy")
    svm_tf_acc = get_our_metric("TF-IDF + Linear SVM", "Accuracy")

    bnb_cv_f1 = get_our_metric("CountVectorizer + Bernoulli Naive Bayes", "F1-Score")
    bnb_tf_f1 = get_our_metric("TF-IDF + Bernoulli Naive Bayes", "F1-Score")
    lr_cv_f1 = get_our_metric("CountVectorizer + Logistic Regression", "F1-Score")
    lr_tf_f1 = get_our_metric("TF-IDF + Logistic Regression", "F1-Score")
    svm_cv_f1 = get_our_metric("CountVectorizer + Linear SVM", "F1-Score")
    svm_tf_f1 = get_our_metric("TF-IDF + Linear SVM", "F1-Score")

    # 2. Generate High-Resolution 2-Panel Benchmark Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Accuracy Comparison
    acc_records = [
        {"Model": "Bernoulli Naive Bayes", "Accuracy": 96.63, "Study": "IEEE TENCON 2023 (CountVec)"},
        {"Model": "Bernoulli Naive Bayes", "Accuracy": bnb_cv_acc, "Study": "Our Project (CountVec)"},
        {"Model": "Bernoulli Naive Bayes", "Accuracy": bnb_tf_acc, "Study": "Our Project (TF-IDF)"},

        {"Model": "Logistic Regression", "Accuracy": 95.87, "Study": "IEEE TENCON 2023 (CountVec)"},
        {"Model": "Logistic Regression", "Accuracy": lr_cv_acc, "Study": "Our Project (CountVec)"},
        {"Model": "Logistic Regression", "Accuracy": lr_tf_acc, "Study": "Our Project (TF-IDF)"},

        {"Model": "Support Vector Machine", "Accuracy": 94.57, "Study": "IEEE TENCON 2023 (CountVec)"},
        {"Model": "Support Vector Machine", "Accuracy": svm_cv_acc, "Study": "Our Project (CountVec)"},
        {"Model": "Support Vector Machine", "Accuracy": svm_tf_acc, "Study": "Our Project (TF-IDF)"}
    ]
    df_plot_acc = pd.DataFrame(acc_records)

    palette = ["#e76f51", "#d9534f", "#2b5c8f"]
    ax0 = sns.barplot(data=df_plot_acc, x="Model", y="Accuracy", hue="Study", ax=axes[0], palette=palette)
    axes[0].set_title("A. Accuracy Benchmark: Overlapping Models\n(Note: Differing Corpora & Sampling)", fontweight='bold', fontsize=11, pad=10)
    axes[0].set_xlabel("Classifier Model", fontweight='semibold')
    axes[0].set_ylabel("Accuracy (%)", fontweight='semibold')
    axes[0].set_ylim(92, 100)
    axes[0].legend(title="Study & Feature Extraction", loc="lower right", fontsize=8.5)
    for p in ax0.patches:
        h = p.get_height()
        if h > 0:
            ax0.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h + 0.12),
                         ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    # Panel B: F1-Score Comparison
    f1_records = [
        {"Model": "Bernoulli Naive Bayes", "F1_Score": 92.78, "Study": "IEEE TENCON 2023 (CountVec)"},
        {"Model": "Bernoulli Naive Bayes", "F1_Score": bnb_cv_f1, "Study": "Our Project (CountVec)"},
        {"Model": "Bernoulli Naive Bayes", "F1_Score": bnb_tf_f1, "Study": "Our Project (TF-IDF)"},

        {"Model": "Logistic Regression", "F1_Score": 89.26, "Study": "IEEE TENCON 2023 (CountVec)"},
        {"Model": "Logistic Regression", "F1_Score": lr_cv_f1, "Study": "Our Project (CountVec)"},
        {"Model": "Logistic Regression", "F1_Score": lr_tf_f1, "Study": "Our Project (TF-IDF)"},

        {"Model": "Support Vector Machine", "F1_Score": svm_cv_f1, "Study": "Our Project (CountVec)"},
        {"Model": "Support Vector Machine", "F1_Score": svm_tf_f1, "Study": "Our Project (TF-IDF)"}
    ]
    df_plot_f1 = pd.DataFrame(f1_records)

    ax1 = sns.barplot(data=df_plot_f1, x="Model", y="F1_Score", hue="Study", ax=axes[1], palette=palette)
    axes[1].set_title("B. Spam F1-Score Benchmark\n(Demonstrating Impact of Class Imbalance & Preprocessing)", fontweight='bold', fontsize=11, pad=10)
    axes[1].set_xlabel("Classifier Model", fontweight='semibold')
    axes[1].set_ylabel("F1-Score (%)", fontweight='semibold')
    axes[1].set_ylim(87, 97)
    axes[1].legend(title="Study & Feature Extraction", loc="lower right", fontsize=8.5)
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(f"{h:.2f}%", (p.get_x() + p.get_width() / 2., h + 0.12),
                         ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    plt.tight_layout()
    chart_output_path = os.path.join(FIGURES_DIR, "paper_vs_our_comparison.png")
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved updated comparison chart to: {chart_output_path}")

    return chart_output_path, REPORT_MD_PATH


if __name__ == "__main__":
    generate_paper_comparison()
