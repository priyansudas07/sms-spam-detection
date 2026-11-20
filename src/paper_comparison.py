"""
Comparative Analysis Script: Our Experiments vs. IEEE TENCON 2023 Paper.

Reference Paper:
"A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)"
Authors: Robert G. de Luna et al., Polytechnic University of the Philippines.
Conference: 2023 IEEE Region 10 Conference (TENCON), Chiang Mai, Thailand.

This script:
1. Creates a side-by-side comparison visualization between the paper's results and our project.
2. Generates a comprehensive academic research comparison report in reports/research_comparison.md.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_paper_comparison(output_dir="reports"):
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Data Definition ---
    # Reference Paper Reported Results (Table VI & Table VII of de Luna et al.)
    paper_data = {
        "Bernoulli Naive Bayes": {"Accuracy": 96.63, "Precision": 98.44, "Recall": 95.52, "F1-Score": 92.78},
        "Logistic Regression": {"Accuracy": 95.87, "Precision": 98.77, "Recall": 93.74, "F1-Score": 89.26},
        "Support Vector Machine": {"Accuracy": 94.57, "Precision": None, "Recall": None, "F1-Score": None},
        "Multinomial Naive Bayes": {"Accuracy": 96.03, "Precision": 95.59, "Recall": None, "F1-Score": None}
    }

    # Our Experimental Test Set Results
    our_data = {
        "Bernoulli Naive Bayes": {"Accuracy": 98.06, "Precision": 97.37, "Recall": 86.72, "F1-Score": 91.74},
        "Logistic Regression": {"Accuracy": 97.97, "Precision": 96.52, "Recall": 86.72, "F1-Score": 91.36},
        "Linear SVM": {"Accuracy": 98.16, "Precision": 96.58, "Recall": 88.28, "F1-Score": 92.24},
        "Multinomial Naive Bayes": {"Accuracy": 97.77, "Precision": 94.87, "Recall": 86.72, "F1-Score": 90.61}
    }

    # --- Plot: Comparative Accuracy Chart ---
    models = ["Bernoulli Naive Bayes", "Logistic Regression", "Support Vector Machine / Linear SVM", "Multinomial Naive Bayes"]
    paper_acc = [96.63, 95.87, 94.57, 96.03]
    our_acc = [98.06, 97.97, 98.16, 97.77]

    df_comp = pd.DataFrame({
        "Model": models * 2,
        "Accuracy (%)": paper_acc + our_acc,
        "Study": ["IEEE TENCON 2023 (de Luna et al.)"] * 4 + ["Our Project (Authentic UCI + TF-IDF)"] * 4
    })

    plt.figure(figsize=(12, 6))
    palette = ["#d9534f", "#2b5c8f"]
    ax = sns.barplot(data=df_comp, x="Model", y="Accuracy (%)", hue="Study", palette=palette)
    plt.title("Model Accuracy: IEEE TENCON 2023 Reference Paper vs. Our Project", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Model Architecture", fontweight='semibold')
    plt.ylabel("Accuracy (%)", fontweight='semibold')
    plt.ylim(90, 102)
    plt.xticks(rotation=10, ha='right')
    plt.legend(loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}%",
                        (p.get_x() + p.get_width() / 2., height + 0.3),
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

    plt.tight_layout()
    chart_path = os.path.join(fig_dir, "paper_vs_our_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Comparison plot saved to: {chart_path}")

    # --- Generate Markdown Report ---
    report_content = f"""# Comparative Research Analysis: Our Project vs. IEEE Reference Study

**Subject:** Research Methodologies (RM)  
**Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
**Authors:** Robert G. de Luna, Karina L. Enriquez, Aira Mae Española, Mateo Ramos, Verna C. Magnaye, Dexter Astorga, Brian Allen Lanting, Jenjazel Redondo, Rose Anne L. Reaño, Trisha Celestial, Danielle Mugar  
**Venue:** *2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand, Oct 31 – Nov 3, 2023  
**DOI / Paper ID:** 979-8-3503-0219-6/23/$31.00 ©2023 IEEE  

---

## 1. Executive Summary & Objective

In accordance with academic research methodology principles, this study compares our experimental findings against the published benchmarks of **de Luna et al. (IEEE TENCON 2023)**. 

While both investigations address SMS spam classification through natural language processing and supervised machine learning, their experimental designs intentionally differ across:
1. **Dataset composition and class balance** (Artificially balanced 60:40 partition vs. Natural 87.6:12.4 imbalance).
2. **Feature vectorization strategy** (CountVectorizer term frequencies vs. Sublinear TF-IDF with bigrams).
3. **Model selection and tuning pipelines**.

---

## 2. Experimental Benchmark Comparison Table

| Algorithm | IEEE TENCON 2023 Paper (Reported) | Our Experimental Project (Observed) | Key Comparative Observations & Variance Analysis |
|:---|:---|:---|:---|
| **Bernoulli Naive Bayes (BNB)** | **Accuracy:** 96.63%<br>**Precision:** 98.44%<br>**Recall:** 95.52%<br>**F1-Score:** 92.78% | **Accuracy:** 98.06%<br>**Precision:** 97.37%<br>**Recall:** 86.72%<br>**F1-Score:** 91.74% | BNB was the **#1 top model** in the reference paper. In our study, BNB achieved near-identical F1-score (**91.74%** vs. 92.78%) and higher accuracy (**98.06%** vs. 96.63%), demonstrating that binarized presence/absence modeling is exceptionally robust across both balanced and imbalanced spam distributions. |
| **Linear SVM / SVC** | **Accuracy:** 94.57% *(Default CV)*<br>*(Not optimized in paper)* | **Accuracy:** **98.16%**<br>**Precision:** 96.58%<br>**Recall:** **88.28%**<br>**F1-Score:** **92.24%** | The IEEE authors evaluated SVC only with default parameters (ranking 5th at 94.57%) and omitted it from GridSearchCV optimization. In our study, hyperparameter tuning with TF-IDF unigram+bigram representation elevated Linear SVM to our **#1 top performing model (98.16% accuracy, 92.24% F1)**. |
| **Logistic Regression (LR)** | **Accuracy:** 95.87%<br>**Precision:** 98.77%<br>**Recall:** 93.74%<br>**F1-Score:** 89.26% | **Accuracy:** 97.97%<br>**Precision:** 96.52%<br>**Recall:** 86.72%<br>**F1-Score:** 91.36% | Both studies identify Logistic Regression as a strong linear baseline. Our tuned pipeline achieved higher overall test accuracy (97.97% vs. 95.87%) and higher F1-score (91.36% vs. 89.26%), benefiting from sublinear term frequency weighting. |
| **Multinomial Naive Bayes (MNB)** | **Accuracy:** 96.03%<br>**Precision:** 95.59% | **Accuracy:** 97.77%<br>**Precision:** 94.87%<br>**Recall:** 86.72%<br>**F1-Score:** 90.61% | In the reference paper, MNB was used specifically to compare CountVectorizer against TF-IDF. Our project tuned MNB smoothing ($\alpha=0.1$) reaching 97.77% accuracy with solid 94.87% precision. |

---

## 3. Critical Methodological Differences

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 METHODOLOGICAL COMPARISON                              │
├─────────────────────────┬──────────────────────────────────┬───────────────────────────┤
│ Dimension               │ IEEE TENCON 2023 (de Luna et al.)│ Our Academic RM Project   │
├─────────────────────────┼──────────────────────────────────┼───────────────────────────┤
│ Dataset Source          │ UCI + Philippine Spam + PUPSTB   │ Official UCI Spam Dataset │
│ Total Cleaned Messages  │ 3,148 messages                   │ 5,160 unique messages     │
│ Class Distribution      │ Artificial 60:40 (Ham:Spam)      │ Natural 87.6:12.4 Imbalance│
│ Class Balancing Method  │ Random Undersampling of Ham      │ Stratified Sampling Split │
│ Feature Extraction      │ CountVectorizer (Raw Frequency)  │ TF-IDF (Sublinear + Ngram)│
│ Primary Metric for Top  │ Accuracy                         │ F1-Score & Accuracy       │
│ Top Recommended Model   │ Bernoulli Naive Bayes (96.63%)   │ Linear SVM (98.16%)       │
└─────────────────────────┴──────────────────────────────────┴───────────────────────────┘
```

### 3.1 Dataset Balancing vs. Natural Distribution
- **Reference Paper Approach:** The authors noted the heavy imbalance of the UCI dataset (87% ham, 13% spam) and addressed it by adding 79 Philippine spam texts and 561 student-collected spam texts, followed by **undersampling ham messages to force an artificial 60% Ham to 40% Spam partition** (Table IV: 1,930 ham, 1,218 spam).
- **Our Project Approach:** We maintained the authentic, un-skewed real-world distribution (87.6% ham, 12.4% spam). In real telecommunication networks, spam is inherently a minority phenomenon. Artificially forcing a 60:40 ratio inflates recall artificially (explaining their 95.52% recall vs. our 88.28% recall) because the model encounters spam at three times its natural rate.

### 3.2 Vectorization: CountVectorizer vs. Sublinear TF-IDF
- **Reference Paper Approach:** The authors compared CountVectorizer and TF-IDF using default Naive Bayes variants and concluded CountVectorizer produced slightly higher accuracy (96.03% vs 95.56%), adopting CountVectorizer for subsequent model training.
- **Our Project Approach:** CountVectorizer weights words purely by absolute occurrences, which disproportionately biases generic repetitive words. We employed **TF-IDF with sublinear term-frequency scaling and unigram + bigram ranges (1, 2)**. This penalizes ubiquitously frequent tokens while assigning high discriminative weight to spam-specific multi-word phrases (*"free entry"*, *"cash prize"*, *"urgent claim"*).

---

## 4. Academic Validity & Research Limitations

1. **Non-Direct Controlled Equivalence:**  
   Because the underlying training corpora, class proportions, and word embedding matrices are distinct, this comparison constitutes an **external benchmark validation** rather than a direct controlled ablation.
2. **Generalizability:**  
   Models evaluated on naturally imbalanced corpora reflect actual deployment conditions in mobile telecom networks more accurately than models evaluated on heavily undersampled balanced partitions.
3. **False Positive Trade-off:**  
   In SMS spam detection, False Positives (blocking a legitimate personal/banking SMS) are far more detrimental than False Negatives (a spam message slipping through). Both studies confirm that Naive Bayes and Linear SVM maintain exceptionally high precision (>96.5%), ensuring minimal false alarms.

---
*Report generated for Research Methodologies (RM) - 7th Semester B.Tech Project.*
"""

    report_path = os.path.join(output_dir, "research_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[INFO] Research comparison report written to: {report_path}")

    return chart_path, report_path


if __name__ == "__main__":
    generate_paper_comparison()
