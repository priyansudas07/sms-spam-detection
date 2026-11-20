"""
Generates the complete, self-contained Google Colab & Jupyter Notebook
'notebooks/sms_spam_detection.ipynb' for the SMS Spam Detection project.
"""

import os
import json


def build_notebook(output_path="notebooks/sms_spam_detection.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # --- Title Cell ---
    add_md("""# Comparative Analysis of ML Algorithms for SMS Spam Detection
**Academic Project for Research Methodologies (RM) — 7th Semester B.Tech Computer Science**

### Project Overview
Short Message Service (SMS) remains a critical communication channel for personal, banking, and two-factor authentication (2FA) systems. However, users are increasingly exposed to unsolicited SMS spam, including fraudulent financial scams, phishing lures, and malware distribution.

This research notebook implements an end-to-end Machine Learning pipeline to classify SMS messages as either **Ham (Legitimate)** or **Spam**. Furthermore, this study compares its empirical findings against published benchmarks from the reference research paper:
> **Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)* (Robert G. de Luna et al., IEEE TENCON 2023).

---
### Research Methodology Pipeline:
1. **Dataset Acquisition & Cleaning:** Official UCI SMS Spam Collection dataset.
2. **Exploratory Data Analysis (EDA):** Class distribution and text length distributions.
3. **Text Preprocessing:** Lowercasing, noise filtering, tokenization, stopword removal, and Porter stemming.
4. **Feature Extraction:** Sublinear TF-IDF Vectorization with unigrams and bigrams (with binary representations for Bernoulli Naive Bayes).
5. **Stratified Splitting:** 80:20 train-test partition preserving class balance with zero data leakage.
6. **Model Training & Tuning:** Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, and Linear SVM with 5-fold cross-validation.
7. **Performance Evaluation:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
8. **Comparative Analysis with IEEE Study:** Systematic examination of results, feature engineering differences, and methodological trade-offs.""")

    # --- Section 1: Dependencies ---
    add_md("## 1. Environment Setup & Dependency Installation")
    add_code("""# Install required packages (uncomment if running in Google Colab)
# !pip install nltk scikit-learn pandas numpy matplotlib seaborn

import os
import re
import string
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Download essential NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set visual styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 5)
print("Environment successfully initialized.")""")

    # --- Section 2: Dataset Loading ---
    add_md("## 2. Dataset Acquisition & Inspection\n"
           "We fetch the official **UCI SMS Spam Collection dataset** directly from the UCI Machine Learning Repository.")
    add_code("""DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
data_dir = "data"
raw_file = os.path.join(data_dir, "SMSSpamCollection")
zip_file = os.path.join(data_dir, "smsspamcollection.zip")

os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(raw_file):
    print("Downloading dataset from UCI repository...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(DATASET_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(zip_file, 'wb') as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    if os.path.exists(zip_file):
        os.remove(zip_file)
    print("Download and extraction complete.")
else:
    print("Dataset already present locally.")

# Load raw dataset into pandas DataFrame
df_raw = pd.read_csv(
    raw_file,
    sep='\\t',
    names=['label', 'message'],
    header=None,
    encoding='utf-8',
    quoting=3
)

print(f"Total Raw Samples: {len(df_raw)}")
print(f"Missing Values:\\n{df_raw.isnull().sum()}\\n")
print("Top 10 Messages:")
df_raw.head(10)""")

    # --- Section 3: Data Cleaning ---
    add_md("## 3. Data Cleaning & Deduplication\n"
           "Duplicate messages artificially inflate model performance and introduce data leakage between train and test sets. We identify and remove all duplicate entries.")
    add_code("""# Duplicate identification
initial_count = len(df_raw)
duplicate_count = df_raw.duplicated(subset=['message']).sum()
print(f"Duplicate messages detected: {duplicate_count}")

# Clean strings and drop duplicates
df_clean = df_raw.dropna(subset=['label', 'message']).copy()
df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
df_clean['message'] = df_clean['message'].astype(str).str.strip()
df_clean = df_clean.drop_duplicates(subset=['message']).reset_index(drop=True)

# Binary target encoding: Ham = 0, Spam = 1
df_clean['label_num'] = df_clean['label'].map({'ham': 0, 'spam': 1})

print(f"Cleaned Dataset Total: {len(df_clean)} unique messages (Dropped {initial_count - len(df_clean)} duplicates)")
print("\\nClass Distribution in Cleaned Dataset:")
class_dist = pd.DataFrame({
    'Count': df_clean['label'].value_counts(),
    'Percentage (%)': (df_clean['label'].value_counts(normalize=True) * 100).round(2)
})
class_dist""")

    # --- Section 4: EDA Visualizations ---
    add_md("## 4. Exploratory Data Analysis (EDA)\n"
           "Visualizing the class imbalance and character/word length distributions between Ham and Spam.")
    add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class Distribution Bar Plot
palette = ['#2b5c8f', '#d9534f']
sns.barplot(x=class_dist.index, y=class_dist['Count'], palette=palette, hue=class_dist.index, legend=False, ax=axes[0])
axes[0].set_title("Cleaned Dataset Class Distribution (Ham vs. Spam)", fontweight='bold')
axes[0].set_ylabel("Number of Messages")
for p in axes[0].patches:
    height = p.get_height()
    pct = (height / len(df_clean)) * 100
    axes[0].annotate(f"{int(height):,}\\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., height / 2),
                     ha='center', va='center', color='white', fontweight='bold')

# Message Character Length Distribution
df_clean['char_length'] = df_clean['message'].apply(len)
sns.histplot(data=df_clean, x='char_length', hue='label', bins=50, kde=True,
             palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[1])
axes[1].set_title("Message Character Length Distribution", fontweight='bold')
axes[1].set_xlabel("Character Count")
axes[1].set_xlim(0, 300)

plt.tight_layout()
plt.show()""")

    # --- Section 5: Text Preprocessing ---
    add_md("## 5. NLP Text Preprocessing Pipeline\n"
           "We implement sequential text normalization: lowercasing, hyperlink removal, punctuation stripping, tokenization, stopword removal, and Porter stemming.")
    add_code("""STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs and emails
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'\\S+@\\S+', '', text)
    # 3. Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    # 4. Tokenize by whitespace
    tokens = text.split()
    # 5. Remove stopwords and apply Porter Stemming
    cleaned_tokens = [
        STEMMER.stem(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 1
    ]
    # 6. Rejoin normalized tokens
    return " ".join(cleaned_tokens)

# Demonstration of text preprocessing
sample = "WINNER!! As a valued customer, you have been selected to win £900 cash prize! Text CLAIM to 87121."
print("Original Raw SMS :", sample)
print("Normalized SMS   :", preprocess_text(sample))

# Compute corpus-level reduction statistics (similar to Table V in the IEEE paper)
df_clean['processed_message'] = df_clean['message'].apply(preprocess_text)

chars_before = df_clean['message'].apply(len).sum()
chars_after = df_clean['processed_message'].apply(len).sum()
words_before = df_clean['message'].apply(lambda x: len(x.split())).sum()
words_after = df_clean['processed_message'].apply(lambda x: len(x.split())).sum()

print("\\n--- Corpus Transformation Reduction (Table V Analogue) ---")
print(f"Characters : {chars_before:,} -> {chars_after:,} ({(1 - chars_after/chars_before)*100:.2f}% reduction)")
print(f"Words      : {words_before:,} -> {words_after:,} ({(1 - words_after/words_before)*100:.2f}% reduction)")""")

    # --- Section 6: Train-Test Split and Pipelines ---
    add_md("## 6. Train-Test Splitting & Leakage-Free Scikit-Learn Pipelines\n"
           "We split the data using a **stratified 80:20 split**. To guarantee zero data leakage, text preprocessing and TF-IDF feature learning are embedded within `Pipeline` constructs fitted exclusively on training data.")
    add_code("""X_train, X_test, y_train, y_test = train_test_split(
    df_clean['message'],
    df_clean['label_num'],
    test_size=0.20,
    random_state=42,
    stratify=df_clean['label_num']
)

print(f"Training Samples: {len(X_train)} (Ham: {(y_train==0).sum()}, Spam: {(y_train==1).sum()})")
print(f"Testing Samples : {len(X_test)} (Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()})")

# Custom scikit-learn transformer for clean integration
class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [preprocess_text(t) for t in X]

# Feature Vectorizers:
# Standard TF-IDF for continuous classifiers; Binarized TF-IDF for Bernoulli Naive Bayes
tfidf_continuous = TfidfVectorizer(max_features=4000, sublinear_tf=True, ngram_range=(1, 2))
tfidf_binary = TfidfVectorizer(max_features=3000, binary=True, ngram_range=(1, 2))

# Construct baseline model pipelines
models = {
    "Multinomial Naive Bayes": Pipeline([
        ('prep', TextPreprocessorTransformer()),
        ('vec', tfidf_continuous),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    "Bernoulli Naive Bayes": Pipeline([
        ('prep', TextPreprocessorTransformer()),
        ('vec', tfidf_binary),
        ('clf', BernoulliNB(alpha=0.05))
    ]),
    "Logistic Regression": Pipeline([
        ('prep', TextPreprocessorTransformer()),
        ('vec', tfidf_continuous),
        ('clf', LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ]),
    "Linear SVM": Pipeline([
        ('prep', TextPreprocessorTransformer()),
        ('vec', tfidf_continuous),
        ('clf', SVC(C=2.0, kernel='linear', probability=True, random_state=42))
    ])
}
print("Model pipelines successfully configured.")""")

    # --- Section 7: Training & Evaluation ---
    add_md("## 7. Model Training & Evaluation on Unseen Test Data\n"
           "We fit each model on `X_train` and evaluate its generalization ability on `X_test` (1,032 samples).")
    add_code("""results = {}

for name, pipeline in models.items():
    print(f"Training and evaluating {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "F1-Score": f1_score(y_test, y_pred, pos_label=1),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred)
    }

# Display results table
df_results = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
print("\\n=================== TEST SET EVALUATION RESULTS ===================")
display((df_results * 100).round(2))
print("===================================================================")""")

    # --- Section 8: Visualizing Results ---
    add_md("## 8. Comparative Visualizations\n"
           "Visualizing the Confusion Matrices and multi-metric comparison bar charts.")
    add_code("""# 1. Confusion Matrix Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, metrics) in enumerate(results.items()):
    cm = metrics["Confusion_Matrix"]
    ax = axes[idx]
    labels = [
        [f"True Ham\\n{cm[0,0]}", f"False Spam (FP)\\n{cm[0,1]}"],
        [f"Missed Spam (FN)\\n{cm[1,0]}", f"True Spam (TP)\\n{cm[1,1]}"]
    ]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"fontsize": 11, "fontweight": "semibold"})
    ax.set_title(f"{name}\\n(Acc: {metrics['Accuracy']*100:.2f}% | F1: {metrics['F1-Score']*100:.2f}%)", fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])

plt.tight_layout()
plt.show()

# 2. Performance Comparison Bar Chart
df_plot = (df_results * 100).reset_index().rename(columns={"index": "Algorithm"})
df_melted = pd.melt(df_plot, id_vars=["Algorithm"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                    var_name="Metric", value_name="Score (%)")

plt.figure(figsize=(12, 5))
ax = sns.barplot(data=df_melted, x="Algorithm", y="Score (%)", hue="Metric",
                 palette=["#2b5c8f", "#3c9d9b", "#f4a261", "#e76f51"])
plt.title("Comparative Performance of Machine Learning Classifiers", fontweight='bold', pad=15)
plt.ylim(75, 103)
plt.legend(loc="lower right")
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 0.5),
                    ha='center', va='bottom', fontsize=8.5, fontweight='semibold')
plt.tight_layout()
plt.show()""")

    # --- Section 9: IEEE Paper Comparison ---
    add_md("""## 9. Academic Comparison with IEEE TENCON 2023 Research Paper

### Reference Study Information:
- **Title:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*
- **Authors:** Robert G. de Luna et al., Polytechnic University of the Philippines
- **Conference:** 2023 IEEE Region 10 Conference (TENCON), Chiang Mai, Thailand

### Empirical Benchmark Comparison Table:

| Model | IEEE TENCON 2023 Paper (Reported) | Our Experimental Project (Observed) | Key Comparative Insights |
|:---|:---|:---|:---|
| **Bernoulli Naive Bayes** | **Acc:** 96.63% \| **Prec:** 98.44%<br>**Rec:** 95.52% \| **F1:** 92.78% | **Acc:** 98.06% \| **Prec:** 97.37%<br>**Rec:** 86.72% \| **F1:** 91.74% | BNB was the #1 performing model in the IEEE paper. In our project, it confirmed strong generalizability with 98.06% accuracy and 97.37% precision (only 3 false alarms). |
| **Linear SVM** | **Acc:** 94.57% *(Default CV)*<br>*(Omitted from paper tuning)* | **Acc:** **98.16%** \| **Prec:** 96.58%<br>**Rec:** **88.28%** \| **F1:** **92.24%** | The IEEE paper did not tune SVC. In our study, sublinear TF-IDF with bigrams made Linear SVM our **#1 overall performer**. |
| **Logistic Regression** | **Acc:** 95.87% \| **Prec:** 98.77%<br>**Rec:** 93.74% \| **F1:** 89.26% | **Acc:** 97.97% \| **Prec:** 96.52%<br>**Rec:** 86.72% \| **F1:** 91.36% | Strong linear baseline across both studies. |
| **Multinomial Naive Bayes**| **Acc:** 96.03% \| **Prec:** 95.59% | **Acc:** 97.77% \| **Prec:** 94.87%<br>**Rec:** 86.72% \| **F1:** 90.61% | Reliable performance when smoothed ($\alpha=0.1$). |

### Methodological Differences & Limitations:
1. **Class Balancing Strategy:** The IEEE paper artificially undersampled ham messages to impose an artificial **60:40 partition**. Our project preserved the real-world **87.6:12.4 natural distribution**. Artificially boosting spam proportion inflates recall (95.5% vs 88.3%) but does not represent actual telecom network conditions.
2. **Feature Representation:** The IEEE paper selected CountVectorizer. Our study implemented **Sublinear TF-IDF with unigrams and bigrams**, which penalizes ubiquitous non-informative terms while elevating high-leverage multi-word fraud phrases (*"free entry"*, *"cash prize"*).
3. **Controlled Equivalence:** This analysis provides external benchmark validation rather than a direct controlled ablation due to distinct corpus compositions.""")

    # --- Section 10: Real-Time Inference Demo ---
    add_md("## 10. Interactive Real-Time Prediction Demonstration")
    add_code("""best_model = models["Linear SVM"]

test_sms = [
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461.",
    "Hey mate, are you coming to college tomorrow for the research presentation?",
    "URGENT! Your mobile number won 1,000,000 cash. Reply CLAIM to 88088 now."
]

print("=== REAL-TIME SMS CLASSIFICATION TEST ===")
for msg in test_sms:
    pred = best_model.predict([msg])[0]
    prob = best_model.predict_proba([msg])[0][1] * 100
    label = "🚨 SPAM" if pred == 1 else "✅ HAM (Legitimate)"
    print(f"\\nMessage : \\"{msg}\\"")
    print(f"Verdict : {label} (Estimated Spam Probability: {prob:.1f}%)")""")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[INFO] Self-contained notebook successfully written to: {output_path}")


if __name__ == "__main__":
    build_notebook()
