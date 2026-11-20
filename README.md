# Comparative Analysis of Machine Learning Algorithms for SMS Spam Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](http://localhost:8501)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)](#)

A machine learning research project exploring and comparing traditional classification algorithms for identifying spam and legitimate SMS messages using Natural Language Processing (NLP).

[Live Demo](#live-application) · [Documentation](#methodology) · [Research Comparison](#comparison-with-ieee-research) · [Author](#author)

---

## Overview

SMS messaging remains a universal telecommunication channel, but it is frequently exploited for unsolicited advertisements, fraudulent financial schemes, and smishing attacks.

This project investigates the effectiveness of traditional machine learning algorithms in distinguishing spam messages from legitimate messages (ham).

Using the official **UCI SMS Spam Collection dataset**, the project applies natural language preprocessing, feature extraction (**TF-IDF** and **CountVectorizer**), and supervised machine learning techniques to construct, optimize, and evaluate classification models under a **$2 \times 4$ full factorial experimental design**.

The primary objective is to compare model performance using multiple evaluation metrics, analyze the impact of feature extraction techniques, and identify the most suitable classifier for authentic imbalanced SMS traffic.

---

## Key Features

- **Binary Text Classification:** Accurate detection of SMS spam versus legitimate ham.
- **NLP Normalization Pipeline:** Lowercasing, URL/email/punctuation removal, whitespace tokenization, stopword removal, and Porter stemming.
- **Dual Feature Extraction:** Systematic comparison between **TF-IDF Vectorizer** (with sublinear frequency scaling) and **CountVectorizer** (Bag-of-Words).
- **Mathematical Binarization:** Strict binary presence/absence indicator representation ($x_{ij} \in \{0, 1\}$) for Bernoulli Naive Bayes.
- **Zero Data Leakage:** Preprocessing and vectorizer fitting restricted exclusively to training data via Scikit-learn Pipelines.
- **Hyperparameter Optimization:** Stratified 5-Fold Cross Validation using `GridSearchCV`.
- **Multi-Metric Evaluation:** Accuracy, Precision, Recall, $F_1$-score, and Confusion Matrices on an untouched test split.
- **Visual Analytics:** High-resolution charts for metric comparison, macro feature extraction differences, and an 8-panel confusion matrix grid.
- **SIGNAL Workstation App:** Real-time interactive Streamlit web application with custom telecommunications security lab styling.
- **Academic Benchmark Validation:** Detailed comparison against published results from IEEE TENCON 2023 (*Robert G. de Luna et al.*).

---

## Research Objectives

1. To develop a machine learning-based SMS spam detection system with zero data leakage.
2. To investigate the effectiveness of different traditional classification algorithms.
3. To conduct a $2 \times 4$ factorial comparison of **TF-IDF** versus **CountVectorizer** across all models.
4. To evaluate models thoroughly using accuracy, precision, recall, and $F_1$-score on natural imbalanced data.
5. To identify the best-performing algorithm and feature extraction combination based on actual experimental results.
6. To benchmark experimental findings against an existing IEEE research paper.

---

## Methodology

The project follows a structured research methodology:

```
┌─────────────────────────────────┐
│     Raw UCI SMS Collection      │ (5,574 samples)
└───────────────┬─────────────────┘
                │ Data Cleaning & Deduplication (414 duplicates dropped)
┌───────────────▼─────────────────┐
│   Cleaned Natural Distribution  │ (5,160 unique messages: 87.56% Ham, 12.44% Spam)
└───────────────┬─────────────────┘
                │ Stratified 80:20 Split (random_state=42)
         ┌──────┴─────────────────────────┐
         ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│ Training Split  │ (4,128 msgs) │ Testing Split   │ (1,032 msgs)
└────────┬────────┘              └────────┬────────┘
         │                                │
         ├────────────────────────┐       │
         ▼                        ▼       │
  [NLP Preprocessing]      [Evaluation Pipeline]
  - Lowercase, URLs, Punct                ▲
  - Stopwords, Stemming                   │
         ▼                                │
  [Feature Extraction]                    │
  - TF-IDF (Cont. & Binary)               │
  - CountVec (Freq. & Binary)             │
         ▼                                │
  [5-Fold Stratified CV]                  │
  - MNB, BNB, LR, Linear SVM              │
         │                                │
         └────────────────────────────────┘
```

---

## Dataset

**Source:** [UCI Machine Learning Repository — SMS Spam Collection (ID: 228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

The dataset contains real SMS messages labelled as:
- **Ham:** Legitimate personal, transactional, or informational messages.
- **Spam:** Unwanted promotions, lottery lures, or phishing attempts.

### Dataset Preparation & Verification:

| Dataset Property | Verified Documented Value |
|:---|:---:|
| **Original Raw Messages** | 5,574 |
| **Missing Values** | 0 |
| **Duplicate Messages Removed** | 414 |
| **Cleaned Unique Messages** | 5,160 |
| **Legitimate Messages (Ham)** | 4,518 (87.56%) |
| **Spam Messages** | 642 (12.44%) |
| **Training Partition (80%)** | 4,128 |
| **Testing Partition (20%)** | 1,032 |

---

## Machine Learning Algorithms

Four supervised machine learning algorithms are evaluated:

| Algorithm | Description | Mathematical Basis |
|:---|:---|:---|
| **Multinomial Naive Bayes (MNB)** | Probabilistic text classifier | Assumes features follow a multinomial distribution representing word frequencies. |
| **Bernoulli Naive Bayes (BNB)** | Multivariate Bernoulli model | Operates strictly on binary token presence/absence indicators ($x_{ij} \in \{0, 1\}$). |
| **Logistic Regression (LR)** | Linear log-odds classifier | Models class posterior probabilities using the logistic sigmoid function with $L_2$ regularization. |
| **Linear Support Vector Machine (Linear SVM)** | Maximum-margin separating hyperplane | Maximizes the geometric margin between classes in high-dimensional sparse feature space. |

---

## Feature Extraction

This research evaluates and compares two foundational feature extraction techniques:

1. **TF-IDF (Term Frequency–Inverse Document Frequency):**
   - Assigns numerical weights based on word frequency within a message and inverse rarity across the corpus.
   - Configured with sublinear term-frequency scaling to dampen the effect of repetitive generic vocabulary.
2. **CountVectorizer (Bag-of-Words):**
   - Encodes absolute occurrence frequencies of tokens and token pairs (unigrams + bigrams).
   - Serves as the baseline comparison method employed in the reference IEEE TENCON 2023 study.

---

## Model Evaluation & Experimental Results

Hyperparameter optimization was performed using 5-fold stratified cross-validation on `X_train`. All eight pipelines were evaluated on the reserved untouched test partition (**1,032 samples: 904 Ham, 128 Spam**):

| Exp # | Feature Extraction | Algorithm | Accuracy | Precision (Spam) | Recall (Spam) | $F_1$-Score (Spam) | False Positives (FP) | Missed Spam (FN) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | **TF-IDF** | Multinomial NB | 97.67% | 94.07% | 86.72% | 90.24% | 7 / 904 | 17 / 128 |
| **E2** | **TF-IDF** | Bernoulli NB *(Binary)* | 98.06% | **97.37%** | 86.72% | 91.74% | **3 / 904** | 17 / 128 |
| **E3** | **TF-IDF** | Logistic Regression | 97.97% | 96.52% | 86.72% | 91.36% | 4 / 904 | 17 / 128 |
| **E4** | **TF-IDF** | Linear SVM *(Top Overall)* | **98.16%** | 96.58% | 88.28% | **92.24%** | 4 / 904 | **15 / 128** |
| **E5** | **CountVectorizer** | Multinomial NB | 97.77% | 92.68% | **89.06%** | 90.84% | 9 / 904 | **14 / 128** |
| **E6** | **CountVectorizer** | Bernoulli NB *(Binary)* | 98.06% | **97.37%** | 86.72% | 91.74% | **3 / 904** | 17 / 128 |
| **E7** | **CountVectorizer** | Logistic Regression | 97.87% | 97.32% | 85.16% | 90.83% | **3 / 904** | 19 / 128 |
| **E8** | **CountVectorizer** | Linear SVM | **98.16%** | **97.39%** | 87.50% | 92.18% | **3 / 904** | 16 / 128 |

---

## Best Performing Model

Based on test results:
- **Top Overall Classifier:** **Linear Support Vector Machine (Linear SVM) with TF-IDF** achieved the highest test accuracy (**98.16%**) and highest $F_1$-score (**92.24%**).
- **Highest Precision (Lowest False Alarms):** **CountVectorizer + Linear SVM** (**97.39%**) and **Bernoulli Naive Bayes** (**97.37%**), registering only **3 false positives out of 904 legitimate messages** (99.67% specificity). In telecommunications filtering, minimizing false positives is paramount to ensure legitimate personal and banking messages are never improperly blocked.

---

## Comparison with IEEE Research

This project benchmarks its methodology against the published IEEE TENCON 2023 paper:
> **A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)**  
> *Authors:* Robert G. de Luna et al., Polytechnic University of the Philippines (IEEE TENCON 2023, Chiang Mai, Thailand).

The IEEE study evaluated eleven classification algorithms using a combination of UCI and additional regional SMS datasets, undersampled ham messages to force an approximate 60:40 partition, and selected CountVectorizer as its final feature extraction technique.

### Published Benchmarks from Table VII of de Luna et al.:
- **Bernoulli Naive Bayes (BNB):** 96.6333% Accuracy | 98.4431% Precision | 95.5174% Recall | 92.7835% $F_1$
- **Random Forest Classifier (RFC):** 96.0297% Accuracy | 98.5228% Precision | 94.8699% Recall | 91.1408% $F_1$ *(Note: 96.03% belongs to Random Forest, NOT Multinomial Naive Bayes)*
- **Logistic Regression (LR):** 95.8709% Accuracy | 98.7711% Precision | 93.7399% Recall | 89.2555% $F_1$
- **Extra Trees Classifier (ETC):** 95.8070% Accuracy | 98.4885% Precision | 94.3495% Recall | 90.5663% $F_1$

### Key Methodological Differences:

| Aspect | This Project | IEEE TENCON 2023 Study |
|:---|:---|:---|
| **Dataset** | Authentic UCI SMS Spam Collection | Merged datasets (UCI + Philippine + Student SMS) |
| **Class Distribution** | Authentic natural distribution (87.56% : 12.44%) | Artificially balanced (~60:40) via ham undersampling |
| **Feature Extraction** | Full Factorial (TF-IDF vs. CountVectorizer) | CountVectorizer |
| **Algorithms Evaluated** | 4 classical models ($2 \times 4$ combinations) | 11 classifiers |
| **Top Performing Model** | Linear SVM (TF-IDF) | Bernoulli Naive Bayes |
| **Top Model Accuracy** | **98.16%** | **96.6333%** |

*Important Academic Note:* The experiments use different regional datasets, class distributions, and feature extraction techniques. Therefore, reported accuracy values should be viewed as external benchmark validations rather than a direct controlled comparison under identical conditions.

---

## Live Application

The project includes an interactive web workstation built with Streamlit titled **SIGNAL — SMS Intelligence Lab**:

- **Real-Time SMS Analysis:** Enter custom or preset SMS text and receive immediate classification (**SPAM DETECTED** vs. **SAFE HAM**).
- **Interactive Model & Vectorizer Selection:** Toggle between TF-IDF and CountVectorizer across all 4 machine learning models.
- **Model Performance Dashboard:** Interactive 8-combination performance matrix and high-resolution comparison charts.
- **Dataset Explorer:** Corpus diagnostics, class distribution histograms, and sample data viewer.
- **Research Comparison:** Direct benchmark comparison and divergence analysis against IEEE TENCON 2023.

---

## Project Structure

```text
sms-spam-detection/
│
├── .streamlit/
│   └── config.toml                     # High-contrast light workstation theme configuration
│
├── data/
│   └── README.md                       # Dataset provenance and download guide
│
├── notebooks/
│   └── sms_spam_detection.ipynb        # End-to-end reproducible research notebook
│
├── src/
│   ├── data_loader.py                  # Dataset downloading, loading, and deduplication
│   ├── preprocessing.py                # NLP pipeline (stemming, tokenization, stopword removal)
│   ├── feature_extraction.py           # Dual vectorizers (TF-IDF vs Count) & binary support
│   ├── model_training.py               # 8 ML pipeline definitions and 5-fold CV grid search
│   ├── evaluation.py                   # Multi-metric evaluation, confusion matrices, and charts
│   ├── paper_comparison.py             # IEEE TENCON 2023 comparative analysis script
│   └── create_notebook.py              # Notebook generation utility
│
├── reports/
│   ├── research_report.md              # Formal 18-section academic research report
│   ├── research_comparison.md          # Detailed IEEE paper comparison report
│   ├── results/
│   │   ├── results_summary.csv         # Machine-readable evaluation table
│   │   └── results_summary.json        # Machine-readable evaluation JSON
│   └── figures/
│       ├── class_distribution.png
│       ├── message_length_distribution.png
│       ├── accuracy_comparison.png
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── f1_comparison.png
│       ├── feature_extraction_comparison.png
│       ├── confusion_matrices_all8.png
│       └── paper_vs_our_comparison.png
│
├── app.py                              # SIGNAL: SMS Intelligence Lab (Streamlit Workstation)
├── requirements.txt                    # Project environment dependencies
├── README.md                           # Main research documentation
├── LICENSE                             # MIT License
└── .gitignore                          # Clean repository version control ignore rules
```

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/priyansudas07/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full evaluation pipeline
```bash
python src/evaluation.py
python src/paper_comparison.py
```

### 5. Launch the Streamlit application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Technologies Used

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-learn (Pipelines, GridSearchCV, LinearSVC, Naive Bayes, Logistic Regression)
- **Natural Language Processing:** NLTK (Tokenization, English Stopwords, Porter Stemmer)
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Development & Research:** Jupyter Notebook, Google Colab, Git & GitHub

---

## Limitations

1. **Monolingual Focus:** The experiment utilizes a single publicly available English SMS corpus; performance on multilingual or code-mixed text (e.g., Hinglish) may require specialized language models.
2. **Contextual Nuances:** Traditional linear and probabilistic models operate on n-gram token statistics and do not capture deeper contextual semantics like modern transformers.
3. **Adversarial Tactics:** Real-world spammers may use character substitution (homoglyphs, leetspeak, zero-width spaces) which can bypass basic whitespace tokenizers.

---

## Future Scope

1. **Lightweight Transformers:** Benchmark DistilBERT and MobileBERT against Linear SVM to assess inference latency versus accuracy on mobile edge devices.
2. **Adversarial Hardening:** Incorporate character-level embeddings to identify character obfuscation and evasion tactics.
3. **Multi-Dialect & Code-Mixed Datasets:** Expand evaluation to include SMS datasets from diverse regional telecommunications markets.

---

## References

1. Robert G. de Luna et al., "A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)," in *Proc. 2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand, 2023, pp. 52–57.
2. T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," in *Proc. 11th ACM Symposium on Document Engineering (DocEng)*, 2011, pp. 259–262.
3. UCI Machine Learning Repository, *SMS Spam Collection Dataset*, ID: 228. Available: [https://archive.ics.uci.edu/dataset/228/sms+spam+collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
4. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

## License

This project is open-source and distributed under the terms of the **[MIT License](LICENSE)**.

---

## Author

**Priyansu Das**    
GitHub: [@priyansudas07](https://github.com/priyansudas07)
