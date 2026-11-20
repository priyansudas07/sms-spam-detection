# Comparative Analysis of ML Algorithms for SMS Spam Detection

**Academic Research Project**  
*Subject:* Research Methodologies (RM) — 7th Semester B.Tech Computer Science and Engineering  
*Repository:* [https://github.com/priyansudas07/sms-spam-detection](https://github.com/priyansudas07/sms-spam-detection)  

---

## 1. Project Overview & Research Context

Short Message Service (SMS) remains a critical worldwide telecommunications channel for peer-to-peer messaging, banking transactions, and Multi-Factor Authentication (MFA). However, mobile subscribers are frequently targeted by unauthorized, fraudulent, and promotional text messages (**SMS Spam**). Such spam ranges from nuisance advertising to credential phishing (smishing) and monetary fraud.

This academic project presents a comprehensive, reproducible comparative research study of supervised Machine Learning algorithms for SMS spam classification. We implement a **$2 \times 4$ full factorial experimental design** evaluating two feature extraction methods across four classic ML classifiers:

$$\text{2 Feature Extractors (TF-IDF vs. CountVectorizer)} \times \text{4 Classifiers (MNB, BNB, LR, Linear SVM)} = \mathbf{8\text{ Experimental Combinations}}$$

Furthermore, this project provides a formal external benchmark validation against the published results of the reference conference paper:
> **Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
> **Authors:** Robert G. de Luna et al., Polytechnic University of the Philippines  
> **Conference:** *2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand (IEEE Cat. No. CFP23TEN-ART, ISBN 979-8-3503-0219-6/23).

---

## 2. Experimental Design ($2 \times 4$ Factorial Matrix)

| Experiment # | Feature Extraction Technique | Classifier Architecture | Mathematical Specification |
|:---:|:---|:---|:---|
| **E1** | **TF-IDF Vectorizer** | Multinomial Naive Bayes (MNB) | Sublinear TF, unigram+bigram, continuous weights |
| **E2** | **TF-IDF Vectorizer** | Bernoulli Naive Bayes (BNB) | Binarized TF-IDF (1 for presence, 0 for absence) |
| **E3** | **TF-IDF Vectorizer** | Logistic Regression (LR) | Sublinear TF, unigram+bigram, $L_2$ regularized |
| **E4** | **TF-IDF Vectorizer** | Linear Support Vector Machine | Sublinear TF, maximum-margin separating hyperplane |
| **E5** | **CountVectorizer** | Multinomial Naive Bayes (MNB) | Absolute token counts, unigrams + bigrams |
| **E6** | **CountVectorizer** | Bernoulli Naive Bayes (BNB) | Boolean term presence indicator matrix ($x_{ij} \in \{0, 1\}$) |
| **E7** | **CountVectorizer** | Logistic Regression (LR) | Raw token counts, unigram+bigram, $L_2$ regularized |
| **E8** | **CountVectorizer** | Linear Support Vector Machine | Raw token counts, maximum-margin separating hyperplane |

---

## 3. Dataset & Data Preprocessing

- **Dataset:** [UCI SMS Spam Collection Dataset (ID: 228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Raw Samples:** 5,574 English SMS messages (4,827 Ham, 747 Spam).
- **Deduplication:** 414 duplicate messages eliminated to prevent synthetic inflation of test scores.
- **Cleaned Dataset Total:** 5,160 unique messages (**4,518 Ham [87.56%]**, **642 Spam [12.44%]**).
- **Natural Class Distribution:** Preserved authentic real-world imbalance without artificial undersampling.
- **NLP Preprocessing:** Lowercasing, regex stripping of URLs/emails/punctuation, whitespace tokenization, English stopword removal, and Porter stemming. Resulted in a **39.7% character reduction** and a **41.3% unique vocabulary reduction** (13,627 $\to$ 8,001 terms).
- **Zero Data Leakage:** Stratified 80:20 split (`4,128` train, `1,032` test) with all feature extractors fitted exclusively on `X_train` inside Scikit-learn Pipelines.

---

## 4. Quantitative Results & Multi-Metric Evaluation

All eight experimental combinations were evaluated on the untouched test partition (**1,032 samples: 904 Ham, 128 Spam**):

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

### Key Research Insights:
1. **Overall Best Performer:** **Linear SVM with TF-IDF (E4)** achieved the highest test accuracy (**98.16%**) and highest $F_1$-score (**92.24%**), intercepting 113 out of 128 spam messages while triggering only 4 false alarms.
2. **Feature Extraction Contrast:**
   - **Linear SVM and Logistic Regression** perform better with **TF-IDF**, as sublinear frequency scaling damps generic word repetition while accentuating discriminative n-grams (*"free entry"*, *"cash prize"*).
   - **Multinomial Naive Bayes** achieved higher Recall (**89.06%**) under **CountVectorizer**, aligning with discrete multinomial likelihood assumptions.
   - **Bernoulli Naive Bayes** yielded identical, consistent performance across both extractors (**98.06% Accuracy, 91.74% $F_1$**), confirming that strict binary indicator matrices ($x_{ij} \in \{0, 1\}$) eliminate extractor disparity.
3. **Operational Specificity:** In mobile networks, False Positives (blocking legitimate personal/banking SMS) are catastrophic. **Bernoulli NB** and **CountVec + Linear SVM** produced only **3 false alarms out of 904 legitimate messages** (99.67% specificity).

---

## 5. Benchmark Comparison with IEEE TENCON 2023 Study

### Published IEEE Reference Benchmarks (Table VII in de Luna et al.):
The reference study evaluated eleven classifiers with CountVectorizer and performed GridSearchCV on its top four:
* **Bernoulli Naive Bayes (BNB):** Accuracy = **96.6333%**, Precision = 98.4431%, Recall = 95.5174%, $F_1$ = 92.7835%
* **Random Forest Classifier (RFC):** Accuracy = **96.0297%**, Precision = 98.5228%, Recall = 94.8699%, $F_1$ = 91.1408%  
  *(Important Correction: 96.0297% belongs to Random Forest, NOT Multinomial Naive Bayes).*
* **Logistic Regression (LR):** Accuracy = **95.8709%**, Precision = 98.7711%, Recall = 93.7399%, $F_1$ = 89.2555%
* **Extra Trees Classifier (ETC):** Accuracy = **95.8070%**, Precision = 98.4885%, Recall = 94.3495%, $F_1$ = 90.5663%

### Academic Variance & Limitations:
1. **Class Balancing Strategy:** The IEEE study artificially undersampled ham messages to force an approximate **60:40 Ham-to-Spam partition**. Our project preserved the authentic real-world **87.6:12.4 natural distribution**. Artificially increasing spam proportion inflates model recall (95.5% vs 88.3%) because spam messages occur at 3x their real-world frequency.
2. **Feature Extraction Differences:** The IEEE paper selected CountVectorizer based on an initial MNB test. Our factorial study reveals that **TF-IDF with sublinear scaling achieves superior performance for Linear SVM and Logistic Regression**.
3. **Controlled Equivalence Disclaimer:** This comparison functions as an **external benchmark validation**, not a direct controlled ablation, due to distinct regional datasets and sampling procedures.

---

## 6. Project Architecture

```text
sms-spam-detection/
│
├── data/
│   └── README.md                       # Dataset provenance and download guide
│
├── notebooks/
│   └── sms_spam_detection.ipynb        # End-to-end research notebook (8-model experiment)
│
├── src/
│   ├── data_loader.py                  # Dataset fetching, loading, and cleaning
│   ├── preprocessing.py                # NLP pipeline (stemming, tokenization, stopword filtering)
│   ├── feature_extraction.py           # Dual vectorizers (TF-IDF vs Count) & binary support
│   ├── model_training.py               # 8 ML pipeline definitions and 5-fold CV grid search
│   ├── evaluation.py                   # Multi-metric evaluation, confusion matrices, and charts
│   ├── paper_comparison.py             # IEEE TENCON 2023 comparative analysis script
│   └── create_notebook.py              # Notebook generation utility
│
├── reports/
│   ├── research_report.md              # Formal 18-section academic research report
│   ├── research_comparison.md          # In-depth IEEE paper comparison document
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
├── app.py                              # Streamlit real-time web application (multi-model)
├── requirements.txt                    # Project environment dependencies
├── README.md                           # Main research documentation
├── LICENSE                             # Open-source license
└── .gitignore                          # Clean repository version control ignore rules
```

---

## 7. Installation & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/priyansudas07/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full 8-Model Factorial Experiment
```bash
python src/evaluation.py
python src/paper_comparison.py
```

### 4. Run the Jupyter / Google Colab Notebook
Open `notebooks/sms_spam_detection.ipynb` in VS Code, JupyterLab, or upload directly to Google Colab.

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the SMS Spam Detector with custom or preset messages, toggle between TF-IDF and CountVectorizer, and view the 8-model performance dashboard.

---
*Developed for 7th Semester B.Tech Project in Research Methodologies (RM).*
