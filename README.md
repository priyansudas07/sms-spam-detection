# Comparative Analysis of ML Algorithms for SMS Spam Detection

**Academic Research Project**  
*Subject:* Research Methodologies (RM) — 7th Semester B.Tech in Computer Science and Engineering  
*Repository:* [sms-spam-detection](https://github.com/priyansudas07/sms-spam-detection)  

---

## 1. Project Overview & Problem Statement

Short Message Service (SMS) remains one of the most widely utilized and accessible communication systems globally, operating independently of internet connectivity. However, the open architecture of telecommunication protocols has made mobile subscribers vulnerable to unauthorized, fraudulent, and unsolicited text messages (**SMS Spam**). Such messages often involve banking fraud, credential harvesting, lottery scams, and malware distribution.

This academic project presents an empirical investigation into traditional Machine Learning (ML) classifiers for automated SMS spam detection. Using natural language processing (NLP) techniques and term frequency-inverse document frequency (TF-IDF) feature engineering, we evaluate, optimize, and compare four supervised classification algorithms.

Furthermore, we conduct an academic comparative analysis against published benchmarks from a peer-reviewed IEEE conference study:
> **Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
> **Authors:** Robert G. de Luna et al.  
> **Conference:** *2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand (IEEE Cat. No. CFP23TEN-ART, ISBN 979-8-3503-0219-6/23).

---

## 2. Research Objectives

1. **Empirical Modeling:** Construct and train four distinct machine learning classifiers on SMS text:
   - Multinomial Naive Bayes (MNB)
   - Bernoulli Naive Bayes (BNB)
   - Logistic Regression (LR)
   - Linear Support Vector Machine (Linear SVM)
2. **Data Leakage Prevention:** Implement end-to-end Scikit-learn pipelines where NLP preprocessing, vocabulary construction, and vectorization are strictly fitted only on training partitions.
3. **Multi-Metric Evaluation:** Evaluate generalization capability on an untouched 20% test fold using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
4. **External Benchmark Comparison:** Analyze performance differences and methodological trade-offs between our findings and the IEEE TENCON 2023 paper.
5. **Interactive Deployment:** Deliver a functional web application using Streamlit for real-time message inference.

---

## 3. Dataset Information

- **Name:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Source:** UCI Machine Learning Repository
- **Raw Samples:** 5,574 English SMS messages tagged as either `ham` (legitimate) or `spam`.
- **Missing Values:** 0 null entries across all attributes.
- **Duplicates Handled:** 414 duplicate messages identified and eliminated to prevent synthetic inflation of validation metrics.
- **Cleaned Dataset Total:** 5,160 unique messages.

### Class Imbalance Profile:
| Partition | Legitimate Messages (Ham) | Spam Messages | Total Messages | Imbalance Ratio |
|:---|:---:|:---:|:---:|:---:|
| **Raw Dataset** | 4,827 (86.60%) | 747 (13.40%) | 5,574 | ~6.5 : 1 |
| **Cleaned Dataset** | 4,518 (87.56%) | 642 (12.44%) | 5,160 | ~7.0 : 1 |

---

## 4. Methodology & Workflow Architecture

The research methodology follows a structured pipeline:

```
┌─────────────────────────┐
│ Raw UCI SMS Collection  │ (5,574 samples)
└────────────┬────────────┘
             │ Deduplication & Whitespace Cleaning
┌────────────▼────────────┐
│ Cleaned Dataset         │ (5,160 unique samples: 4,518 Ham, 642 Spam)
└────────────┬────────────┘
             │ Stratified 80:20 Train-Test Split (random_state=42)
      ┌──────┴─────────────────────────┐
      ▼                                ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ Training Split (80%)    │    │ Unseen Test Split (20%) │
│ 4,128 messages          │    │ 1,032 messages          │
└────────────┬────────────┘    └────────────┬────────────┘
             │                              │
             ├────────────────────────┐     │
             ▼                        ▼     │
      [NLP Preprocessing]      [Evaluation Pipeline]
      - Lowercase conversion                ▲
      - Noise & URL removal                 │
      - Whitespace Tokenization             │
      - Stopword removal                    │
      - Porter Stemming                     │
             ▼                              │
      [TF-IDF Feature Matrix]               │
      - Unigrams + Bigrams (1, 2)           │
      - Continuous (MNB, LR, SVM)           │
      - Binarized presence/absence (BNB)    │
             ▼                              │
      [5-Fold CV Grid Search]               │
      - MNB, BNB, LR, Linear SVM            │
             │                              │
             └──────────────────────────────┘
```

### Preprocessing Transformation Impact:
| Metric | Before Transformation | After Transformation | Reduction |
|:---|:---:|:---:|:---:|
| **Characters** | 408,360 | 246,088 | **-39.74%** |
| **Total Words** | 79,521 | 44,883 | **-43.56%** |
| **Unique Vocabulary** | 13,627 | 8,001 | **-41.29%** |

---

## 5. Experimental Results & Performance Comparison

All four tuned models were evaluated on the untouched test partition (`1,032` samples: 904 Ham, 128 Spam).

### Quantitative Metrics Summary:
| Machine Learning Algorithm | Accuracy (%) | Precision (Spam) (%) | Recall (Spam) (%) | F1-Score (Spam) (%) |
|:---|:---:|:---:|:---:|:---:|
| **Multinomial Naive Bayes** | 97.77% | 94.87% | 86.72% | 90.61% |
| **Bernoulli Naive Bayes** | 98.06% | **97.37%** | 86.72% | 91.74% |
| **Logistic Regression** | 97.97% | 96.52% | 86.72% | 91.36% |
| **Linear SVM** *(Top Performer)* | **98.16%** | 96.58% | **88.28%** | **92.24%** |

### Confusion Matrix Breakdown (Test Fold: 904 Ham, 128 Spam):
| Algorithm | True Ham (TN) | False Positives (FP) *(False Alarm)* | False Negatives (FN) *(Missed Spam)* | True Spam (TP) |
|:---|:---:|:---:|:---:|:---:|
| **Linear SVM** | 900 (99.6%) | 4 (0.4%) | 15 (11.7%) | **113 (88.3%)** |
| **Bernoulli Naive Bayes** | **901 (99.7%)** | **3 (0.3%)** | 17 (13.3%) | 111 (86.7%) |
| **Logistic Regression** | 900 (99.6%) | 4 (0.4%) | 17 (13.3%) | 111 (86.7%) |
| **Multinomial Naive Bayes** | 898 (99.3%) | 6 (0.7%) | 17 (13.3%) | 111 (86.7%) |

### Research Observations:
1. **Superiority of Linear SVM:** Linear SVM attained the highest overall accuracy (**98.16%**) and F1-score (**92.24%**). Sparse, high-dimensional TF-IDF feature spaces are well known to be linearly separable, allowing the maximum-margin hyperplane of SVM to generalize effectively with minimal overfitting.
2. **High Precision of Bernoulli Naive Bayes:** When modeled strictly with binary presence/absence indicator features, BNB produced the highest precision (**97.37%**) with only 3 false alarms out of 904 legitimate messages. In practical anti-spam filtering, minimizing False Positives is paramount to prevent blocking authentic personal or transactional communications.

---

## 6. Comparative Analysis with IEEE TENCON 2023 Study

### Benchmark Comparison Table:
| Algorithm | IEEE TENCON 2023 Reference Paper | Our Experimental Project | Academic Variance Analysis |
|:---|:---|:---|:---|
| **Bernoulli Naive Bayes** | **Acc:** 96.63% \| **Prec:** 98.44%<br>**Rec:** 95.52% \| **F1:** 92.78% | **Acc:** 98.06% \| **Prec:** 97.37%<br>**Rec:** 86.72% \| **F1:** 91.74% | BNB was the #1 model in the IEEE paper. In our project, it confirmed strong generalizability with 98.06% accuracy and 91.74% F1-score across a naturally imbalanced distribution. |
| **Linear SVM** | **Acc:** 94.57% *(Default CV)*<br>*(Omitted from paper's tuning)* | **Acc:** **98.16%** \| **Prec:** 96.58%<br>**Rec:** **88.28%** \| **F1:** **92.24%** | The IEEE study evaluated SVC only with default settings (ranking 5th) and did not optimize it. In our project, tuning $C=2.0$ with TF-IDF n-grams established Linear SVM as the top model. |
| **Logistic Regression** | **Acc:** 95.87% \| **Prec:** 98.77%<br>**Rec:** 93.74% \| **F1:** 89.26% | **Acc:** 97.97% \| **Prec:** 96.52%<br>**Rec:** 86.72% \| **F1:** 91.36% | Strong linear baseline across both studies; sublinear TF-IDF scaling yielded higher accuracy and F1 in our experiments. |
| **Multinomial Naive Bayes** | **Acc:** 96.03% \| **Prec:** 95.59% | **Acc:** 97.77% \| **Prec:** 94.87%<br>**Rec:** 86.72% \| **F1:** 90.61% | Comparable performance when laplace smoothing was tuned ($\alpha=0.1$). |

### Methodological Differences & Research Limitations:
1. **Class Balancing Strategy:** The reference paper forced an artificial **60:40 Ham-to-Spam partition** by undersampling ham messages. Our project preserved the real-world **87.6:12.4 natural distribution**. Artificially increasing spam presence inflates model recall (95.5% vs 88.3%) because spam patterns appear at triple their true incidence rate.
2. **Feature Extraction:** The reference paper selected CountVectorizer (raw token frequency). We adopted **Sublinear TF-IDF with unigrams and bigrams**, which penalizes high-frequency non-informative terms while capturing multi-word phishing phrases (*"free entry"*, *"cash prize"*, *"claim now"*).
3. **Controlled Equivalence:** This comparison serves as an external benchmark validation rather than a direct controlled ablation due to variance in corpora and feature spaces.

---

## 7. Project Structure

```text
sms-spam-detection/
│
├── data/
│   └── README.md                       # Dataset provenance and download guide
│
├── notebooks/
│   └── sms_spam_detection.ipynb        # End-to-end research notebook (EDA to evaluation)
│
├── src/
│   ├── data_loader.py                  # Dataset fetching, loading, and cleaning
│   ├── preprocessing.py                # NLP pipeline (stemming, tokenization, stopword filtering)
│   ├── feature_extraction.py           # TF-IDF vectorization and stratified train-test splitting
│   ├── model_training.py               # 4 ML pipeline definitions and 5-fold CV grid search
│   ├── evaluation.py                   # Metrics calculation, confusion matrices, and charts
│   ├── paper_comparison.py             # IEEE TENCON 2023 comparative analysis script
│   └── create_notebook.py              # Notebook generation utility
│
├── reports/
│   ├── research_comparison.md          # Comprehensive academic comparison report
│   └── figures/
│       ├── class_distribution.png
│       ├── message_length_distribution.png
│       ├── confusion_matrices.png
│       ├── model_comparison_barchart.png
│       └── paper_vs_our_comparison.png
│
├── app.py                              # Streamlit real-time web application
├── requirements.txt                    # Project environment dependencies
├── README.md                           # Main research documentation
├── LICENSE                             # Permissive open-source license
└── .gitignore                          # Clean repository version control ignore rules
```

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.9+ installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/priyansudas07/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Research Experiment
To run the full data download, cleaning, training, tuning, and evaluation pipeline:
```bash
python src/evaluation.py
python src/paper_comparison.py
```

### 4. Run the Jupyter / Google Colab Notebook
Open `notebooks/sms_spam_detection.ipynb` in VS Code, JupyterLab, or upload to Google Colab.

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the SMS Spam Detector with custom or preset messages.

---

## 9. Future Scope

1. **Transformer-Based Comparison:** Benchmark lightweight transformer models (DistilBERT, MobileBERT) against traditional ML models to evaluate inference latency vs. accuracy trade-offs on edge mobile devices.
2. **Adversarial Robustness:** Study evasion techniques such as character obfuscation (homoglyphs, leetspeak, zero-width characters) and develop character-level defense mechanisms.
3. **Multilingual and Code-Mixed SMS Support:** Expand vocabulary coverage for mixed-language SMS formats (e.g., Hinglish, Taglish) commonly observed in emerging telecommunication markets.

---
*Developed for 7th Semester B.Tech Project in Research Methodologies (RM).*
