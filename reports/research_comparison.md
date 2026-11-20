# Academic Comparative Analysis: Our Experiments vs. IEEE TENCON 2023 Study

**Subject:** Research Methodologies (RM) — 7th Semester B.Tech Project  
**Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
**Authors:** Robert G. de Luna, Karina L. Enriquez, Aira Mae Española, Mateo Ramos, Verna C. Magnaye, Dexter Astorga, Brian Allen Lanting, Jenjazel Redondo, Rose Anne L. Reaño, Trisha Celestial, Danielle Mugar  
**Venue:** *2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand, 31 Oct - 3 Nov 2023  
**DOI/ISBN:** 979-8-3503-0219-6/23/$31.00 ©2023 IEEE  

---

## 1. Executive Summary & Context

The primary goal of this comparative analysis is to benchmark our empirical experiments against the peer-reviewed IEEE TENCON 2023 study by **de Luna et al.** In accordance with scientific research methodology principles, this comparison is framed as an **external benchmark validation** rather than a direct controlled experiment, because the underlying corpora, class distributions, and feature engineering procedures differ.

---

## 2. Published IEEE Reference Results (Table VII in de Luna et al.)

The authors tested 11 classifiers and performed GridSearchCV hyperparameter optimization on their Top 4 algorithms:

| Rank | Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **Bernoulli Naive Bayes (BNB)** | **96.6333%** | 98.4431% | **95.5174%** | **92.7835%** |
| **2** | **Random Forest Classifier (RFC)** | **96.0297%** | 98.5228% | 94.8699% | 91.1408% |
| **3** | **Logistic Regression (LR)** | **95.8709%** | **98.7711%** | 93.7399% | 89.2555% |
| **4** | **Extra Trees Classifier (ETC)** | **95.8070%** | 98.4885% | 94.3495% | 90.5663% |

> **Important Academic Clarification:**  
> The 96.0297% accuracy in Table VII belongs to **Random Forest (RFC)**, an ensemble tree algorithm, **NOT Multinomial Naive Bayes**. In de Luna et al., Multinomial Naive Bayes was evaluated solely during preliminary vectorizer screening in Figure 2 (yielding 96.03% on CountVectorizer vs. 95.56% on TF-IDF) and was not included among the top four optimized models.

---

## 3. Our Experimental Results (Untouched Test Fold: 1,032 Samples)

| Exp # | Feature Extraction | Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FP (False Alarms) | FN (Missed Spam) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | **TF-IDF** | Multinomial NB | 97.67% | 94.07% | 86.72% | 90.24% | 7 | 17 |
| **E2** | **TF-IDF** | Bernoulli NB *(Binary)* | 98.06% | 97.37% | 86.72% | 91.74% | 3 | 17 |
| **E3** | **TF-IDF** | Logistic Regression | 97.97% | 96.52% | 86.72% | 91.36% | 4 | 17 |
| **E4** | **TF-IDF** | Linear SVM | 98.16% | 96.58% | 88.28% | 92.24% | 4 | 15 |
| **E5** | **CountVectorizer** | Multinomial NB | 97.77% | 92.68% | 89.06% | 90.84% | 9 | 14 |
| **E6** | **CountVectorizer** | Bernoulli NB *(Binary)* | 98.06% | 97.37% | 86.72% | 91.74% | 3 | 17 |
| **E7** | **CountVectorizer** | Logistic Regression | 97.87% | 97.32% | 85.16% | 90.83% | 3 | 19 |
| **E8** | **CountVectorizer** | Linear SVM | 98.16% | 97.39% | 87.50% | 92.18% | 3 | 16 |

---

## 4. Deep Methodological Variance Analysis

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEMATIC RESEARCH COMPARISON                            │
├─────────────────────────┬──────────────────────────────────┬───────────────────────────┤
│ Evaluation Dimension    │ IEEE TENCON 2023 (de Luna et al.)│ Our Project (Factorial RM)│
├─────────────────────────┼──────────────────────────────────┼───────────────────────────┤
│ Dataset Source          │ UCI + Philippine Spam + PUPSTB   │ Authentic UCI SMS Corpus  │
│ Total Cleaned Messages  │ 3,148 samples                    │ 5,160 unique samples      │
│ Class Distribution      │ Artificially Balanced (60:40)    │ Authentic Imbalance (87.6:12.4)│
│ Balancing Technique     │ Random Undersampling of Ham      │ None (Natural Stratified) │
│ Feature Extractions     │ CountVectorizer (Chosen in Fig 2)│ Full Factorial (TF-IDF vs Count)│
│ Bernoulli Event Model   │ Boolean occurrence modeling      │ Strict binary indicator   │
│ Cross-Validation        │ 5-Fold CV on training partition  │ Stratified 5-Fold CV      │
│ Top Reported Model      │ Bernoulli Naive Bayes (96.63%)   │ Linear SVM with TF-IDF    │
└─────────────────────────┴──────────────────────────────────┴───────────────────────────┘
```

### 4.1 Impact of Class Distribution (60:40 vs 87.6:12.4)
- **Reference Paper Findings:** The authors undersampled the ham messages to create an artificial 60% ham to 40% spam partition. Because spam messages occurred at approximately three times their natural rate, models were exposed to higher spam density, artificially elevating Recall (95.52% for BNB).
- **Our Project Findings:** We intentionally retained the authentic natural class distribution (87.56% Ham : 12.44% Spam). This is academically essential because telecommunication channels are naturally dominated by legitimate traffic. On our imbalanced test set, precision is exceptionally high (up to 97.37%), with only 3 to 4 false alarms out of 904 legitimate messages, which is the primary operational requirement of commercial anti-spam engines.

### 4.2 Feature Extraction: TF-IDF vs. CountVectorizer
- **Reference Paper Findings:** de Luna et al. conducted a preliminary test on Naive Bayes and observed CountVectorizer slightly surpassed TF-IDF (96.03% vs 95.56%), proceeding to use CountVectorizer exclusively for the remaining 10 algorithms.
- **Our Project Findings:** By completing a full 2x4 factorial comparison across all four algorithms:
  - Linear SVM and Logistic Regression perform superiorly under **TF-IDF with sublinear scaling** because continuous L2-normalized weights dampen the noise of repetitive spam words while rewarding discriminative bigrams ("free entry", "cash prize").
  - Multinomial Naive Bayes performs slightly better with **CountVectorizer** because discrete word counts align closely with multinomial likelihood assumptions.
  - Bernoulli Naive Bayes yields identical, highly competitive performance under both vectorizers when the presence/absence binary constraint is strictly enforced.

---

## 5. Research Limitations & Academic Disclaimer

1. **Non-Direct Controlled Equivalence:**  
   Because the training corpora (Philippine local messages vs. original UCI SMS), class balancing techniques, and feature matrices differ, our higher test accuracy (98.16% vs. 96.63%) should **not** be claimed as absolute algorithmic superiority. Rather, it validates model robustness under distinct real-world sampling strategies.
2. **False Positive Penalties:**  
   In mobile communications, a False Positive (marking a legitimate personal/banking SMS as spam) has severe consequences compared to a False Negative (a spam text slipping into the inbox). Both studies confirm that Naive Bayes and Linear SVM maintain exceptionally low false positive rates.

---
*Report generated for Research Methodologies (RM) 7th Semester B.Tech Project.*
