# Comparative Analysis of Machine Learning Algorithms for SMS Spam Detection

**Academic Research Project Report**  
**Subject:** Research Methodologies (RM) — 7th Semester B.Tech  
**Department:** Computer Science and Engineering  
**Repository:** [https://github.com/priyansudas07/sms-spam-detection](https://github.com/priyansudas07/sms-spam-detection)  

---

## 1. Abstract

Short Message Service (SMS) remains a critical worldwide communication channel, yet mobile subscribers are increasingly targeted by unsolicited fraudulent, promotional, and phishing text messages. This research study presents a comparative empirical investigation of supervised Machine Learning (ML) classifiers for automated SMS spam detection using the official UCI SMS Spam Collection dataset. We implement a $2 \times 4$ full factorial experimental design evaluating two feature extraction methodologies (**Term Frequency-Inverse Document Frequency [TF-IDF]** and **CountVectorizer [Bag-of-Words]**) across four classification algorithms: **Multinomial Naive Bayes (MNB)**, **Bernoulli Naive Bayes (BNB)**, **Logistic Regression (LR)**, and **Linear Support Vector Machine (Linear SVM)**. Text preprocessing incorporates case normalization, noise stripping, whitespace tokenization, stopword filtering, and Porter stemming, resulting in a 41.3% reduction in vocabulary size. Zero data leakage is enforced through Scikit-learn Pipeline encapsulation, and hyperparameter tuning is conducted via stratified 5-fold cross-validation. Evaluating all eight pipelines on an untouched 20% test fold (1,032 samples: 904 Ham, 128 Spam), **Linear SVM with TF-IDF** achieved the highest overall performance with **98.16% Accuracy** and **92.24% $F_1$-Score**. Furthermore, we conduct an external benchmark validation against the published findings of **de Luna et al. (IEEE TENCON 2023)**, analyzing the methodological implications of class balancing, feature representation choices, and false positive penalties in mobile networks.

---

## 2. Introduction

Short Message Service (SMS), introduced under the GSM standards, provides cellular text communication without requiring active internet access. Due to near-universal accessibility and open telecommunication protocols, SMS achieves open rates exceeding 90%. However, this high engagement makes mobile users prime targets for SMS spam. Spam messages range from benign commercial marketing to malicious spear-phishing (smishing), fraudulent banking lures, and malware distribution.

Traditional rule-based and keyword-blacklisting mechanisms fail against evolving spam patterns, where spammers alter syntax, introduce intentional misspellings, or mimic legitimate institutional alerts. Machine learning techniques provide an adaptive framework by learning underlying statistical and lexical distributions of legitimate (ham) versus fraudulent (spam) messages.

---

## 3. Problem Statement

Automated SMS spam classification poses three critical methodological challenges:
1. **Severe Natural Class Imbalance:** Legitimate text messages naturally outnumber spam messages by approximately 7:1 in real-world mobile networks. Models optimized purely for accuracy tend to exhibit high false-alarm rates or overlook minority spam occurrences.
2. **Short Text Sparsity:** SMS messages are bounded by character limits (typically 160 characters), resulting in sparse lexical representations with limited contextual signals compared to full-length emails.
3. **High Cost of False Positives:** In operational anti-spam systems, a False Positive (incorrectly blocking an authentic personal, institutional, or banking alert) carries significantly worse consequences than a False Negative (a spam message slipping into the inbox).

---

## 4. Research Objectives

The core objectives of this study are:
1. To implement an end-to-end NLP and machine learning pipeline for SMS classification using the official UCI SMS corpus.
2. To design and execute a $2 \times 4$ factorial experiment comparing continuous **TF-IDF Vectorization** against discrete **CountVectorizer** across four classical ML classifiers.
3. To strictly enforce the mathematical event model of **Bernoulli Naive Bayes** using binary presence/absence indicator matrices ($x_{ij} \in \{0, 1\}$).
4. To guarantee zero data leakage by isolating preprocessing and feature extraction fitting exclusively to training folds.
5. To evaluate model generalization on unseen test data using Accuracy, Precision, Recall, $F_1$-score, and Confusion Matrices.
6. To perform a structured, academically defensible comparative analysis against published benchmarks from the **IEEE TENCON 2023** reference study (*Robert G. de Luna et al.*).
7. To deliver an interactive Streamlit web application for real-time inference and model comparison.

---

## 5. Literature Review

Machine learning for text classification has been extensively researched:
- **Mukerjee (2020)** evaluated CountVectorizer with Naive Bayes, demonstrating that simple probabilistic classifiers achieve strong baselines in text filtering.
- **Krishnaveni and Radha (2021)** compared Naive Bayes and Support Vector Machines, concluding that SVM's margin-maximizing hyperplane delivers superior generalization in high-dimensional text spaces.
- **Kudupudi and Nair (2021)** utilized TF-IDF with Logistic Regression, demonstrating that inverse document frequency weighting dampens ubiquitous vocabulary and stabilizes linear decision boundaries.
- **Sonowal (2020)** investigated correlation-based feature selection to reduce feature dimensionality for ensemble methods in phishing detection.
- **Yerima and Bashar (2022)** examined semi-supervised novelty detection on non-spam data, highlighting the challenge of class imbalance in telecommunications.

---

## 6. Existing IEEE Research Study

Our benchmark reference study is:
* **Title:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*
* **Authors:** Robert G. de Luna et al., Polytechnic University of the Philippines
* **Venue:** *2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand
* **Conference Date:** October 31 – November 3, 2023 | **DOI/ISBN:** 979-8-3503-0219-6/23

### Key Aspects of the IEEE Study:
1. **Dataset:** Merged three data sources: UCI SMS Spam Collection, Philippine Spam SMS, and student-collected messages. Ham messages were artificially undersampled to force an approximate **60:40 Ham-to-Spam distribution** (1,930 ham, 1,218 spam, Total: 3,148).
2. **Feature Extraction:** Tested CountVectorizer vs. TF-IDF on Naive Bayes and selected **CountVectorizer**.
3. **Algorithms Evaluated:** Tested 11 classification algorithms; top four models were optimized via GridSearchCV.
4. **Reported Optimized Benchmark (Table VII):**
   - **Bernoulli Naive Bayes (BNB):** Accuracy = 96.6333%, Precision = 98.4431%, Recall = 95.5174%, $F_1$ = 92.7835%
   - **Random Forest Classifier (RFC):** Accuracy = 96.0297%, Precision = 98.5228%, Recall = 94.8699%, $F_1$ = 91.1408%
   - **Logistic Regression (LR):** Accuracy = 95.8709%, Precision = 98.7711%, Recall = 93.7399%, $F_1$ = 89.2555%
   - **Extra Trees Classifier (ETC):** Accuracy = 95.8070%, Precision = 98.4885%, Recall = 94.3495%, $F_1$ = 90.5663%
   *(Note: Random Forest achieved 96.03%, NOT Multinomial Naive Bayes).*

---

## 7. Proposed Methodology

The experimental framework follows a systematic 5-stage pipeline:
1. **Corpus Acquisition & Hygiene:** Ingestion of the official UCI SMS corpus, missing-value validation, and deduplication.
2. **Text Normalization:** NLP pipeline transforming unstructured strings into clean root tokens.
3. **Dual Feature Extraction:** Parallel transformation into TF-IDF and CountVectorizer matrices (with binary representations for BernoulliNB).
4. **Stratified Splitting & Optimization:** Stratified 80:20 split and 5-fold cross-validation `GridSearchCV` on training data.
5. **Multi-Metric Evaluation:** Unseen test set evaluation, visualization generation, and benchmark comparison.

```
┌────────────────────────────────────────────────────────┐
│             UCI SMS Corpus (5,574 Messages)            │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│     Deduplicated Clean Corpus (5,160 Unique Texts)     │
│       - 4,518 Ham (87.56%) | 642 Spam (12.44%)         │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│          Stratified 80:20 Split (random_state=42)      │
│     Train: 4,128 Samples    |    Test: 1,032 Samples   │
└────────────┬─────────────────────────────┬─────────────┘
             ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  NLP Pipeline (X_train)  │  │  NLP Pipeline (X_test)   │
│  - Lowercase, URLs, Punct│  │  - Transform only        │
│  - Stopwords, Stemming   │  │  - Zero leakage          │
└────────────┬─────────────┘  └────────────┬─────────────┘
             ▼                             │
┌──────────────────────────┐               │
│ Feature Extraction Grid  │               │
│ - TF-IDF (Cont. & Bin.)  │               │
│ - CountVec (Freq & Bin.) │               │
└────────────┬─────────────┘               │
             ▼                             │
┌──────────────────────────┐               │
│ 5-Fold Stratified CV Grid│               │
│ - MNB, BNB, LR, Lin. SVM │               │
└────────────┬─────────────┘               │
             ▼                             ▼
┌────────────────────────────────────────────────────────┐
│     Comprehensive Evaluation on Unseen Test Partition  │
│  Accuracy, Precision, Recall, F1, Confusion Matrices   │
└────────────────────────────────────────────────────────┘
```

---

## 8. Dataset Description

We utilize the official **UCI SMS Spam Collection dataset**:
- **Source:** [UCI Machine Learning Repository (ID: 228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Raw Samples:** 5,574 English text messages.
- **Attributes:** Two columns: `label` (`ham` or `spam`) and `message` (SMS string).
- **Missing Values:** Zero missing or null values.
- **Deduplication:** 414 duplicate messages removed to eliminate synthetic test leakage.
- **Cleaned Dataset Total:** 5,160 unique messages (4,518 Ham [87.56%], 642 Spam [12.44%]).

```
Figure: Cleaned Class Distribution
Ham:  4,518 (87.56%)
Spam:   642 (12.44%)
Ratio: ~7.0 : 1 natural imbalance
```

---

## 9. Text Preprocessing

Each text message undergoes sequential morphological cleaning:
1. **Case Normalization:** Converts all characters to lowercase to unify vocabulary tokens (e.g., `FREE`, `Free`, and `free` map to `free`).
2. **Noise and Hyperlink Stripping:** Regular expressions remove URLs (`https?://\S+`), email handles, and non-alphanumeric punctuation.
3. **Tokenization:** Breaks the message string into discrete word tokens.
4. **Stopword Removal:** Eliminates standard grammatical stopwords (`the`, `is`, `at`, `which`) using the NLTK English corpus.
5. **Stemming:** Applies the Porter Stemmer to reduce inflections to root morphemes (e.g., `winning`, `winner`, `wins` $\to$ `win`).
6. **Whitespace Consolidation:** Eliminates redundant spaces, tabs, and line breaks.

### Quantitative Text Transformation Impact:
| Corpus Metric | Before Transformation | After Transformation | Absolute Reduction |
|:---|:---:|:---:|:---:|
| **Character Volume** | 408,360 | 246,088 | **-39.74%** |
| **Total Word Tokens** | 79,521 | 44,883 | **-43.56%** |
| **Unique Vocabulary** | 13,627 | 8,001 | **-41.29%** |

---

## 10. Feature Extraction

We implement two distinct vectorization models configured for unigram and bigram extraction (`ngram_range=(1, 2)`) with a maximum vocabulary size of 4,000 features:

### 10.1 Term Frequency-Inverse Document Frequency (TF-IDF)
TF-IDF calculates the relative importance of token $t$ in message $d$ across corpus $D$:
$$\text{TF}(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Sublinear term-frequency scaling prevents highly repetitive spam words from disproportionately skewing document vectors.

### 10.2 CountVectorizer (Bag-of-Words)
CountVectorizer constructs document vectors based on absolute token counts:
$$x_{ij} = f_{j, i}$$
where $f_{j, i}$ is the frequency of word $j$ in document $i$.

### 10.3 Mathematical Binarization for Bernoulli Naive Bayes
BernoulliNB operates on a multivariate Bernoulli event model, assuming each feature is a binary Boolean indicator:
$$x_{ij} = \begin{cases} 1 & \text{if word } j \text{ occurs in document } i \\ 0 & \text{otherwise} \end{cases}$$
We explicitly configure `binary=True` in both vectorizers when paired with BernoulliNB.

---

## 11. Machine Learning Algorithms

Four classical supervised classification algorithms are evaluated:

1. **Multinomial Naive Bayes (MNB):** Models word frequency distributions under a multinomial distribution:
   $$P(c \mid \mathbf{x}) \propto P(c) \prod_{j=1}^{V} P(w_j \mid c)^{x_j}, \quad P(w_j \mid c) = \frac{N_{jc} + \alpha}{N_c + \alpha V}$$
2. **Bernoulli Naive Bayes (BNB):** Models presence/absence of tokens:
   $$P(c \mid \mathbf{x}) \propto P(c) \prod_{j=1}^{V} \left[ P(w_j \mid c)^{x_j} (1 - P(w_j \mid c))^{1 - x_j} \right]$$
3. **Logistic Regression (LR):** Parametric linear model optimizing log-odds:
   $$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
4. **Linear Support Vector Machine (Linear SVM):** Constructs a maximum-margin separating hyperplane:
   $$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

---

## 12. Experimental Setup

- **Hardware & Environment:** Python 3.14, Scikit-learn 1.9, Pandas 3.0, NLTK 3.10.
- **Dataset Partition:** Stratified 80:20 split (`4,128` training messages, `1,032` testing messages) with fixed seed (`random_state=42`).
- **Cross-Validation:** Stratified 5-fold cross-validation on `X_train` optimizing for positive class $F_1$-score.
- **Leakage Prevention:** Scikit-learn `Pipeline` objects fitted strictly on training folds.

---

## 13. Results and Discussion

All eight experimental combinations were evaluated on the untouched test partition (`1,032` samples: 904 Ham, 128 Spam).

### 13.1 Comprehensive Test Set Performance Table:
| Exp # | Feature Extraction | Algorithm | Accuracy (%) | Precision (%) | Recall (%) | $F_1$-Score (%) | False Positives (FP) | Missed Spam (FN) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | **TF-IDF** | Multinomial NB | 97.67% | 94.07% | 86.72% | 90.24% | 7 / 904 | 17 / 128 |
| **E2** | **TF-IDF** | Bernoulli NB *(Binary)* | 98.06% | **97.37%** | 86.72% | 91.74% | **3 / 904** | 17 / 128 |
| **E3** | **TF-IDF** | Logistic Regression | 97.97% | 96.52% | 86.72% | 91.36% | 4 / 904 | 17 / 128 |
| **E4** | **TF-IDF** | Linear SVM *(Top Overall)* | **98.16%** | 96.58% | **88.28%** | **92.24%** | 4 / 904 | **15 / 128** |
| **E5** | **CountVectorizer** | Multinomial NB | 97.77% | 92.68% | **89.06%** | 90.84% | 9 / 904 | **14 / 128** |
| **E6** | **CountVectorizer** | Bernoulli NB *(Binary)* | 98.06% | **97.37%** | 86.72% | 91.74% | **3 / 904** | 17 / 128 |
| **E7** | **CountVectorizer** | Logistic Regression | 97.87% | 97.32% | 85.16% | 90.83% | **3 / 904** | 19 / 128 |
| **E8** | **CountVectorizer** | Linear SVM | **98.16%** | **97.39%** | 87.50% | 92.18% | **3 / 904** | 16 / 128 |

### 13.2 Key Findings:
1. **Best Model Overall:** **Linear SVM with TF-IDF** achieved the highest accuracy (**98.16%**) and highest $F_1$-score (**92.24%**), correctly intercepting 113 out of 128 spam messages with only 4 false alarms.
2. **Feature Extraction Contrast:**
   - **Linear SVM and Logistic Regression** perform better with **TF-IDF**, as sublinear scaling and inverse document frequencies dampen ubiquitous vocabulary noise while accentuating high-signal phrases (*"free entry"*, *"cash prize"*).
   - **Multinomial Naive Bayes** achieved higher recall (**89.06%** vs. 86.72%) under **CountVectorizer**, aligning with its discrete multinomial count assumptions.
   - **Bernoulli Naive Bayes** produced identical performance under both vectorizers (**98.06% Accuracy, 91.74% $F_1$**), confirming that binarizing presence/absence produces consistent results regardless of the underlying extraction tool.
3. **Operational Specificity:** In real telecommunication deployments, False Positives are severely penalized. **Bernoulli NB** and **CountVec + Linear SVM** produced only **3 false alarms out of 904 legitimate messages** (99.67% specificity).

---

## 14. Comparison with IEEE Research

| Dimension | IEEE TENCON 2023 Study (*de Luna et al.*) | Our Experimental Project | Academic Analysis |
|:---|:---|:---|:---|
| **Dataset Source** | Merged (UCI + Philippine + Student) | Authentic UCI SMS Collection | Distinct regional slang & vocabulary |
| **Class Distribution** | Artificially balanced 60:40 partition | Natural 87.6:12.4 class imbalance | Forced balance inflates recall artificially |
| **Feature Extraction** | CountVectorizer | Full Factorial (TF-IDF vs CountVec) | Systematic ablation across both methods |
| **Top Performing Model** | Bernoulli Naive Bayes (96.63%) | Linear SVM with TF-IDF (98.16%) | SVM margin maximization excels on TF-IDF |
| **Table VII Top 4 Models** | BNB (96.63%), RFC (96.03%), LR (95.87%), ETC (95.81%) | Lin. SVM (98.16%), BNB (98.06%), LR (97.97%), MNB (97.77%) | Corrected: RFC is 96.03%, not MNB |

---

## 15. Limitations

1. **Non-Direct Equivalence:** This comparison constitutes an external benchmark validation rather than a direct ablation due to differences in training corpora and artificial balancing.
2. **English Language Focus:** The UCI dataset comprises predominantly English text; multi-lingual or code-mixed text (e.g., Taglish or Hinglish) requires specialized tokenization.
3. **Adversarial SMS Evasion:** Sophisticated spam tactics (zero-width spaces, leetspeak, homoglyphs) may degrade token-based representations without character-level defenses.

---

## 16. Conclusion

This empirical research demonstrated that traditional machine learning algorithms, when paired with rigorous text preprocessing and sublinear TF-IDF vectorization, provide high detection accuracy and low false-alarm rates for SMS spam detection. Linear SVM achieved the highest overall performance ($98.16\%$ Accuracy, $92.24\%$ $F_1$), while Bernoulli Naive Bayes exhibited exceptional precision ($97.37\%$). Comparing our findings with the IEEE TENCON 2023 study demonstrates the critical influence of class balancing and feature representation on model evaluation.

---

## 17. Future Scope

1. **Lightweight Deep Learning & Transformers:** Benchmark DistilBERT and MobileBERT against Linear SVM to assess inference latency vs. accuracy trade-offs on mobile devices.
2. **Adversarial Defenses:** Integrate character-level convolutional embeddings to defend against leetspeak and obfuscated hyperlinks.
3. **Continuous Concept Drift Learning:** Implement online incremental learning algorithms to adapt to seasonal and evolving spam campaigns.

---

## 18. References

1. R. G. de Luna, K. L. Enriquez, A. M. Española, et al., "A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)," in *Proc. 2023 IEEE Region 10 Conference (TENCON)*, Chiang Mai, Thailand, 2023, pp. 52–57.
2. T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," in *Proc. 11th ACM Symposium on Document Engineering (DocEng)*, 2011, pp. 259–262.
3. A. Mukerjee, "Spam Filtering Using Bag-of-Words," *The Startup*, 2020.
4. N. Krishnaveni and V. Radha, "Comparison of Naïve Bayes and SVM Classifier for Detection of Spam SMS using Natural Language Processing," *Int. J. Semantic Comput.*, vol. 11, no. 2, pp. 2260–2265, 2021.
5. N. Kudupudi and S. Nair, "Spam message detection using logistic regression," *Int. J. Adv. Comput. Sci. Appl.*, vol. 9, no. 9, pp. 815–818, 2021.
6. S. Yerima and A. Bashar, "Semi-supervised novelty detection with one class SVM for SMS spam detection," in *Proc. 2022 29th Int. Conf. on Systems, Signals and Image Processing (IWSSIP)*, 2022, pp. 1–4.
7. G. Sonowal, "Detecting Phishing SMS Based on Multiple Correlation Algorithms," *SN Comput. Sci.*, vol. 1, no. 6, p. 361, 2020.
8. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
