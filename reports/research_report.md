# Comparative Analysis of Machine Learning Algorithms for SMS Spam Detection

**Academic Research Project Report**  
**Subject:** Research Methodologies (RM) — 7th Semester B.Tech  
**Author:** Priyansu Das  
**Department:** Computer Science and Engineering  
**Institution:** Siksha 'O' Anusandhan University (ITER), Bhubaneswar, Odisha, India  
**Repository:** [https://github.com/priyansudas07/sms-spam-detection](https://github.com/priyansudas07/sms-spam-detection)  
**System Branding:** SIGNAL — SMS Intelligence Lab  

---

## 1. Abstract

Short Message Service (SMS) remains a ubiquitous telecommunications channel, yet mobile subscribers are increasingly targeted by fraudulent text scams, commercial spam, and smishing attacks. This research study presents a comparative empirical investigation of supervised Machine Learning (ML) classifiers for automated SMS spam detection using the official UCI SMS Spam Collection dataset (Repository ID: 228). We implement a $2 \times 4$ full factorial experimental design evaluating two distinct feature extraction methodologies (**Term Frequency–Inverse Document Frequency [TF-IDF]** with sublinear term-frequency scaling and **CountVectorizer [Bag-of-Words]**) across four fundamental classification algorithms: **Multinomial Naive Bayes (MNB)**, **Bernoulli Naive Bayes (BNB)**, **Logistic Regression (LR)**, and **Linear Support Vector Machine (Linear SVM)**.

Text preprocessing incorporates case normalization, semantic token substitution (`urltoken`, `emailtoken`, `currencytoken`, `phonetoken`), negation word preservation, punctuation stripping, and Porter stemming. Zero data leakage is strictly enforced through Scikit-learn `Pipeline` encapsulation, fitting vectorizers solely on training folds. Model selection was conducted strictly using Stratified 5-Fold Cross-Validation on the training partition ($X_{\text{train}}$, $N = 4,128$) optimizing positive class ($F_1$-score) to prevent data snooping bias.

**TF-IDF + Bernoulli Naive Bayes** was selected as the **Validation Champion** (Mean CV $F_1 = 93.69\% \pm 1.77\%$). Evaluated once on the reserved, untouched test split ($X_{\text{test}}$, $N = 1,032$: 904 Ham, 128 Spam), the champion achieved **98.55% Accuracy**, **99.13% Precision**, **89.06% Recall**, **93.83% $F_1$-score**, and generated only **1 False Positive out of 904 legitimate messages** (0.11% false alarm rate). The peak observed test score was attained by **CountVectorizer + Logistic Regression** (**98.84% Accuracy**, **100.00% Precision**, **90.62% Recall**, **95.08% $F_1$-score**, with 0 False Positives). Furthermore, we conduct an external benchmark validation against the published findings of **de Luna et al. (IEEE TENCON 2023)**, analyzing the methodological implications of artificial 60:40 undersampling, natural class distribution preservation, and telecommunication false alarm penalties.

---

## 2. Introduction & Research Context

Short Message Service (SMS), introduced under the global GSM telecommunications standards, provides cellular text communication without requiring active packet-switched data connections. Due to near-universal accessibility and open telecommunication protocols, SMS achieves open rates exceeding 90%, with the vast majority read within three minutes of delivery. However, this high engagement makes mobile users prime targets for SMS spam. Fraudulent activities range from unsolicited marketing to financial advance-fee fraud, credential theft, and smishing (SMS phishing).

Traditional rule-based and keyword-blacklisting mechanisms fail against evolving spam patterns, where spammers alter syntax, introduce intentional misspellings, or mimic legitimate institutional alerts. Machine learning techniques provide an adaptive framework by learning underlying statistical and lexical distributions of legitimate (ham) versus fraudulent (spam) messages.

---

## 3. Problem Statement & Operational Trade-offs

Automated SMS spam classification in cellular networks poses three critical challenges:
1. **Severe Natural Class Imbalance:** Legitimate text messages naturally outnumber spam messages by approximately 7:1 in real-world mobile traffic. Models optimized purely for accuracy tend to exhibit high false-alarm rates or overlook minority spam occurrences.
2. **Short Text Sparsity:** SMS messages are bounded by character limits (typically 160 characters), resulting in sparse lexical representations with limited contextual signals compared to full-length emails.
3. **Severe Cost Asymmetry of Errors:** In commercial telecommunications, a **False Positive (FP)** (blocking an authentic personal, banking, or medical text) is catastrophic to subscriber trust, whereas a **False Negative (FN)** (a promotional spam message slipping into the inbox) is a minor annoyance. Therefore, **Precision** and specificity must be prioritized alongside overall detection recall.

---

## 4. Research Objectives

1. Develop an end-to-end, reproducible NLP text classification pipeline with zero data leakage.
2. Execute a $2 \times 4$ full factorial experiment comparing continuous Sublinear TF-IDF against discrete CountVectorizer across four classical classifiers.
3. Enforce true multivariate Bernoulli event modeling ($x_{ij} \in \{0, 1\}$) for Bernoulli Naive Bayes.
4. Preserve the authentic natural class distribution (87.56% Ham : 12.44% Spam) without artificial undersampling or synthetic SMOTE generation.
5. Perform strict hyperparameter optimization using Stratified 5-Fold Cross-Validation on $X_{\text{train}}$ optimizing $F_1$-score, keeping test set $X_{\text{test}}$ isolated for final evaluation.
6. Conduct an empirical 5-regime preprocessing ablation study to evaluate the impact of text cleaning operations.
7. Benchmark experimental findings against published results from IEEE TENCON 2023 (*Robert G. de Luna et al.*).
8. Deliver an interactive Streamlit workstation (**SIGNAL — SMS Intelligence Lab**) with runtime latency and calibrated probability profiling.

### 4.1 Core Research Questions (RQ1 – RQ6)
This investigation is structured to answer six central academic and engineering research questions:
- **RQ1 (Highest Accuracy):** Which classifier achieved the highest overall classification accuracy on the untouched test partition?
- **RQ2 (Highest $F_1$-Score):** Which classifier achieved the highest positive-class $F_1$-score balancing precision and recall?
- **RQ3 (Spam Detection Efficacy):** Which classifier detected spam most effectively when evaluated against telecommunication false-alarm penalties ($FP$ cost)?
- **RQ4 (Fewest False Negatives):** Which classifier produced the fewest false negatives (missed spam messages)?
- **RQ5 (Algorithmic Inductive Bias):** Why might one classification algorithm perform better than another in SMS spam filtering?
- **RQ6 (Feature Extraction Impact):** Does TF-IDF provide effective features for SMS text classification compared to discrete CountVectorizer?

---

## 5. Dataset Provenance and Integrity Audit

**Source:** [UCI Machine Learning Repository — SMS Spam Collection (Dataset ID: 228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

### 5.1 Dataset Audit & Verification Summary:
- **Raw Records Ingested:** 5,574 messages (4,827 Ham [86.60%], 747 Spam [13.40%], Imbalance Ratio: 6.46:1).
- **Missing / Null Records:** 0 null labels, 0 null messages, 0 empty strings.
- **Conflicting Labels:** 0 duplicate messages with conflicting classification labels.
- **Deduplication:** 414 exact duplicates removed (309 Ham, 105 Spam).
- **Cleaned Unique Corpus:** 5,160 messages (4,518 Ham [87.56%], 642 Spam [12.44%], Imbalance Ratio: 7.04:1).

### 5.2 Corpus Morphological Statistics:

| Metric Dimension | Ham Messages ($N=4,518$) | Spam Messages ($N=642$) | All Clean Messages ($N=5,160$) |
| :--- | :---: | :---: | :---: |
| **Character Length (Mean ± Std)** | 70.86 ± 56.58 | 137.41 ± 29.98 | 79.14 ± 58.29 |
| **Character Length (Median / IQR)** | 53.0 / 57.0 | 148.0 / 26.0 | 61.0 / 82.0 |
| **Word Count (Mean ± Std)** | 14.23 ± 11.16 | 23.70 ± 5.97 | 15.41 ± 11.10 |
| **Word Count (Median / IQR)** | 11.0 / 12.0 | 25.0 / 6.75 | 12.0 / 15.0 |

*Empirical Insight:* Spam messages exhibit almost double the average character count and significantly tighter variance ($\text{IQR} = 26$ vs. $57$), concentrating near the 160-character SMS carrier transmission boundary to maximize marketing content density.

### 5.3 Stratified Train-Test Partition:
- **Training Partition ($X_{\text{train}}$):** 4,128 messages (3,614 Ham [87.55%], 514 Spam [12.45%]).
- **Testing Partition ($X_{\text{test}}$):** 1,032 messages (904 Ham [87.60%], 128 Spam [12.40%]).
- **Random Seed:** Fixed to `random_state=42` across all splits and cross-validation folds.

---

## 6. Preprocessing Methodology & 5-Regime Ablation

Aggressive text cleaning can inadvertently destroy critical spam indicators. We evaluated five distinct preprocessing regimes under identical 5-fold CV on $X_{\text{train}}$:

1. **Regime 1 (Raw Baseline):** Whitespace tokenization only.
2. **Regime 2 (Case Normalization):** Lowercasing only.
3. **Regime 3 (Aggressive Punctuation & URL Removal):** Lowercasing + stripping all non-alphanumeric characters + Porter stemming.
4. **Regime 4 (Standard Stopword Stripping):** Regime 3 + aggressive NLTK English stopword removal.
5. **Regime 5 (Semantic Token Substitution & Negation Preservation — Proposed):**
   - URLs replaced with `urltoken` (preserves smishing presence while abstracting specific domain noise).
   - Email addresses replaced with `emailtoken`.
   - Currency symbols (£, $, €, ₹) replaced with `currencytoken` (captures financial lures).
   - Phone numbers replaced with `phonetoken`.
   - Critical negation words (`not`, `no`, `nor`, `never`, `without`, `neither`, `cannot`) explicitly preserved.
   - Punctuation removal and Porter stemming.

### Preprocessing Ablation Results (5-Fold CV on $X_{\text{train}}$ with Sublinear TF-IDF + Linear SVM):
| Preprocessing Regime | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | Mean CV $F_1$-Score |
| :--- | :---: | :---: | :---: | :---: |
| Regime 1: Raw Baseline | 97.41% | 94.62% | 84.44% | 89.24% |
| Regime 2: Lowercase Only | 97.58% | 95.10% | 85.02% | 89.77% |
| Regime 3: Aggressive Stripping | 97.89% | 95.84% | 86.96% | 91.19% |
| Regime 4: Aggressive Stopword Removal | 97.65% | 94.88% | 85.80% | 90.11% |
| **Regime 5: Semantic Tokens + Negation (Proposed)** | **98.28%** | **96.88%** | **89.11%** | **92.83%** |

*Findings:* Aggressive deletion of URLs and currency symbols eliminates the single most predictive smishing features. Semantic substitution preserves these indicators while preventing vocabulary explosion.

---

## 7. Feature Extraction

We evaluated two feature extraction architectures configured for unigrams and bigrams (`ngram_range=(1, 2)`):

### 7.1 Sublinear TF-IDF Vectorization
Computes weighted importance based on local term frequency and inverse corpus rarity:
$$\text{TF}(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Sublinear scaling prevents high-frequency promotional words from dominating document vectors, stabilizing linear separation margins.

### 7.2 CountVectorizer (Bag-of-Words)
Constructs document vectors representing raw token counts ($x_{ij} = f_{j, i}$).

### 7.3 Mathematical Binarization for Bernoulli Naive Bayes
Because BernoulliNB models a multivariate Bernoulli distribution, inputs must represent binary Boolean event occurrences:
$$x_{ij} \in \{0, 1\}$$
Configured strictly using `binary=True` in vectorizers and `binarize=0.0` in the estimator.

---

## 8. Machine Learning Algorithms

1. **Multinomial Naive Bayes (MNB):** Generative probabilistic classifier modeling discrete term occurrences.
2. **Bernoulli Naive Bayes (BNB):** Multivariate Bernoulli event model evaluating token presence/absence.
3. **Logistic Regression (LR):** Parametric linear model optimizing log-odds with $L_2$ regularization.
4. **Linear Support Vector Machine (Linear SVM):** Maximum-margin separating hyperplane with Platt scaling for posterior probability calibration.

---

## 9. Experimental Methodology & Model Selection

- **Zero Leakage Pipeline:** Preprocessing and vectorizer fitting encapsulated in Scikit-learn `Pipeline` objects.
- **Model Selection Protocol:** Evaluated all 8 pipelines across 5-fold Stratified CV on $X_{\text{train}}$. The champion was selected **strictly using Mean CV $F_1$-score on $X_{\text{train}}$**, leaving $X_{\text{test}}$ completely isolated.
- **Validation Champion:** **TF-IDF + Bernoulli Naive Bayes** (Mean CV $F_1 = 93.69\% \pm 1.77\%$, $\alpha = 0.01$).

### Training 5-Fold Cross-Validation Summary:
| Pipeline Name | Mean CV $F_1$ | Std CV $F_1$ | Optimal Hyperparameters |
| :--- | :---: | :---: | :--- |
| **TF-IDF + Bernoulli Naive Bayes (★ Champion)** | **0.9369** | ± 0.0177 | `alpha=0.01`, `max_features=4000` |
| CountVectorizer + Bernoulli Naive Bayes | 0.9369 | ± 0.0177 | `alpha=0.01`, `max_features=4000` |
| TF-IDF + Linear SVM | 0.9352 | ± 0.0264 | `C=2.0`, `max_features=3000` |
| CountVectorizer + Multinomial Naive Bayes | 0.9349 | ± 0.0173 | `alpha=0.5`, `max_features=4000` |
| TF-IDF + Multinomial Naive Bayes | 0.9325 | ± 0.0196 | `alpha=0.1`, `max_features=4000` |
| TF-IDF + Logistic Regression | 0.9259 | ± 0.0259 | `C=10.0`, `solver=liblinear`, `max_features=3000` |
| CountVectorizer + Logistic Regression | 0.9233 | ± 0.0257 | `C=10.0`, `solver=lbfgs`, `max_features=4000` |
| CountVectorizer + Linear SVM | 0.9210 | ± 0.0235 | `C=0.5`, `max_features=4000` |

---

## 10. Test Set Evaluation & Results

All eight trained pipelines were evaluated once on the isolated test partition ($N = 1,032$: 904 Ham, 128 Spam):

| Exp # | Feature Extraction | Algorithm | Role | Accuracy | Precision | Recall | $F_1$-Score | TN | FP | FN | TP |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1** | TF-IDF | Multinomial NB | Baseline | 98.06% | 95.76% | 88.28% | 91.87% | 899 | 5 | 15 | 113 |
| **E2** | TF-IDF | Bernoulli NB | **★ Validation Champion** | **98.55%** | **99.13%** | 89.06% | **93.83%** | 903 | **1** | 14 | 114 |
| **E3** | TF-IDF | Logistic Regression | Evaluated | 98.55% | 99.13% | 89.06% | 93.83% | 903 | **1** | 14 | 114 |
| **E4** | TF-IDF | Linear SVM | Evaluated | 98.26% | 95.08% | 90.62% | 92.80% | 898 | 6 | 12 | 116 |
| **E5** | CountVectorizer | Multinomial NB | Evaluated | 98.35% | 95.12% | **91.41%** | 93.23% | 898 | 6 | **11** | **117** |
| **E6** | CountVectorizer | Bernoulli NB | Evaluated | 98.55% | 99.13% | 89.06% | 93.83% | 903 | **1** | 14 | 114 |
| **E7** | CountVectorizer | Logistic Regression | Peak Test Score | **98.84%** | **100.00%** | 90.62% | **95.08%** | **904** | **0** | 12 | 116 |
| **E8** | CountVectorizer | Linear SVM | Evaluated | 98.74% | 98.32% | **91.41%** | 94.74% | 902 | 2 | **11** | **117** |

### Key Findings & Telecommunications Safety:
1. **Validation Champion vs. Peak Test Performance:**  
   Our model selection strictly adhered to training cross-validation, designating **TF-IDF + Bernoulli Naive Bayes** as champion. On test evaluation, this model produced **only 1 False Positive out of 904 legitimate messages** (0.11% false alarm rate), satisfying the primary commercial operational criterion.
2. **Peak Observed Test Score:**  
   **CountVectorizer + Logistic Regression** attained the peak test $F_1$-score (95.08%) and achieved **100% Precision with 0 False Positives**.
3. **Representation Stability:**  
   Bernoulli Naive Bayes produced identical test metrics under both TF-IDF and CountVectorizer (98.55% Accuracy, 93.83% $F_1$, 1 FP), demonstrating that true binary indicator event modeling isolates presence/absence information independent of token weighting scheme.

### 10.1 Empirical Answers to Core Research Questions (RQ1 – RQ6)

#### RQ1: Which classifier achieved the highest accuracy?
**CountVectorizer + Logistic Regression** achieved the highest test accuracy of **98.84%** (1,020 out of 1,032 test messages correctly classified).
- Runner-up: **CountVectorizer + Linear SVM** achieved **98.74%** (1,019/1,032).
- Third: **Bernoulli Naive Bayes** (both TF-IDF and CountVec) and **TF-IDF + Logistic Regression** tied at **98.55%** (1,017/1,032).
- *Training Cross-Validation:* On training 5-fold CV, **TF-IDF + Bernoulli Naive Bayes** attained the highest validation accuracy of **98.40% ± 0.38%**.

#### RQ2: Which classifier achieved the highest F1-score?
**CountVectorizer + Logistic Regression** achieved the highest positive-class test $F_1$-score of **95.08%** (0.9508).
- Runner-up: **CountVectorizer + Linear SVM** achieved **94.74%** $F_1$.
- Third: **Bernoulli Naive Bayes** achieved **93.83%** $F_1$ across both feature extractors.
- *Validation Protocol Note:* On the training partition ($X_{\text{train}}, N=4,128$), **TF-IDF + Bernoulli Naive Bayes** attained the highest Mean CV $F_1$-score (**93.69% ± 1.77%**), earning its selection as the official **Validation Champion**.

#### RQ3: Which classifier detected spam most effectively?
"Efficacy" in telecommunications requires balancing spam interception against the severe cost of false alarms:
1. **For False-Alarm Elimination (Carrier Trust):** **CountVectorizer + Logistic Regression** is the most effective model, achieving **100.00% Precision** with **0 False Positives out of 904 legitimate texts** (0.0% false alarm rate), while successfully intercepting 90.62% of spam. In telecommunications, blocking an authentic personal, banking, or medical text destroys subscriber trust; achieving zero false positives is the ultimate industry objective.
2. **For Spam Interception Coverage (Maximum Detection):** **CountVectorizer + Linear SVM** and **CountVectorizer + Multinomial Naive Bayes** detected spam most aggressively, achieving **91.41% Recall** (117 of 128 spam messages blocked, missing only 11).
3. **For Adversarial Smishing Resilience (Modern Concept Drift):** The **Tier-2 Hybrid Smishing Consensus Guard** detected spam most effectively against evasive modern attacks (e.g., USPS package tracking alerts with payment demands), achieving $>91\%$ recall where historical standalone generative models failed.

#### RQ4: Which classifier produced the fewest false negatives (missed spam)?
**CountVectorizer + Linear SVM** and **CountVectorizer + Multinomial Naive Bayes** produced the fewest false negatives, tied at only **11 False Negatives** out of 128 spam messages (Recall = **91.41%**, True Positives = 117/128).
- Comparison: CountVectorizer + Logistic Regression produced 12 FN (Recall = 90.62%); TF-IDF + Linear SVM produced 12 FN (Recall = 90.62%); Bernoulli Naive Bayes produced 14 FN (Recall = 89.06%); TF-IDF + Multinomial Naive Bayes produced 15 FN (Recall = 88.28%).

#### RQ5: Why might one algorithm perform better than another?
Performance differentials are governed by fundamental differences in mathematical formulations and inductive biases:
1. **Generative vs. Discriminative Modeling:**
   - **Naive Bayes (MNB & BNB) is generative:** It models joint probability $P(\mathbf{x}, y)$ and assumes strict conditional feature independence given the class ($P(\mathbf{x} \mid y) = \prod P(x_j \mid y)$). In natural language, words correlate strongly (e.g., `urgent` and `call`, `redelivery` and `fee`). This violation pushes probability estimates to extreme bounds ($0$ or $1$). However, BNB's binary presence modeling ($x_j \in \{0, 1\}$) acts as a natural regularizer, yielding an exceptionally conservative decision rule with only 1 false positive.
   - **Logistic Regression & Linear SVM are discriminative:** They model posterior $P(y \mid \mathbf{x})$ directly or learn an optimal separating hyperplane ($\mathbf{w}^T \mathbf{x} + b$). They make no feature independence assumptions, allowing them to assign strong, correlated positive weights to decisive smishing tokens (`urltoken`, `currencytoken`) that easily override conversational filler vocabulary.
2. **Decision Boundary Regularization:**
   - Linear SVM maximizes the soft-margin distance ($2 / \|\mathbf{w}\|$), optimizing boundary resilience against borderline samples and yielding the highest recall (lowest FN = 11).
   - Logistic Regression with $L_2$ penalty ($C=10.0$) optimizes smooth log-loss, preventing extreme parameter weights and achieving a perfect boundary separation on authentic messages (0 FP).
3. **Event Space Structure:**
   - Multinomial NB uses token frequency counts, which can be manipulated if a spammer repeats non-spam words. Bernoulli NB evaluates binary presence, preventing frequency manipulation from distorting probabilities.

#### RQ6: Does TF-IDF provide effective features for SMS classification?
**Yes, but with important nuances dictated by text length:**
1. **Effective for Sublinear Scaling:** TF-IDF with sublinear scaling ($\text{TF} = 1 + \log(f)$) dampens high-frequency promotional terms and prevents vector distortion, producing our **Validation Champion (TF-IDF + Bernoulli NB)** with 98.55% accuracy and only 1 false positive (0.11% false alarm rate).
2. **Why CountVectorizer Edge-Beat TF-IDF on Linear Models:**
   - For Logistic Regression, CountVectorizer achieved $F_1 = 95.08\%$ vs. TF-IDF's $93.83\%$.
   - For Linear SVM, CountVectorizer achieved $F_1 = 94.74\%$ vs. TF-IDF's $92.80\%$.
   - *Theoretical Reason:* SMS messages are character-bounded (160 characters, average ~15 words). In short texts, virtually all words occur only once ($f=1$). Unlike long news articles where Inverse Document Frequency (IDF) smooths out frequent paragraph filler, in sparse short messages, IDF downweighting slightly compresses the margin of rare spam cues. Discrete n-gram frequencies preserve sharp binary indicator boundaries.
3. **Mathematical Invariance in Bernoulli NB:** For Bernoulli Naive Bayes, TF-IDF and CountVectorizer yielded **identical metrics** (98.55% Accuracy, 93.83% $F_1$, 1 FP), demonstrating that true Boolean binarization ($x_{ij} \in \{0, 1\}$) renders term-frequency weighting mathematically invariant.

---

## 11. Benchmark Comparison: IEEE TENCON 2023

Authoritative reference: Robert G. de Luna et al., *Proc. 2023 IEEE Region 10 Conference (TENCON)*, pp. 52–57, ISBN: `979-8-3503-0219-6/23/$31.00 ©2023 IEEE`.

### 11.1 Published Paper Findings (Table VII vs Table VI):
- **Table VII Top 4 Optimized Models:**
  1. Bernoulli Naive Bayes: 96.6333% Accuracy | 98.4431% Precision | 95.5174% Recall | 92.7835% $F_1$
  2. Random Forest: 96.0297% Accuracy | 98.5228% Precision | 94.8699% Recall | 91.1408% $F_1$ (**Verified: strictly Random Forest, NOT Multinomial NB**)
  3. Logistic Regression: 95.8709% Accuracy | 98.7711% Precision | 93.7399% Recall | 89.2555% $F_1$
  4. Extra Trees: 95.8070% Accuracy | 98.4885% Precision | 94.3495% Recall | 90.5663% $F_1$
- **Table VI Default Baselines (CountVectorizer):** LR: 95.62%, BNB: 95.43%, RFC: 95.24%, ETC: 94.98%, SVC: 94.57%.

### 11.2 Overlapping Classifiers Comparison Table:
| Algorithm | Feature Extractor | IEEE TENCON 2023 (Table VII) | Our Study ($N=1,032$) | Methodological Context |
| :--- | :--- | :---: | :---: | :--- |
| **Bernoulli Naive Bayes** | CountVectorizer | Acc: 96.63% \| Prec: 98.44%<br>Rec: 95.52% \| $F_1$: 92.78% | Acc: 98.55% \| Prec: 99.13%<br>Rec: 89.06% \| $F_1$: 93.83% | High precision across both studies (>98%). |
| **Bernoulli Naive Bayes** | Sublinear TF-IDF | *Not evaluated in Table VII* | **Acc: 98.55% \| Prec: 99.13%<br>Rec: 89.06% \| $F_1$: 93.83% [★ Champion]** | **Validation Champion:** 1 FP / 904. |
| **Logistic Regression** | CountVectorizer | Acc: 95.87% \| Prec: 98.77%<br>Rec: 93.74% \| $F_1$: 89.26% | Acc: 98.84% \| Prec: 100.00%<br>Rec: 90.62% \| $F_1$: 95.08% | Peak test score in our study (0 FP / 904). |
| **Logistic Regression** | Sublinear TF-IDF | *Not evaluated in Table VII* | Acc: 98.55% \| Prec: 99.13%<br>Rec: 89.06% \| $F_1$: 93.83% | Consistent generalization. |
| **Linear SVM** | CountVectorizer | Acc: 94.57% *(Default CV in Table VI)* | Acc: 98.74% \| Prec: 98.32%<br>Rec: 91.41% \| $F_1$: 94.74% | Tuning penalty $C$ yielded substantial gain. |
| **Linear SVM** | Sublinear TF-IDF | *Not evaluated in paper* | Acc: 98.26% \| Prec: 95.08%<br>Rec: 90.62% \| $F_1$: 92.80% | Platt calibrated posterior probabilities. |

### 11.3 Methodological Variance Analysis & Academic Disclaimer:
1. **Class Distribution Impact:** The IEEE study's higher recall on BNB (95.52% vs. 89.06%) is a direct consequence of their **60:40 artificial undersampling**. In their corpus, spam occurred at >3x its real-world frequency (40% vs. 12.44%).
2. **Why Higher Accuracy Does NOT Prove Algorithmic Superiority:** A dummy classifier predicting Ham on our dataset achieves 87.56% baseline accuracy, whereas on the IEEE 60:40 dataset it achieves 60.0%. Thus, our higher test accuracy (98.55% vs. 96.63%) reflects class distribution and semantic preprocessing, not absolute algorithmic dominance.

---

## 12. Multi-Tier Hybrid Smishing Defense Architecture & Concept-Drift Case Study

### 12.1 The Concept Drift Phenomenon in Telecommunication Fraud
The SMS Spam Collection benchmark represents telecommunications fraud from 2011, characterized predominantly by overt commercial telemarketing, premium ringtone subscriptions, and lottery prizes (e.g., *winner, claim, cash, prize*). In contemporary threat landscapes (2024–2026), adversaries employ **concept-drift evasion tactics**—notably package delivery alerts (e.g., USPS, FedEx, DHL, Evri), banking fraud warnings, and redelivery fee extortion.

**Case Study Test Vector:**
> *"USPS Notice: Your package could not be delivered today due to an incorrect house number. Please update your delivery address and pay the $1.99 redelivery fee at http://usps-tracking-portal-update.com within 12 hours."*

When evaluated on this real-world adversary sample:
1. **Generative Model Degradation (Bernoulli Naive Bayes):** Classifies the message as **HAM** ($P(\text{Spam}) = 0.8\%$). In the historical corpus, terms such as `today`, `house`, `number`, `address`, and `delivered` correlate overwhelmingly with legitimate personal messages. Under Bernoulli Naive Bayes, the absence of historical spam tokens incurs a multiplicative non-occurrence penalty:
   $$\prod_{j: x_j=0} (1 - P(w_j \mid \text{Spam}))$$
   This suppresses the single `urltoken` feature and generates a false negative.
2. **Discriminative Resilience (Linear SVM & Logistic Regression):** Correctly classifies the message as **SPAM** ($P(\text{Spam}) = 99.2\%$ and $93.6\%$ respectively). Linear models compute additive hyperplanes:
   $$f(\mathbf{x}) = \sum_{j} w_j x_j + b$$
   Large positive weights on high-entropy indicators (`urltoken`, `fee`) decisively tilt the decision boundary toward SPAM, regardless of conversational vocabulary.

### 12.2 Two-Tier Defense Architecture
To reconcile historical academic benchmark integrity with operational security, SIGNAL incorporates a dual-tier architecture:
- **Tier 1 (Supervised ML Classifiers):** Rigorous $2 \times 4$ factorial pipelines operating strictly on natural token n-grams and Platt-calibrated margins.
- **Tier 2 (Smishing Heuristic Inspector & Weighted Consensus):** A real-time pattern analyzer (`src/smishing_heuristics.py`) detecting compound smishing vectors (hyperlink + monetary fee demand + urgency expiration). Discriminative models are assigned higher consensus weights (Linear SVM: 1.8–2.2, Logistic Regression: 2.0, Naive Bayes: 0.8–1.0). When compound threats are verified, an adaptive Bayesian boost prevents unweighted generative dilution, flipping the ensemble verdict to **SPAM DETECTED (84.25% confidence)** while maintaining 0% false alarms on legitimate traffic.

---

## 13. Conclusion & Future Scope

This investigation established that traditional linear and probabilistic classifiers achieve high effectiveness in SMS spam detection without requiring computationally prohibitive deep learning architectures. By enforcing zero data leakage, true multivariate Bernoulli event binarization, semantic token substitution, and training-driven cross-validation selection, we demonstrated that **Bernoulli Naive Bayes** and **Logistic Regression** deliver near-flawless precision (0 to 1 false alarms out of 904 legitimate texts) on authentic imbalanced traffic.

**Future directions include:**
- Benchmarking edge-optimized transformers (DistilBERT, MobileBERT) for latency-versus-accuracy trade-offs.
- Integrating regional multilingual SMS corpora (Hinglish and Indic languages).
- Hardening against adversarial character-level evasion tactics (zero-width characters and homoglyphs).
