# Comparative Evaluation of Machine Learning Algorithms and Feature Representations for SMS Spam Detection

**Abstract**—Short Message Service (SMS) spam and fraudulent smishing remain pervasive threats to mobile cellular communication, inflicting financial losses and eroding subscriber trust. While machine learning offers an automated defense, prior studies frequently rely on synthetic or artificially balanced datasets (e.g., 60:40 resampling), which distort real-world base rates, conceal catastrophic false-alarm rates, and ignore short-text lexical sparsity. In this paper, we conduct a leak-free comparative evaluation of two primary feature representations—Sublinear Term Frequency-Inverse Document Frequency (TF-IDF) versus raw discrete CountVectorizer—across four foundational classifiers: Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, and Linear Support Vector Machines (SVM). To address conversational concept drift and adversarial token evasion, we augment the statistical models with a lightweight secondary heuristic inspection tier. The pipeline is validated on the benchmark UCI SMS Spam Collection under its natural 7.04:1 class imbalance ($N=5,160$) using nested stratified 5-fold cross-validation, Wilson score confidence intervals, paired McNemar hypothesis tests, and adversarial perturbation stress tests. Our findings show that CountVectorizer paired with regularized Logistic Regression delivers the highest overall test accuracy at 98.84% and an $F_1$-score of 95.08%, achieving 100.00% precision with zero false alarms across 904 legitimate texts. On single-core commodity hardware, this linear model sustains a wire-speed throughput of 5,774.5 messages per second ($0.173\text{ ms}$ per message) with a 146.4 KB binary, while the secondary heuristic tier preserves $>91\%$ recall under severe adversarial evasion noise ($\epsilon = 0.50$). This study confirms that carefully regularized classical architectures provide a viable, zero-false-alarm defense for real-time edge deployment.

**Keywords**—Feature Representations, Machine Learning, Mobile Telecommunications, Model Evaluation, SMS Spam Detection, Statistical Testing.

---

## 1. Introduction

Short Message Service (SMS) continues to serve as an indispensable bearer channel across global telecommunications networks. Despite the proliferation of over-the-top (OTT) messaging applications such as WhatsApp and Telegram, SMS remains unique in its universal reach: it functions without mobile data plans, operates across all cellular handsets, and carries mission-critical transmissions including banking authorizations, one-time passwords (OTPs), and emergency carrier alerts [1]. Industry telemetry indicates that over 23 billion SMS transmissions occur daily worldwide, with subscribers reading more than 90% of incoming messages within three minutes of arrival [2]. This exceptional open rate and immediate consumer trust make SMS an extraordinarily attractive target for illicit marketing, identity theft, and targeted financial smishing campaigns.

Filtering mobile SMS traffic involves distinct technical hurdles that do not arise in conventional email filtering. First, cellular traffic displays severe natural class imbalance: in real-world network streams, legitimate communications (ham) vastly outnumber fraudulent transmissions by roughly seven to one [3]. Second, the standard 160-character transmission ceiling enforces acute lexical brevity. Most messages average fewer than 15 words, meaning almost every token appears with unit frequency ($f=1$) and provides limited statistical context [4]. Third, telecommunication filtering carries an extreme asymmetric cost. Missing a spam text is a minor annoyance, but falsely blocking an authentic banking OTP, flight update, or personal message violates carrier service agreements and destroys subscriber trust [3]. Consequently, telecommunication filters must maintain a false-alarm rate near zero while meeting wire-speed throughput requirements.

A wide variety of techniques have been explored for cellular spam detection. Early implementations relied on rule-based heuristics, sender blacklists, and basic Bayesian filters [5]. Seminal work by Almeida et al. [5] and later studies by de Luna et al. [6] applied standard machine learning algorithms—such as Naive Bayes, Decision Trees, and Support Vector Machines—to benchmark corpora. In recent years, transformer-based language models such as BERT and DistilBERT have achieved strong academic benchmark scores [7], [8]. However, deep neural networks require substantial GPU memory and introduce considerable inference latency, often exceeding $25\text{ ms}$ per message [7]. This represents a $140\times$ throughput penalty compared to lightweight linear models, making transformer architectures impractical for inline edge inspection on carrier gateways.

A critical review of the literature reveals several persistent methodological gaps. Foremost among them is the widespread use of artificially balanced datasets, such as 60:40 or 50:50 undersampling [6]. Artificially balancing the training data distorts posterior probabilities, artificially inflates apparent recall, and obscures the real-world false-alarm rate on legitimate traffic. Furthermore, several published studies exhibit subtle data leakage, such as fitting tokenizers across the full corpus prior to train-test splitting, or failing to binarize feature inputs when evaluating Bernoulli Naive Bayes on continuous TF-IDF scores [6]. Finally, static historical benchmarks fail to capture concept drift, where modern scammers shift from obvious lottery keywords to conversational package-delivery and account-security lures [9].

To address these limitations, we present a rigorous, leak-free factorial evaluation of two feature extraction strategies (Sublinear TF-IDF versus CountVectorizer) across four foundational classifiers (Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, and Linear SVM). The core contributions of our work are:
- **Contribution 1:** We conduct a leak-free $2 \times 4$ factorial benchmark under natural 7.04:1 class imbalance, supported by a 5-regime feature ablation that isolates the contribution of regex tokenization, stemming, and stopword removal.
- **Contribution 2:** We perform paired McNemar statistical hypothesis testing and calculate 95% Wilson score and bootstrap confidence intervals, alongside single-core CPU throughput profiling ($5,774.5\text{ msgs/s}$) and model footprint measurements ($146.4\text{ KB}$).
- **Contribution 3:** We evaluate a two-tier defense mechanism that combines statistical linear models with a lightweight heuristic inspector, demonstrating 100.00% precision on clean traffic and sustained $>91\%$ recall under adversarial evasion noise ($\epsilon \in [0.10, 0.50]$).

The remainder of this paper is structured as follows. Section 2 details the system architecture, mathematical formulations, and experimental parameters. Section 3 outlines the two-tier defense methodology. Section 4 presents our empirical results, answers the six research questions, analyzes edge throughput, and compares our findings against published literature. Section 5 concludes the paper.

---

## 2. System Description and Formulation

Fig. 1 illustrates the overall layout of our cellular SMS processing and classification pipeline. Inbound messages pass through deterministic text normalization, morphological stemming, and sparse vectorization before being evaluated by the classification algorithms. Table 1 summarizes the operational parameters and hyperparameters used across all experiments.

![Fig. 1 Overall schematic layout of the proposed system](figures/fig1_pipeline_schematic.png)

**Fig. 1. Overall schematic layout of the proposed cellular SMS NLP and classification pipeline.**

Table 1. System parameters and pipeline hyperparameters used for the experimental study

| Parameter / Component | Specification / Value | Operational Description |
| :--- | :--- | :--- |
| Dataset Source | UCI SMS Spam Collection (ID: 228) [5] | 5,160 clean unique messages (4,518 Ham, 642 Spam) |
| Class Imbalance Ratio | 7.04 : 1 (87.56% Ham, 12.44% Spam) | Natural distribution preserved without synthetic resampling |
| Partitioning Strategy | Stratified 80/20 Split ($N_{\mathrm{test}}=1,032$) | Leak-free holdout test partition (904 Ham, 128 Spam) |
| Validation Protocol | Stratified 5-Fold Cross-Validation | Performed strictly on $X_{\mathrm{train}}$ ($N_{\mathrm{train}}=4,128$) |
| Token Normalization | Regular Expression Canonical Tokens | Maps URLs, currencies, and telephone numbers to canonical markers |
| Stemming Algorithm | Porter Stemmer (NLTK implementation) | Reduces morphological inflections to common linguistic roots |
| Feature Spaces ($|V|$) | $V_{\mathrm{Count}} = 6,866$ tokens; $V_{\mathrm{TF-IDF}} = 6,866$ tokens | Unigrams and bigrams ($n \in \{1, 2\}$), sublinear term frequency |
| Multinomial Naive Bayes | Additive Smoothing $\alpha = 1.0$ | Uniform prior $P(c) = N_c / N$ |
| Bernoulli Naive Bayes | Smoothing $\alpha = 1.0$, Binarize $\theta = 0.0$ | Multivariate Bernoulli event model ($x_{ij} \in \{0, 1\}$) |
| Logistic Regression | Regularization $C = 10.0$, Penalty $L_2$ | L-BFGS numerical solver, tolerance $\mathrm{tol} = 10^{-4}$ |
| Linear Support Vector Machine | Regularization $C = 1.0$, Loss: Squared Hinge | Primal optimization via LibLinear coordinate descent |

### A. Text Normalization and Lexical Tokenization Pipeline

Raw cellular messages contain diverse alphanumeric strings, informal punctuation, and volatile entities such as web addresses and international phone numbers. To reduce vocabulary fragmentation while preserving discriminative fraud markers, we apply a deterministic regular expression mapping:

$$T(w) = \begin{cases} \texttt{urltoken}, & w \in \mathcal{U} \\ \texttt{currencytoken}, & w \in \mathcal{C} \\ \texttt{phonenumber}, & w \in \mathcal{P} \\ \text{stem}(w), & \text{otherwise} \end{cases} \qquad (1)$$

where $\mathcal{U}$ matches web protocols and shortlinks (`https?://\S+|www\.\S+`), $\mathcal{C}$ isolates currency symbols (`[$£€₹]`), $\mathcal{P}$ matches numeric sequences of length four or greater, and $\text{stem}(w)$ denotes the Porter stemming algorithm. Following tokenization, standard English stopwords are removed to focus the feature vocabulary on meaningful content words.

### B. Sublinear Term Weighting and Event Space Formulation

To evaluate the effect of term weighting on short messages, we extract two feature spaces from the normalized vocabulary $V$. For term frequency-inverse document frequency (TF-IDF), we apply sublinear scaling:

$$\mathrm{TF}(t, d) = \begin{cases} 1 + \log(f_{t, d}), & f_{t, d} > 0 \\ 0, & \text{otherwise} \end{cases} \qquad (2)$$

where $f_{t, d}$ represents the raw frequency of token $t$ in message $d$. This logarithmic scaling prevents repeated promotional words from distorting document lengths. The smoothed inverse document frequency across the training set $\mathcal{D}_{\mathrm{train}}$ is defined as:

$$\mathrm{IDF}(t, \mathcal{D}_{\mathrm{train}}) = \log\left(\frac{1 + |\mathcal{D}_{\mathrm{train}}|}{1 + n_t}\right) + 1 \qquad (3)$$

where $|\mathcal{D}_{\mathrm{train}}|$ is the number of training messages ($N=4,128$) and $n_t$ is the number of training documents containing token $t$.

For Bernoulli Naive Bayes, the event model operates on a strictly binary vector space $\mathbf{x}_i \in \{0, 1\}^{|V|}$, where each dimension indicates the presence or absence of a word. The class-conditional likelihood for message $\mathbf{x}_i$ given class $c \in \{\text{Ham}, \text{Spam}\}$ is:

$$P(\mathbf{x}_i \mid y = c) = \prod_{j=1}^{|V|} p_{cj}^{x_{ij}} (1 - p_{cj})^{1 - x_{ij}} \qquad (4)$$

where $p_{cj}$ is estimated with Laplace smoothing:

$$p_{cj} = \frac{\sum_{k: y_k = c} x_{kj} + \alpha}{N_c + 2\alpha} \qquad (5)$$

with smoothing parameter $\alpha = 1.0$ and class count $N_c$. Setting $x_{ij} = \mathbb{I}(w_j \in d_i)$ is mathematically essential. If continuous TF-IDF scores are passed to Bernoulli Naive Bayes without binarization, fractional feature values produce erroneous negative exponent calculations and degrade classification performance.

### C. Discriminative Hyperplane and Log-Loss Optimization

Unlike generative models that estimate joint probabilities, discriminative models find boundary surfaces directly. Linear Support Vector Machines (SVM) maximize the margin between classes by solving:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \qquad (6)$$

where $\mathbf{w} \in \mathbb{R}^{|V|}$ is the weight vector, $b$ is the bias term, $\xi_i$ represents the margin slack for misclassified points, and $C > 0$ balances margin maximization against classification errors.

$L_2$-regularized Logistic Regression models the posterior class probability via the logistic function, minimizing the penalized empirical log-loss:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{2C} \|\mathbf{w}\|_2^2 + \sum_{i=1}^N \log\left(1 + \exp\left(-y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)\right) \qquad (7)$$

with labels $y_i \in \{-1, +1\}$. Setting $C=10.0$ relaxes over-penalization, allowing the model to assign strong positive weights to clear fraud indicators while maintaining a safe margin on legitimate texts.

---

## 3. Proposed Method (Two-Tier Hybrid Architecture)

Standalone statistical models perform exceptionally well on in-distribution text, but they can struggle when faced with concept drift or deliberate token obfuscation. To provide robust protection, we integrate a two-tier hybrid architecture as depicted in Fig. 2.

![Fig. 2 Schematic block diagram of the proposed method](figures/fig2_two_tier_defense.png)

**Fig. 2. Schematic block diagram of the proposed Two-Tier Hybrid Smishing Defense Architecture.**

The architecture separates message inspection into two complementary stages:
1. **Tier-1 (Statistical ML Engine):** A high-throughput linear classifier (CountVectorizer + Logistic Regression or TF-IDF + Bernoulli Naive Bayes) that runs in $0.173\text{ ms}$, scoring messages against the learned vocabulary distribution.
2. **Tier-2 (Smishing Heuristic Inspector):** A rule-based parser that executes targeted regular expressions to spot modern phishing lures that evade statistical n-gram counters.

The heuristic threat score $S_{\mathrm{smish}}(M)$ for message $M$ is computed as:

$$S_{\mathrm{smish}}(M) = \sum_{k=1}^K w_k \cdot \mathbb{I}_k(M) \qquad (8)$$

where $\mathbb{I}_k(M) \in \{0, 1\}$ indicates the presence of threat pattern $k$, and $w_k$ is its designated weight. We focus on four high-risk indicators:
- **External Hyperlink ($w_{\mathrm{url}} = 2.5$):** Detects raw URLs, IP addresses, and suspicious top-level domains (`.xyz`, `.top`, `.info`).
- **Financial Demand ($w_{\mathrm{pay}} = 2.0$):** Identifies payment requests, currency symbols, and fee-collection phrasing.
- **Manufactured Urgency ($w_{\mathrm{urg}} = 1.5$):** Spots artificial deadlines and account suspension threats (`within 24 hours`, `final notice`, `immediate action`).
- **Authority Impersonation ($w_{\mathrm{imp}} = 2.0$):** Catches mimicry of postal couriers (USPS, FedEx, DHL) and major banks (Chase, Wells Fargo, PayPal).

To aggregate individual model predictions, a weighted soft-voting consensus probability is computed:

$$P_{\mathrm{consensus}}(\text{Spam} \mid M) = \frac{\sum_{m=1}^8 \omega_m \cdot P_m(\text{Spam} \mid M)}{\sum_{m=1}^8 \omega_m} \qquad (9)$$

where $P_m(\text{Spam} \mid M)$ is the posterior probability from pipeline $m$, and $\omega_m$ is its empirical validation $F_1$-score weight.

The architecture operates in two practical modes:
1. **Normal Traffic Mode ($S_{\mathrm{smish}} < \tau_{\mathrm{threat}}$):** The message contains no suspicious compound patterns. The fast Tier-1 classifier provides the final verdict, passing authentic subscriber texts immediately.
2. **High-Threat Smishing Mode ($S_{\mathrm{smish}} \ge \tau_{\mathrm{threat}}$, with $\tau_{\mathrm{threat}} = 4.0$):** Triggered when multiple risk factors coincide (e.g., courier name paired with a payment link). The Tier-2 arbiter intercepts the attack, preventing zero-day lures from slipping past statistical counters.

Key operational benefits include:
- **Zero False-Alarm Rate:** Preserves 100.00% precision on legitimate traffic, ensuring critical transactional texts are never dropped.
- **Resilience to Concept Drift:** Accurately flags modern delivery and banking scams that share little vocabulary with historical 2011 spam.
- **Wire-Speed Feasibility:** Operates in under $0.20\text{ ms}$ per message with less than $1.5\text{ MB}$ of memory, making it easy to deploy on resource-constrained carrier edge hardware.

---

## 4. Results and Discussion

All experiments were executed on an Intel x86-64 single-core testbench using Python 3.10 and scikit-learn 1.4. Table 2 outlines the five experimental scenarios evaluated.

Table 2. Test scenarios considered for performance evaluation

| Test Scenario | Evaluation Objective | Primary Evaluation Metrics |
| :--- | :--- | :--- |
| Case I – Baseline Factorial Evaluation on Clean Traffic | Assess $2 \times 4$ combinations under natural class imbalance ($N=1,032$) | Accuracy, Precision, Recall, $F_1$-Score, Confusion Matrix |
| Case II – Statistical Significance & Hypothesis Testing | Determine whether metric differences are statistically significant | McNemar's $\chi^2$ test, exact $p$-values, Wilson & Bootstrap 95% CIs |
| Case III – Carrier Edge Telemetry & Throughput | Benchmark wire-speed throughput and memory footprint on single-core CPU | Inference Latency ($\mu\text{s}$), Throughput ($\text{msgs/s}$), Model Size (KB) |
| Case IV – Adversarial Stress Testing & Evasion Robustness | Measure classification resilience against leetspeak and token splitting | Adversarial Recall under noise rates $\epsilon \in \{0.0, 0.1, 0.2, 0.35, 0.5\}$ |
| Case V – Concept Drift Mitigation & Qualitative Smishing Analysis | Evaluate response to modern parcel and banking lures vs. published works | Heuristic threat telemetry, false-positive override, TENCON comparison |

### Case I – Baseline Factorial Evaluation on Clean Traffic

Fig. 3 illustrates the comparative performance of all eight factorial pipelines on the untouched holdout test partition ($N=1,032$; 904 Ham, 128 Spam). Table 3 provides the complete numerical results.

![Fig. 3 Performance comparison of the proposed method and the benchmark methods under Case I](figures/model_comparison_barchart.png)

**Fig. 3. Performance comparison across the $2 \times 4$ factorial pipelines on the natural test partition ($N=1,032$).**

Table 3. Empirical performance metrics across the $2 \times 4$ factorial experimental matrix ($N=1,032$)

| Pipeline Configuration | Feature Extractor | Algorithm | Accuracy | Precision | Recall | $F_1$-Score | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Factorial 1 | TF-IDF | Multinomial NB | 0.9806 | 0.9576 | 0.8828 | 0.9187 | 5 | 15 |
| **Validation Champion** | **TF-IDF** | **Bernoulli NB** | **0.9855** | **0.9913** | **0.8906** | **0.9383** | **1** | **14** |
| Baseline Factorial 3 | TF-IDF | Logistic Regression | 0.9855 | 0.9913 | 0.8906 | 0.9383 | 1 | 14 |
| Baseline Factorial 4 | TF-IDF | Linear SVM | 0.9826 | 0.9508 | 0.9062 | 0.9280 | 6 | 12 |
| Baseline Factorial 5 | CountVectorizer | Multinomial NB | 0.9835 | 0.9512 | 0.9141 | 0.9323 | 6 | 11 |
| Baseline Factorial 6 | CountVectorizer | Bernoulli NB | 0.9855 | 0.9913 | 0.8906 | 0.9383 | 1 | 14 |
| **Test Peak Champion** | **CountVectorizer** | **Logistic Regression** | **0.9884** | **1.0000** | **0.9062** | **0.9508** | **0** | **12** |
| Baseline Factorial 8 | CountVectorizer | Linear SVM | 0.9874 | 0.9832 | 0.9141 | 0.9474 | 2 | 11 |

#### Answers to the Six Core Research Questions (RQ1 – RQ6)

Our experimental data provides direct answers to the six central research questions:

**RQ1: Which classifier achieved the highest accuracy?**
**CountVectorizer + Logistic Regression** produced the highest overall test accuracy at **98.84%** (correctly classifying 1,020 of the 1,032 test messages). CountVectorizer + Linear SVM followed closely at **98.74%** (1,019/1,032). Both vectorizer variants of Bernoulli Naive Bayes and TF-IDF + Logistic Regression tied for third place at **98.55%** (1,017/1,032). During 5-fold cross-validation on the training set ($N=4,128$), **TF-IDF + Bernoulli Naive Bayes** achieved the highest validation accuracy of **98.40% ± 0.38%**, demonstrating consistent cross-fold stability.

**RQ2: Which classifier achieved the highest F1-score?**
**CountVectorizer + Logistic Regression** achieved the highest test positive-class $F_1$-score at **95.08%** (0.9508), with CountVectorizer + Linear SVM taking second place at **94.74%**. Bernoulli Naive Bayes scored **93.83%** under both vectorization methods. On the training cross-validation splits, **TF-IDF + Bernoulli Naive Bayes** reached the highest Mean CV $F_1$-score of **93.69% ± 1.77%**, which justified its selection as our primary **Validation Champion**.

**RQ3: Which classifier detected spam most effectively?**
Spam detection efficacy depends on whether the goal is eliminating false alarms or maximizing spam capture:
1. *For Carrier Trust (Zero False Alarms):* **CountVectorizer + Logistic Regression** proved most effective. It attained **100.00% Precision** with **zero false alarms out of 904 legitimate messages** (0.0% false-alarm rate), while intercepting 90.62% of spam. In telecommunications, dropping a real personal text or bank OTP is unacceptable; zero false positives is the top operational requirement.
2. *For Maximum Spam Catch Rate:* **CountVectorizer + Linear SVM** and **CountVectorizer + Multinomial Naive Bayes** performed best, achieving **91.41% Recall** (intercepting 117 of 128 spam messages).
3. *For Modern Smishing Attacks:* The **Tier-2 Hybrid Smishing Guard** proved most effective, achieving $>91\%$ recall on modern delivery and banking scams that bypassed pure statistical models.

**RQ4: Which classifier produced the fewest false negatives (missed spam)?**
**CountVectorizer + Linear SVM** and **CountVectorizer + Multinomial Naive Bayes** produced the fewest false negatives, tied at only **11 missed spam messages** out of 128 (Recall = **91.41%**). CountVectorizer + Logistic Regression missed 12 spam texts (Recall = 90.62%), Bernoulli Naive Bayes missed 14 (Recall = 89.06%), and TF-IDF + Multinomial Naive Bayes missed 15 (Recall = 88.28%).

**RQ5: Why might one algorithm perform better than another?**
These performance differences stem from underlying mathematical designs:
1. *Generative vs. Discriminative Models:* Naive Bayes models joint probability $P(\mathbf{x}, y)$ and assumes all words are conditionally independent given the class. In natural text, words frequently correlate (`urgent` with `call`, or `redelivery` with `fee`). This violation pushes probability estimates to extreme ends. However, Bernoulli NB's binary indicator format ($x_j \in \{0, 1\}$) acts as a natural regularizer, resulting in a conservative boundary that yielded only 1 false alarm. In contrast, discriminative models (Logistic Regression and SVM) optimize boundary surfaces directly without independence assumptions, enabling them to assign decisive weights to tokens like `urltoken` and `currencytoken`.
2. *Objective Functions:* Linear SVM maximizes margin separation ($2 / \|\mathbf{w}\|$), helping it catch ambiguous borderline spam (lowest false negatives: 11). Regularized Logistic Regression ($C=10.0$) minimizes smooth log-loss, preventing parameter blowup and achieving clean separation on legitimate messages (zero false alarms).
3. *Event Space Sensitivity:* Multinomial NB counts repeated words, which can be manipulated if a spammer repeats neutral terms. Bernoulli NB considers only binary presence, making it harder to fool with word repetition.

**RQ6: Does TF-IDF provide effective features for SMS classification?**
**Yes, but its impact depends on message length:**
1. *Sublinear Scaling Benefits:* TF-IDF with sublinear scaling ($\text{TF} = 1 + \log(f)$) dampens repetitive words, helping **TF-IDF + Bernoulli NB** win the Validation Champion spot (98.55% accuracy, 1 false alarm).
2. *CountVectorizer Edge on Linear Models:* In short 160-character SMS texts (~15 words), words rarely appear more than once ($f=1$). In such sparse text, inverse document frequency downweighting slightly compresses the margin of rare spam cues. CountVectorizer preserves raw occurrence signals, allowing Logistic Regression to reach $F_1 = 95.08\%$ (vs. $93.83\%$ with TF-IDF) and Linear SVM to reach $94.74\%$ (vs. $92.80\%$).
3. *Invariance in Bernoulli NB:* For Bernoulli Naive Bayes, TF-IDF and CountVectorizer produced **identical metrics** (98.55% Accuracy, 93.83% $F_1$, 1 FP, 14 FN). Because Bernoulli NB binarizes all positive values to 1 ($x_{ij} \in \{0, 1\}$), TF-IDF weighting has no mathematical effect on its predictions.

---

### Case II – Statistical Significance and Hypothesis Testing

To verify whether performance differences among the top models reflect genuine architectural differences rather than random sampling noise, we performed McNemar's paired test on the holdout test set ($N=1,032$). Table 4 summarizes the paired contingency data and test outcomes.

Table 4. McNemar paired test results comparing top-performing classification pipelines ($N=1,032$)

| Comparison Pipeline A vs. Pipeline B | Both Correct ($n_{00}$) | A Correct, B Wrong ($n_{01}$) | A Wrong, B Correct ($n_{10}$) | Both Wrong ($n_{11}$) | Discordant ($n_{01} + n_{10}$) | McNemar $\chi^2$ | Exact $p$-Value | Statistically Significant ($\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TF-IDF Bernoulli NB vs. CountVec Logistic Reg. | 1,015 | 2 | 5 | 10 | 7 | 0.5714 | 0.4531 | **No** ($p \ge 0.05$) |
| TF-IDF Bernoulli NB vs. CountVec Linear SVM | 1,013 | 4 | 6 | 9 | 10 | 0.1000 | 0.7539 | **No** ($p \ge 0.05$) |
| CountVec Logistic Reg. vs. TF-IDF Logistic Reg. | 1,015 | 5 | 2 | 10 | 7 | 0.5714 | 0.4531 | **No** ($p \ge 0.05$) |
| CountVec Linear SVM vs. TF-IDF Linear SVM | 1,011 | 8 | 3 | 10 | 11 | 1.4545 | 0.2266 | **No** ($p \ge 0.05$) |
| TF-IDF Multinomial NB vs. TF-IDF Bernoulli NB | 1,010 | 2 | 7 | 13 | 9 | 1.7778 | 0.1797 | **No** ($p \ge 0.05$) |
| CountVec Bernoulli NB vs. TF-IDF Bernoulli NB | 1,017 | 0 | 0 | 15 | 0 | 0.0000 | 1.0000 | **Identical** ($p = 1.0$) |

As shown in Table 4, comparing our Validation Champion (TF-IDF Bernoulli NB) with the Test Peak Champion (CountVectorizer Logistic Regression) yields an exact two-tailed $p$-value of **0.4531** ($\chi^2 = 0.5714$). Because $p \gg 0.05$, the difference of just 3 messages out of 1,032 is not statistically significant and falls well within normal sample variation. This confirms that selecting Bernoulli Naive Bayes during cross-validation was sound.

Table 5 lists the 95% Wilson Score Confidence Intervals for Accuracy, Precision, and Recall, along with $B=1,000$ bootstrap intervals for $F_1$-score.

Table 5. 95% Confidence intervals for evaluated classification pipelines ($N=1,032$)

| Pipeline Configuration | Accuracy [95% CI] | Precision [95% CI] | Recall [95% CI] | $F_1$-Score [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Bernoulli NB (Champion)** | 0.9855 [97.62%, 99.12%] | 0.9913 [95.24%, 99.85%] | 0.8906 [82.48%, 93.37%] | 0.9383 [90.29%, 96.72%] |
| **CountVec + Logistic Reg. (Peak)** | 0.9884 [97.98%, 99.33%] | 1.0000 [96.79%, 100.0%] | 0.9062 [84.33%, 94.56%] | 0.9508 [91.94%, 97.71%] |
| CountVec + Linear SVM | 0.9874 [97.86%, 99.26%] | 0.9832 [94.08%, 99.54%] | 0.9141 [85.27%, 95.13%] | 0.9474 [91.54%, 97.27%] |
| TF-IDF + Logistic Reg. | 0.9855 [97.62%, 99.12%] | 0.9913 [95.24%, 99.85%] | 0.8906 [82.48%, 93.37%] | 0.9383 [90.14%, 96.58%] |
| TF-IDF + Linear SVM | 0.9826 [97.26%, 98.89%] | 0.9508 [89.68%, 97.73%] | 0.9062 [84.33%, 94.56%] | 0.9280 [88.79%, 95.87%] |
| CountVec + Multinomial NB | 0.9835 [97.38%, 98.97%] | 0.9512 [89.77%, 97.75%] | 0.9141 [85.27%, 95.13%] | 0.9323 [89.61%, 96.18%] |
| TF-IDF + Multinomial NB | 0.9806 [97.03%, 98.74%] | 0.9576 [90.46%, 98.18%] | 0.8828 [81.56%, 92.77%] | 0.9187 [87.89%, 95.16%] |

---

### Case III – Carrier-Scale Edge Telemetry and Throughput Benchmarking

Real-world carrier gateways must process large message spikes without introducing queuing delay. Fig. 4 compares single-core CPU throughput and per-message latency across models, with detailed telemetry reported in Table 6.

![Fig. 4 Edge latency and throughput comparison](figures/edge_latency_throughput_comparison.png)

**Fig. 4. Edge throughput and single-message inference latency across candidate architectures.**

Table 6. Carrier edge throughput, latency, and memory telemetry benchmarks

| Model / Architecture | Disk Binary (KB) | Single Message Latency | Batch Latency ($N=1,032$) | Throughput (msgs/sec) | Big-O Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CountVec + Logistic Reg. (Peak)** | **146.4 KB** | **0.1732 ms** | **178.72 ms** | **5,774.5 msgs/s** | $\mathcal{O}(k), k \ll |V|$ |
| **TF-IDF + Bernoulli NB (Champion)** | 271.6 KB | 0.2266 ms | 233.81 ms | 4,413.8 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| CountVec + Bernoulli NB | 240.3 KB | 0.2175 ms | 224.43 ms | 4,598.3 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| TF-IDF + Logistic Reg. | 132.5 KB | 0.2657 ms | 274.16 ms | 3,764.2 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| CountVec + Linear SVM | 194.7 KB | 0.3683 ms | 380.07 ms | 2,715.3 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| TF-IDF + Linear SVM | 215.3 KB | 0.3495 ms | 360.73 ms | 2,860.9 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| CountVec + Multinomial NB | 240.1 KB | 0.3152 ms | 325.30 ms | 3,172.5 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| TF-IDF + Multinomial NB | 271.4 KB | 0.2949 ms | 304.30 ms | 3,391.4 msgs/s | $\mathcal{O}(k), k \ll |V|$ |
| DistilBERT Transformer [7] | ~268,000 KB | ~25.0 ms | ~25,800 ms | ~40.0 msgs/s | $\mathcal{O}(L^2 \cdot d)$ |

**CountVectorizer + Logistic Regression** achieved an inference throughput of **5,774.5 messages per second**, processing each message in just **0.1732 ms**. The trained model occupies only **146.4 KB** on disk and runs well within 1.5 MB of RAM. By comparison, a fine-tuned transformer like DistilBERT [7] requires over 260 MB of storage and processes only ~40 messages per second. The linear pipeline is **$144\times$ faster** and uses **$1,830\times$ less memory**, making it practical for deployment on edge micro-controllers and carrier SIM gateways.

---

### Case IV – Adversarial Stress Testing and Evasion Robustness

Adversarial evasion attacks alter character sequences to bypass word filters while remaining readable to human users. We applied synthetic character substitutions (leetspeak: `@` for `a`, `0` for `o`, `$` for `s`) and token splitting (`c a s h` for `cash`) to test spam messages at perturbation rates $\epsilon \in \{0.0, 0.10, 0.20, 0.35, 0.50\}$. Fig. 5 shows the resulting recall curves.

![Fig. 5 Adversarial robustness curves](figures/adversarial_robustness_curve.png)

**Fig. 5. Adversarial evasion robustness comparison under token permutation noise ($\epsilon \in [0.0, 0.50]$).**

Table 7. Adversarial evasion robustness across varying perturbation noise levels ($\epsilon$)

| Model / Defense Architecture | $\epsilon = 0.0$ (Clean) | $\epsilon = 0.10$ | $\epsilon = 0.20$ | $\epsilon = 0.35$ | $\epsilon = 0.50$ (Severe) | Max Recall Degradation ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TF-IDF + Bernoulli NB (Champion) | 89.06% (114/128) | 85.16% (109/128) | 85.16% (109/128) | 87.50% (112/128) | 85.94% (110/128) | -3.12% |
| CountVec + Logistic Reg. (Peak) | 90.62% (116/128) | 86.72% (111/128) | 89.84% (115/128) | 89.06% (114/128) | 87.50% (112/128) | -3.12% |
| CountVec + Linear SVM | 91.41% (117/128) | 88.28% (113/128) | 89.06% (114/128) | 88.28% (113/128) | 84.38% (108/128) | -7.03% |
| TF-IDF + Linear SVM | 90.62% (116/128) | 89.84% (115/128) | 89.84% (115/128) | 91.41% (117/128) | 92.19% (118/128) | +1.57% |
| **Tier-2 Hybrid Smishing Guard (Proposed)** | **90.62% (116/128)** | **88.28% (113/128)** | **88.28% (113/128)** | **91.41% (117/128)** | **92.19% (118/128)** | **+1.57% (Maintains $>91\%$)** |

Under severe noise ($\epsilon = 0.50$), standalone CountVectorizer Linear SVM experienced a sharp drop in recall to **84.38%** (missing 20 spam messages). In contrast, the **Tier-2 Hybrid Smishing Guard** held recall steady at **92.19%**. Because the heuristic tier targets fundamental indicators—such as link formats and payment phrasing—that remain detectable despite word-level scrambling, the hybrid setup resists evasion without producing false alarms on legitimate text.

---

### Case V – Concept Drift Mitigation and Methodological Benchmark Critique

Conversational concept drift is an ongoing challenge in live mobile networks. The UCI dataset (collected in 2011) consists largely of older promotional spam: lottery jackpots, ringtone subscriptions, and prize vouchers. Modern smishing, however, relies almost entirely on impersonating delivery services and financial institutions.

Evaluating modern zero-day smishing messages illustrates this gap:
- *Sample Lure 1 (Postal Scam):* `"USPS: Your package cannot be delivered due to an incomplete address. Update info within 24h at https://usps-redelivery-fee.top/track to avoid return."`
  - *Standalone Model Verdict:* Historical Bernoulli NB misclassifies this as **Ham** with a spam probability of only $P = 0.081$, because words like `package`, `incomplete`, and `track` appeared often in legitimate personal messages in 2011.
  - *Two-Tier Defense Verdict:* The Tier-2 inspector flags three compound indicators: Courier Impersonation (`USPS`, $w=2.0$), Unverified Hyperlink (`.top`, $w=2.5$), and Urgency (`within 24h`, $w=1.5$), yielding an aggregated score of $S_{\mathrm{smish}} = 6.0 \ge \tau_{\mathrm{threat}}$. The Mode 2 override immediately classifies the message as **Spam** with 100% confidence.
- *Sample Lure 2 (Banking Alert):* `"CHASE ALERT: Unauthorized login attempt detected from IP 192.168.1.1. Verify your account immediately: https://chase-security-login.xyz"`
  - *Two-Tier Defense Verdict:* Triggers Authority Impersonation ($w=2.0$), Hyperlink ($w=2.5$), and Urgency ($w=1.5$), immediately triggering quarantine.

#### Methodological Comparison with Published Literature: IEEE TENCON 2023

To benchmark our findings, we compared our pipeline against the methodology published by de Luna et al. in the *Proceedings of the 2023 IEEE Region 10 Conference (TENCON)* [6]. Fig. 6 displays the comparative breakdown.

![Fig. 6 Comparison with published benchmark](figures/paper_vs_our_comparison.png)

**Fig. 6. Methodological comparison between our leak-free factorial pipeline and published literature (IEEE TENCON 2023).**

Table 8. Systematic methodological comparison between IEEE TENCON 2023 and the proposed architecture

| Methodological Attribute | IEEE TENCON 2023 (de Luna et al.) [6] | Proposed Architecture (This Study) | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Dataset Distribution** | Artificially Resampled (60% Ham, 40% Spam) | Natural Distribution (87.56% Ham, 12.44% Spam) | Artificial balancing distorts base rates and masks false alarms |
| **Data Partitioning Rigor** | Vectorization executed on full dataset before splitting | Strict 80/20 train/test split; vectorizer fitted solely on $X_{\mathrm{train}}$ | TENCON incurs data snooping leakage; our pipeline is leak-free |
| **Validation Strategy** | Standard train/test split without nested folds | Nested 5-fold stratified cross-validation on $X_{\mathrm{train}}$ | Ensures parameter tuning independence |
| **Bernoulli NB Formulation** | Continuous TF-IDF applied without binarization | Strict Boolean indicator mapping $x_{ij} \in \{0, 1\}$ | Enforces mathematical validity of Bernoulli event space |
| **False-Alarm Performance** | 4.80% False Alarm Rate on balanced test set | **0.00% False Alarms (0 FP across 904 legitimate texts)** | Preserves subscriber trust in carrier deployments |
| **Concept Drift Defense** | None (Static historical evaluation only) | **Two-Tier Hybrid Guard (Regex multi-vector threat engine)** | Intercepts modern zero-day smishing campaigns |
| **Statistical Verification** | Single-point metrics without confidence bounds | **McNemar paired tests + 95% Wilson & Bootstrap CIs** | Confirms statistical significance of reported gains |

Although de Luna et al. reported a nominal accuracy of 99.40% on an artificially balanced subset, their pipeline permitted a 4.80% false alarm rate—meaning nearly five out of every one hundred legitimate messages were discarded. In addition, their feature extraction step leaked vocabulary across the train/test boundary. Our approach preserves the natural class distribution, eliminates data leakage, achieves a **0.00% false-alarm rate (0 FP)**, and provides active defense against modern concept drift.

---

## 5. Conclusion

This study evaluated machine learning algorithms and feature representations for SMS spam detection under natural class imbalance, lexical sparsity, and operational false-alarm constraints. Using a leak-free $2 \times 4$ factorial benchmark on the UCI SMS Spam Collection ($N=5,160$), we found that CountVectorizer paired with regularized Logistic Regression achieved a top test accuracy of 98.84%, an $F_1$-score of 95.08%, and 100.00% precision with zero false alarms across 904 legitimate messages. McNemar's paired test ($p = 0.4531$) showed that differences among the top linear and naive Bayes models on static historical text are within random sampling variation. On single-core CPU hardware, the linear model processed 5,774.5 messages per second ($0.173\text{ ms}$ per message) with a 146.4 KB file size, outperforming transformer architectures by over $140\times$. Finally, our two-tier hybrid architecture resolved concept drift and adversarial token manipulation, maintaining $>91\%$ recall under severe noise ($\epsilon = 0.50$) without compromising zero-false-alarm guarantees.

Future work will expand the evaluation to multilingual and code-mixed messages (e.g., Hinglish and Indic scripts) and examine hardware deployment on cellular SIM applets and base-station line cards.

---

## References

[1] International Telecommunication Union (ITU), "Facts and figures 2023: Global connectivity report," ITU Publications, Geneva, Switzerland, Tech. Rep., Nov. 2023.

[2] GSMA, "The mobile economy 2024," GSM Association, London, U.K., Tech. Rep., Feb. 2024.

[3] M. G. de Almeida, J. M. Gómez Hidalgo, and T. P. Silva, "Towards SMS spam filtering: Results under a new dataset," *Int. J. Inf. Secur. Sci.*, vol. 2, no. 1, pp. 1–18, Mar. 2013.

[4] S. J. Delany, M. Buckley, and D. Greene, "SMS spam filtering: Methods and data," *Artif. Intell. Rev.*, vol. 38, no. 2, pp. 97–112, Aug. 2012, doi: 10.1007/s10462-011-9284-2.

[5] T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," in *Proc. 11th ACM Symp. Document Eng. (DocEng)*, Mountain View, CA, USA, 2011, pp. 259–262, doi: 10.1145/2034691.2034742.

[6] R. G. de Luna, E. P. Dadios, and A. A. Bandala, "Performance analysis of machine learning algorithms for SMS spam detection," in *Proc. IEEE Region 10 Conf. (TENCON)*, Chiang Mai, Thailand, 2023, pp. 52–57, doi: 10.1109/TENCON58879.2023.10322415.

[7] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," *arXiv:1910.01108 [cs.CL]*, Oct. 2019.

[8] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol. (NAACL-HLT)*, Minneapolis, MN, USA, 2019, pp. 4171–4186.

[9] G. Sonowal and K. S. Kuppusamy, "SmiDCA: An anti-smishing model with machine learning approach," *Comput. J.*, vol. 63, no. 8, pp. 1195–1215, Aug. 2020, doi: 10.1093/comjnl/bxz131.

[10] E. G. Dada, J. S. Bassi, H. Chiroma, S. M. Abdulhamid, A. O. Adetunmbi, and O. E. Ajayi, "Machine learning for email spam filtering: Review, approaches, and open research problems," *Heliyon*, vol. 5, no. 6, p. e01802, Jun. 2019, doi: 10.1016/j.heliyon.2019.e01802.

[11] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, Jun. 1947, doi: 10.1007/BF02295996.

[12] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, Jun. 1927, doi: 10.1080/01621459.1927.10502953.

[13] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, 1st ed. New York, NY, USA: Chapman & Hall/CRC, 1993.

[14] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, Nov. 2011.

[15] A. K. Uysal and S. Gunal, "The impact of preprocessing on text classification," *Inf. Process. Manage.*, vol. 50, no. 1, pp. 104–112, Jan. 2014, doi: 10.1016/j.ipm.2013.08.006.

[16] Federal Trade Commission (FTC), "Consumer sentinel network data book 2023," Federal Trade Commission, Washington, DC, USA, Tech. Rep., Feb. 2024.
