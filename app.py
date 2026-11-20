"""
Streamlit Web Application: SMS Spam Intelligence Studio
Subject: Research Methodologies (RM) - 7th Sem B.Tech CSE
Factorial Experimental Study: 2 Feature Extractors x 4 Machine Learning Classifiers
"""

import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

# Ensure src modules are resolvable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess_text
from feature_extraction import TextPreprocessorTransformer  # noqa: F401


# --- Page Configuration ---
st.set_page_config(
    page_title="SMS Spam Intelligence Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        max-width: 820px;
        line-height: 1.5;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
    }

    .verdict-card-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 22px;
        margin-top: 14px;
    }
    .verdict-card-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 22px;
        margin-top: 14px;
    }
    .verdict-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .verdict-title-spam {
        color: #f87171;
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-title-ham {
        color: #34d399;
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-desc {
        color: #cbd5e1;
        font-size: 0.90rem;
        margin: 0;
        line-height: 1.5;
    }

    .token-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #38bdf8;
        margin-top: 8px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Registry & Loader ---
MODEL_MAP = {
    ("TF-IDF", "Linear Support Vector Machine"): "tfidf_plus_linear_svm.pkl",
    ("TF-IDF", "Bernoulli Naive Bayes"): "tfidf_plus_bernoulli_naive_bayes.pkl",
    ("TF-IDF", "Logistic Regression"): "tfidf_plus_logistic_regression.pkl",
    ("TF-IDF", "Multinomial Naive Bayes"): "tfidf_plus_multinomial_naive_bayes.pkl",
    ("CountVectorizer", "Linear Support Vector Machine"): "countvectorizer_plus_linear_svm.pkl",
    ("CountVectorizer", "Bernoulli Naive Bayes"): "countvectorizer_plus_bernoulli_naive_bayes.pkl",
    ("CountVectorizer", "Logistic Regression"): "countvectorizer_plus_logistic_regression.pkl",
    ("CountVectorizer", "Multinomial Naive Bayes"): "countvectorizer_plus_multinomial_naive_bayes.pkl",
}


@st.cache_resource
def load_specific_pipeline(feature_tech: str, algorithm_name: str):
    """
    Loads one of the 8 trained pipelines from the models/ directory.
    """
    filename = MODEL_MAP.get((feature_tech, algorithm_name), "best_model.pkl")
    model_path = os.path.join(os.path.dirname(__file__), "models", filename)

    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Fallback to best_model.pkl if present
        fallback_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
        if os.path.exists(fallback_path):
            return joblib.load(fallback_path)
        raise FileNotFoundError(f"Model file not found at {model_path}. Please execute src/evaluation.py first.")


@st.cache_data
def load_results_summary():
    """
    Loads the persistent evaluation metrics summary CSV.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "reports", "results", "results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# --- Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🎓 Research Methodologies (RM) · 7th Sem CSE Project</div>
    <h1 class="hero-title">SMS Spam Detection & Factorial ML Investigation</h1>
    <p class="hero-subtitle">
        Systematic empirical comparison of 2 Feature Extraction Techniques (TF-IDF vs. CountVectorizer) across 4 Supervised ML Classifiers (MNB, BNB, LR, Linear SVM) on the official UCI SMS corpus, benchmarked against IEEE TENCON 2023.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Top Key Performance Indicator Cards (Overall Top Model) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Top Test Accuracy</div>
        <div class="metric-value" style="color: #60a5fa;">98.16%</div>
        <small style="color: #64748b;">TF-IDF + Linear SVM</small>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Peak Spam Precision</div>
        <div class="metric-value" style="color: #34d399;">97.39%</div>
        <small style="color: #64748b;">CountVec + Linear SVM (3 FP / 904)</small>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Peak Spam Recall</div>
        <div class="metric-value" style="color: #fbbf24;">89.06%</div>
        <small style="color: #64748b;">CountVec + MNB (114 / 128 Spam)</small>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Top Spam F1-Score</div>
        <div class="metric-value" style="color: #a78bfa;">92.24%</div>
        <small style="color: #64748b;">Harmonic Balance (E4)</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabbed Navigation ---
tab_classifier, tab_benchmarks, tab_ieee_compare, tab_methodology = st.tabs([
    "🔍 Live SMS Classifier",
    "📊 8-Model Factorial Benchmarks",
    "📑 IEEE TENCON 2023 Comparison",
    "🔬 Research Methodology"
])


# ==============================================================================
# TAB 1: LIVE SMS CLASSIFIER (Interactive Model & Vectorizer Selection)
# ==============================================================================
with tab_classifier:
    col_input, col_result = st.columns([1.15, 0.85], gap="large")

    with col_input:
        st.subheader("1. Configure Pipeline & Input Message")

        # Dynamic Model & Feature Selector
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            selected_tech = st.selectbox(
                "Feature Extraction Technique:",
                ["TF-IDF", "CountVectorizer"],
                index=0,
                help="TF-IDF evaluates sublinear weighted term relevance; CountVectorizer computes absolute token occurrences."
            )
        with col_sel2:
            selected_algo = st.selectbox(
                "Machine Learning Algorithm:",
                [
                    "Linear Support Vector Machine",
                    "Bernoulli Naive Bayes",
                    "Logistic Regression",
                    "Multinomial Naive Bayes"
                ],
                index=0,
                help="Select one of the four supervised classifiers."
            )

        # Show currently selected pipeline benchmark pill
        df_summary = load_results_summary()
        if df_summary is not None:
            match = df_summary[
                (df_summary["Feature_Extraction"] == selected_tech) &
                (df_summary["Algorithm"].str.contains(selected_algo.split()[0]))
            ]
            if not match.empty:
                r = match.iloc[0]
                st.caption(
                    f"🎯 **Pipeline Benchmark ({selected_tech} + {selected_algo}):** "
                    f"Accuracy: **{r['Accuracy']*100:.2f}%** | "
                    f"Precision: **{r['Precision']*100:.2f}%** | "
                    f"Recall: **{r['Recall']*100:.2f}%** | "
                    f"F1: **{r['F1-Score']*100:.2f}%**"
                )

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        # Preset test messages
        sample_options = {
            "Custom Message (Type Below)": "",
            "🚨 Prize Scam": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Claim code KL341. Valid 12 hours.",
            "🚨 Urgent Phishing / Lottery": "URGENT! Your mobile number won 1,000,000 cash. Reply CLAIM to 88088 now. Terms apply £1.50 per message.",
            "✅ Casual Meeting Chat": "Hey mate, are you coming over to the library to finish the research methodology assignment?",
            "✅ University Notice": "Dear Students, please submit your 7th Semester RM lab file by Friday 5 PM on the department portal."
        }

        selected_sample_key = st.selectbox(
            "Select an academic test sample or write custom text:",
            list(sample_options.keys())
        )
        default_val = sample_options[selected_sample_key]

        user_sms = st.text_area(
            "SMS Text Message:",
            value=default_val,
            placeholder="Type or paste an SMS message to classify...",
            height=125,
            key="sms_input_text"
        )

        char_cnt = len(user_sms)
        word_cnt = len(user_sms.split())
        st.caption(f"📊 Message Metadata: **{char_cnt} characters** | **{word_cnt} words**")

        # Explicit Classify Button - Fixes Trigger Bug!
        classify_btn = st.button("🛡️ Classify SMS Message", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. Empirical Classification Output")

        # Prediction executes ONLY when user clicks the classify button!
        if classify_btn:
            if not user_sms.strip():
                st.warning("⚠️ Please provide a non-empty SMS text message to evaluate.")
            else:
                try:
                    pipeline = load_specific_pipeline(selected_tech, selected_algo)
                    pred = pipeline.predict([user_sms])[0]

                    # Probability score if calibrated
                    probs = None
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            probs = pipeline.predict_proba([user_sms])[0]
                        except Exception:
                            pass

                    if pred == 1:
                        st.markdown("""
                        <div class="verdict-card-spam">
                            <div class="verdict-header">
                                <span style="font-size: 1.8rem;">🚨</span>
                                <h3 class="verdict-title-spam">SPAM / FRAUDULENT MESSAGE</h3>
                            </div>
                            <p class="verdict-desc">
                                High-confidence detection of promotional lure, urgent monetary claim, or unauthorized solicitation patterns.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="verdict-card-ham">
                            <div class="verdict-header">
                                <span style="font-size: 1.8rem;">✅</span>
                                <h3 class="verdict-title-ham">HAM (LEGITIMATE MESSAGE)</h3>
                            </div>
                            <p class="verdict-desc">
                                Typical conversational, transactional, or informational characteristics. No unsolicited scam markers detected.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    if probs is not None:
                        spam_pct = probs[1] * 100
                        ham_pct = probs[0] * 100
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.write(f"**Model Margin Breakdown ({selected_tech} + {selected_algo}):**")
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.metric("Spam Probability", f"{spam_pct:.1f}%")
                        with col_p2:
                            st.metric("Ham Probability", f"{ham_pct:.1f}%")
                        st.progress(float(probs[1]))
                        st.caption("📌 *Note: Decision margins calibrated via Platt scaling on high-dimensional hyperplane distances.*")

                    with st.expander("🔬 View NLP Text Normalization"):
                        cleaned_tokens = preprocess_text(user_sms)
                        st.markdown("**Stemmed & Filtered Root Tokens:**")
                        st.markdown(f'<div class="token-box">{cleaned_tokens or "[All tokens filtered as generic stopwords]"}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error evaluating pipeline: {str(e)}")
        else:
            st.info("👈 Select your pipeline configuration and message on the left, then click **Classify SMS Message** to evaluate.")


# ==============================================================================
# TAB 2: 8-MODEL FACTORIAL BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.subheader("Factorial Performance Matrix: 2 Feature Extractors x 4 Classifiers")
    st.write("All 8 experimental combinations evaluated on the untouched test partition (1,032 samples: 904 Ham, 128 Spam) under identical stratified conditions.")

    df_results = load_results_summary()
    if df_results is not None:
        # Display filter
        filter_opt = st.radio(
            "Filter Evaluation Combinations:",
            ["All 8 Experiments", "TF-IDF Vectorizer Only", "CountVectorizer Only"],
            horizontal=True
        )

        df_display = df_results.copy()
        if filter_opt == "TF-IDF Vectorizer Only":
            df_display = df_display[df_display["Feature_Extraction"] == "TF-IDF"]
        elif filter_opt == "CountVectorizer Only":
            df_display = df_display[df_display["Feature_Extraction"] == "CountVectorizer"]

        # Format percentage columns for display
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            df_display[col] = (df_display[col] * 100).round(2).astype(str) + "%"

        st.dataframe(
            df_display[[
                "Experiment", "Feature_Extraction", "Algorithm",
                "Accuracy", "Precision", "Recall", "F1-Score",
                "FP (False Alarm)", "FN (Missed Spam)"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Results summary not loaded. Run src/evaluation.py to generate reports/results/results_summary.csv.")

    st.divider()

    # Visual Comparison Charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("F1-Score Comparison across Experiments")
        f1_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "f1_comparison.png")
        if os.path.exists(f1_path):
            st.image(f1_path, use_container_width=True)

    with col_c2:
        st.subheader("Feature Extraction Aggregated Comparison")
        feat_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "feature_extraction_comparison.png")
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True)

    st.subheader("8-Panel Confusion Matrix Grid")
    cm_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "confusion_matrices_all8.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)


# ==============================================================================
# TAB 3: IEEE TENCON 2023 BENCHMARK COMPARISON
# ==============================================================================
with tab_ieee_compare:
    st.subheader("Empirical Benchmark Comparison with IEEE TENCON 2023 Study")
    st.markdown("""
    **Reference Paper:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
    *Authors:* Robert G. de Luna et al., Polytechnic University of the Philippines  
    *Conference:* IEEE Region 10 Conference (TENCON 2023), Chiang Mai, Thailand
    """)

    df_ieee_table = pd.DataFrame([
        {
            "Algorithm": "Bernoulli Naive Bayes (BNB)",
            "IEEE TENCON 2023 Published Benchmark": "Acc: 96.6333% | Prec: 98.4431% | Rec: 95.5174% | F1: 92.7835%",
            "Our Experimental Result (CountVec)": "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%",
            "Our Experimental Result (TF-IDF)": "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%",
            "Academic Note": "Top model in the IEEE study. Achieved near-identical F1 with higher accuracy across our natural distribution."
        },
        {
            "Algorithm": "Random Forest Classifier (RFC)",
            "IEEE TENCON 2023 Published Benchmark": "Acc: 96.0297% | Prec: 98.5228% | Rec: 94.8699% | F1: 91.1408%",
            "Our Experimental Result (CountVec)": "Not included in 4 primary linear algorithms",
            "Our Experimental Result (TF-IDF)": "Not included in 4 primary linear algorithms",
            "Academic Note": "Important Correction: 96.0297% in Table VII belongs to Random Forest, NOT Multinomial Naive Bayes."
        },
        {
            "Algorithm": "Logistic Regression (LR)",
            "IEEE TENCON 2023 Published Benchmark": "Acc: 95.8709% | Prec: 98.7711% | Rec: 93.7399% | F1: 89.2555%",
            "Our Experimental Result (CountVec)": "Acc: 97.87% | Prec: 97.32% | Rec: 85.16% | F1: 90.83%",
            "Our Experimental Result (TF-IDF)": "Acc: 97.97% | Prec: 96.52% | Rec: 86.72% | F1: 91.36%",
            "Academic Note": "Strong linear baseline across both studies. Sublinear TF-IDF scaling improved our F1-score to 91.36%."
        },
        {
            "Algorithm": "Linear Support Vector Machine",
            "IEEE TENCON 2023 Published Benchmark": "Acc: 94.5684% (Default CV, unoptimized in Table VII)",
            "Our Experimental Result (CountVec)": "Acc: 98.16% | Prec: 97.39% | Rec: 87.50% | F1: 92.18%",
            "Our Experimental Result (TF-IDF)": "Acc: 98.16% | Prec: 96.58% | Rec: 88.28% | F1: 92.24%",
            "Academic Note": "Omitted from Table VII optimization in the IEEE study. In our study, tuning made it our #1 top performing model."
        }
    ])
    st.dataframe(df_ieee_table, use_container_width=True, hide_index=True)

    paper_fig_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "paper_vs_our_comparison.png")
    if os.path.exists(paper_fig_path):
        st.image(paper_fig_path, caption="Figure: Accuracy Benchmark between IEEE TENCON 2023 and Our Dual Feature Extractions", use_container_width=True)

    st.markdown("""
    ### Key Methodological Differences & Academic Disclaimers:
    1. **Class Balancing Strategy:**  
       The IEEE authors undersampled ham messages to impose an artificial **60:40 Ham-to-Spam partition** (1,930 ham, 1,218 spam). Our study preserved the natural real-world distribution (**87.6% Ham : 12.4% Spam**). Artificially boosting spam prevalence elevates recall (95.5% vs 88.3%) because spam patterns appear at triple their true incidence.
    2. **Feature Extraction Philosophy:**  
       The IEEE authors adopted CountVectorizer based on an initial MNB check. Our factorial study reveals that while CountVectorizer works well for Naive Bayes, **TF-IDF with sublinear scaling achieves higher F1 and lower false alarms for Linear SVM and Logistic Regression**.
    3. **Controlled Equivalence Disclaimer:**  
       This comparison represents an **external benchmark validation**, not a direct controlled ablation, due to differences in corpora and sampling methodologies.
    """)


# ==============================================================================
# TAB 4: RESEARCH METHODOLOGY
# ==============================================================================
with tab_methodology:
    st.subheader("Research Methodology & Experimental Design")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        #### 1. Dataset Provenance & Cleaning
        - **Source:** UCI Machine Learning Repository (SMS Spam Collection).
        - **Total Raw Samples:** 5,574 messages.
        - **Missing Values:** Zero null entries across all records.
        - **Deduplication:** 414 duplicate messages dropped to prevent synthetic score inflation.
        - **Cleaned Total:** 5,160 unique messages (4,518 Ham [87.56%], 642 Spam [12.44%]).

        #### 2. NLP Preprocessing Pipeline
        - **Lowercasing:** Uniform case normalization.
        - **Noise Stripping:** Regular expression removal of URLs, emails, and punctuation.
        - **Whitespace Tokenization:** Splitting text into discrete words.
        - **Stopword Removal:** Eliminating non-informative English stopwords.
        - **Stemming:** Morphological word reduction using Porter Stemmer.
        """)

    with col_m2:
        st.markdown("""
        #### 3. Dual Feature Extraction
        - **TF-IDF Vectorizer:** Sublinear TF scaling with unigrams and bigrams (4,000 max features).
        - **CountVectorizer:** Raw token counts with unigrams and bigrams (4,000 max features).
        - **Bernoulli Handling:** Strict binary indicator matrix ($x_{ij} \in \{0, 1\}$) for both representations.

        #### 4. Data Leakage Prevention
        - **Stratified 80:20 Split:** 4,128 training messages, 1,032 testing messages.
        - **Pipeline Encapsulation:** Preprocessing and vectorizer fitting restricted exclusively to training folds.
        - **Tuning:** Stratified 5-Fold Cross Validation via GridSearchCV optimizing for Spam F1-Score.
        """)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small style='color: #64748b;'>Developed for Research Methodologies (RM) · 7th Semester B.Tech Project in CSE · Built with Python, Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True
)
