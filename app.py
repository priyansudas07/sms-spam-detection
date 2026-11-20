"""
Streamlit Web Application: SMS Spam Detection
Subject: Research Methodologies (RM) - 7th Sem B.Tech CSE
"""

import os
import sys
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

# --- Premium Custom Styling (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Brand Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
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
        margin-bottom: 12px;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        max-width: 780px;
        line-height: 1.5;
    }

    /* Stat Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
    }

    /* Verdict Cards */
    .verdict-card-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
    }
    .verdict-card-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin-top: 16px;
    }
    .verdict-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .verdict-title-spam {
        color: #f87171;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-title-ham {
        color: #34d399;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-desc {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Token Code Box */
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

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loader ---
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        from data_loader import load_raw_data, clean_data
        from feature_extraction import split_data
        from model_training import tune_models, save_model
        raw_df = load_raw_data(os.path.join(os.path.dirname(__file__), "data", "SMSSpamCollection"))
        clean_df, _ = clean_data(raw_df)
        X_train, _, y_train, _ = split_data(clean_df)
        tuned_models, _ = tune_models(X_train, y_train)
        best_pipeline = tuned_models["Linear SVM"]
        save_model(best_pipeline, model_path)
        return best_pipeline


# --- Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🎓 Research Methodologies (RM) · 7th Sem Project</div>
    <h1 class="hero-title">SMS Spam Detection & Empirical Analysis</h1>
    <p class="hero-subtitle">
        Comparative investigation of Supervised Machine Learning algorithms (Multinomial NB, Bernoulli NB, Logistic Regression, Linear SVM) on the official UCI SMS corpus with external benchmark validation against IEEE TENCON 2023.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Top Key Performance Indicator Cards ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value" style="color: #60a5fa;">98.16%</div>
        <small style="color: #64748b;">Linear SVM (Test Set)</small>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Spam Precision</div>
        <div class="metric-value" style="color: #34d399;">96.58%</div>
        <small style="color: #64748b;">Only 4 False Positives / 904</small>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Spam Recall</div>
        <div class="metric-value" style="color: #fbbf24;">88.28%</div>
        <small style="color: #64748b;">113 / 128 Spam Blocked</small>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Spam F1-Score</div>
        <div class="metric-value" style="color: #a78bfa;">92.24%</div>
        <small style="color: #64748b;">Balanced Harmonic Mean</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabbed Navigation ---
tab_classifier, tab_benchmarks, tab_ieee_compare, tab_methodology = st.tabs([
    "🔍 Live SMS Classifier",
    "📊 Empirical Model Benchmarks",
    "📑 IEEE Paper Comparison",
    "🔬 Research Methodology"
])


# ==============================================================================
# TAB 1: LIVE SMS CLASSIFIER
# ==============================================================================
with tab_classifier:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("Message Input & Test Cases")

        # Preset Chips
        sample_options = {
            "Custom Message (Type Below)": "",
            "🚨 Prize Scam": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Claim code KL341.",
            "🚨 Urgent Phishing": "URGENT! Your mobile account has won a guaranteed £1000 cash prize. Reply CLAIM to 87121 immediately. Terms apply.",
            "✅ Casual Chat": "Hey mate, are you coming over to the library to finish the research methodology assignment?",
            "✅ University Notice": "Dear Students, please submit your 7th Semester RM lab file by Friday 5 PM on the department portal."
        }

        selected_key = st.selectbox(
            "Select a pre-loaded test case or write your own:",
            list(sample_options.keys())
        )
        default_val = sample_options[selected_key]

        user_sms = st.text_area(
            "SMS Message Text:",
            value=default_val,
            placeholder="Paste or type an SMS message to evaluate...",
            height=130,
            key="sms_input_area"
        )

        char_cnt = len(user_sms)
        word_cnt = len(user_sms.split())
        st.caption(f"📊 Message Length: **{char_cnt} characters** | **{word_cnt} words**")

        classify_btn = st.button("🛡️ Classify SMS Message", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Classification Verdict")

        if classify_btn or (user_sms and user_sms != ""):
            if not user_sms.strip():
                st.info("👈 Enter an SMS message or select a pre-loaded example to run the classifier.")
            else:
                model = load_trained_model()
                pred = model.predict([user_sms])[0]
                
                # Predict probability if supported
                probs = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba([user_sms])[0]
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
                            High-confidence match with promotional spam, lottery fraud, or credential lure patterns. This message should be filtered and flagged for user protection.
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
                            Standard conversational, institutional, or transactional language characteristics. No unsolicited fraud markers detected.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                if probs is not None:
                    spam_pct = probs[1] * 100
                    ham_pct = probs[0] * 100
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("**Model Probability Distribution:**")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Spam Confidence", f"{spam_pct:.1f}%")
                    with col_p2:
                        st.metric("Ham Confidence", f"{ham_pct:.1f}%")
                    st.progress(float(probs[1]))
                    st.caption("📌 *Note: Decision margins calibrated via Platt scaling on high-dimensional hyperplane distances.*")

                # NLP Normalization
                with st.expander("🔬 View NLP Text Normalization"):
                    cleaned_tokens = preprocess_text(user_sms)
                    st.markdown("**Stemmed & Filtered Root Tokens:**")
                    st.markdown(f'<div class="token-box">{cleaned_tokens or "[All tokens filtered as generic stopwords]"}</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Enter an SMS message or select an example on the left, then click **Classify SMS Message**.")


# ==============================================================================
# TAB 2: EMPIRICAL MODEL BENCHMARKS
# ==============================================================================
with tab_benchmarks:
    st.subheader("Model Performance on Untouched Test Set (1,032 Samples)")
    st.write("All algorithms were evaluated on a stratified 20% test partition (904 Ham, 128 Spam) with zero data leakage.")

    # Results Table
    df_eval = pd.DataFrame([
        {"Algorithm": "Linear Support Vector Machine (Linear SVM)", "Accuracy": "98.16%", "Precision": "96.58%", "Recall": "88.28%", "F1-Score": "92.24%", "False Positives": "4 / 904"},
        {"Algorithm": "Bernoulli Naive Bayes (BNB)", "Accuracy": "98.06%", "Precision": "97.37%", "Recall": "86.72%", "F1-Score": "91.74%", "False Positives": "3 / 904"},
        {"Algorithm": "Logistic Regression (LR)", "Accuracy": "97.97%", "Precision": "96.52%", "Recall": "86.72%", "F1-Score": "91.36%", "False Positives": "4 / 904"},
        {"Algorithm": "Multinomial Naive Bayes (MNB)", "Accuracy": "97.77%", "Precision": "94.87%", "Recall": "86.72%", "F1-Score": "90.61%", "False Positives": "6 / 904"}
    ])
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.subheader("Confusion Matrix Grid")
        cm_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning("Confusion matrix chart not found. Run src/evaluation.py to generate.")

    with col_fig2:
        st.subheader("Metric Comparison Bar Chart")
        bar_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "model_comparison_barchart.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
        else:
            st.warning("Model comparison bar chart not found. Run src/evaluation.py to generate.")


# ==============================================================================
# TAB 3: IEEE PAPER COMPARISON
# ==============================================================================
with tab_ieee_compare:
    st.subheader("Comparison with Published IEEE Reference Study")
    st.markdown("""
    **Reference Study:** *A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)*  
    *Conference:* IEEE Region 10 Conference (TENCON 2023), Chiang Mai, Thailand
    """)

    df_paper = pd.DataFrame([
        {
            "Algorithm": "Bernoulli Naive Bayes",
            "IEEE TENCON 2023 (Reported)": "Acc: 96.63% | Prec: 98.44% | Rec: 95.52% | F1: 92.78%",
            "Our Project (Observed)": "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%",
            "Academic Observation": "Top model in the reference paper. Achieved near-identical F1-score in our study with higher overall accuracy."
        },
        {
            "Algorithm": "Linear Support Vector Machine",
            "IEEE TENCON 2023 (Reported)": "Acc: 94.57% (Default parameters, unoptimized)",
            "Our Project (Observed)": "Acc: 98.16% | Prec: 96.58% | Rec: 88.28% | F1: 92.24%",
            "Academic Observation": "The IEEE paper evaluated SVC only on default settings. Hyperparameter tuning with TF-IDF bigrams made it our #1 top model."
        },
        {
            "Algorithm": "Logistic Regression",
            "IEEE TENCON 2023 (Reported)": "Acc: 95.87% | Prec: 98.77% | Rec: 93.74% | F1: 89.26%",
            "Our Project (Observed)": "Acc: 97.97% | Prec: 96.52% | Rec: 86.72% | F1: 91.36%",
            "Academic Observation": "Strong linear baseline in both studies; sublinear TF-IDF scaling improved our accuracy and F1 score."
        },
        {
            "Algorithm": "Multinomial Naive Bayes",
            "IEEE TENCON 2023 (Reported)": "Acc: 96.03% | Prec: 95.59%",
            "Our Project (Observed)": "Acc: 97.77% | Prec: 94.87% | Rec: 86.72% | F1: 90.61%",
            "Academic Observation": "Highly competitive with Laplace smoothing (alpha=0.1)."
        }
    ])
    st.dataframe(df_paper, use_container_width=True, hide_index=True)

    comp_fig_path = os.path.join(os.path.dirname(__file__), "reports", "figures", "paper_vs_our_comparison.png")
    if os.path.exists(comp_fig_path):
        st.image(comp_fig_path, caption="Figure: Accuracy Comparison between IEEE TENCON 2023 and Our Study", use_container_width=True)

    st.markdown("""
    ### Critical Methodological Distinctions:
    1. **Natural vs. Artificial Class Balance:**  
       The IEEE paper undersampled ham messages to force an artificial **60:40 partition**. Our project preserved the real-world **87.6% Ham : 12.4% Spam** natural distribution. Artificially increasing spam proportion inflates recall (95.5% vs 88.3%) because spam messages occur at 3x their real-world frequency.
    2. **Word Embedding Strategy:**  
       The IEEE paper selected CountVectorizer (raw token count). We employed **Sublinear TF-IDF with unigrams and bigrams**, which penalizes common non-informative words while capturing multi-word phishing phrases (*"free entry"*, *"cash prize"*).
    """)


# ==============================================================================
# TAB 4: RESEARCH METHODOLOGY
# ==============================================================================
with tab_methodology:
    st.subheader("Research Methodology & Implementation Details")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        #### 1. Dataset Provenance & Cleaning
        - **Source:** UCI Machine Learning Repository (SMS Spam Collection).
        - **Total Raw Samples:** 5,574 messages.
        - **Missing Values:** Zero null values.
        - **Deduplication:** 414 duplicate messages dropped to prevent synthetic score inflation.
        - **Cleaned Total:** 5,160 unique messages (4,518 Ham, 642 Spam).

        #### 2. NLP Preprocessing Pipeline
        - **Case Normalization:** Uniform lowercase conversion.
        - **Punctuation & Noise Stripping:** Regular expression removal of URLs, email handles, and non-alphanumeric noise.
        - **Stopword Filtering:** Removal of high-frequency English stopwords.
        - **Stemming:** Morphological word reduction using Porter Stemmer.
        """)

    with col_m2:
        st.markdown("""
        #### 3. Feature Extraction
        - **Vectorization:** TF-IDF with sublinear term-frequency scaling.
        - **Vocabulary:** 4,000 maximum n-gram features (unigrams + bigrams).
        - **Bernoulli Handling:** Strict binary presence/absence indicator representation.

        #### 4. Data Leakage Prevention
        - **Stratified Splitting:** 80% train (4,128 messages), 20% test (1,032 messages).
        - **Pipelines:** Feature fitting and vocabulary generation restricted strictly to training partitions.
        - **Tuning:** Stratified 5-Fold Cross Validation via GridSearchCV.
        """)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small style='color: #64748b;'>Developed for Research Methodologies (RM) · 7th Semester B.Tech Project in CSE · Built with Python, Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True
)
