"""
SIGNAL: SMS INTELLIGENCE LAB
SMS Spam Detection & Factorial Machine Learning Workstation
Subject: Research Methodologies (RM) - 7th Sem B.Tech CSE
"""

import os
import sys
import time
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
    page_title="SIGNAL — SMS Intelligence Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Editorial Swiss Design System (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    /* Global Root Resets */
    html, body, p, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #17232D;
    }

    /* Background overrides */
    .stApp {
        background-color: #F5F4EF;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        color: #17232D !important;
        letter-spacing: -0.02em !important;
    }

    /* Top Application Header */
    .signal-header {
        background: #FFFFFF;
        border-bottom: 1px solid #D9DFE1;
        padding: 16px 28px;
        margin: -4rem -4rem 1.5rem -4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .signal-header-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        color: #17232D;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .signal-header-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #68747B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EAECE9 !important;
        border-right: 1px solid #D9DFE1 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.90rem;
        color: #17232D;
        padding: 6px 10px;
        border-radius: 6px;
    }

    /* Editorial Cards */
    .editorial-card {
        background: #FFFFFF;
        border: 1px solid #D9DFE1;
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }

    /* Metric Display Units */
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid #D9DFE1;
        border-radius: 6px;
        padding: 14px 18px;
        height: 100%;
    }
    .metric-panel-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #68747B;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-panel-value {
        font-family: 'Manrope', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        color: #17232D;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-panel-caption {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #68747B;
        margin-top: 4px;
    }

    /* Verdict Display Badges */
    .verdict-box-spam {
        background: #FFFFFF;
        border: 2px solid #E66B36;
        border-left: 8px solid #E66B36;
        border-radius: 6px;
        padding: 20px 24px;
    }
    .verdict-box-ham {
        background: #FFFFFF;
        border: 2px solid #27866B;
        border-left: 8px solid #27866B;
        border-radius: 6px;
        padding: 20px 24px;
    }
    .verdict-badge-spam {
        display: inline-block;
        background: #E66B36;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .verdict-badge-ham {
        display: inline-block;
        background: #27866B;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    /* Empty state box */
    .empty-state-box {
        background: #FFFFFF;
        border: 1px dashed #D9DFE1;
        border-radius: 8px;
        padding: 48px 24px;
        text-align: center;
        color: #68747B;
    }

    /* Monospace Tech Tag */
    .tech-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        background: #EAECE9;
        color: #17232D;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #D9DFE1;
    }

    /* Data table header styling */
    .dataframe {
        font-size: 0.85rem !important;
        border: 1px solid #D9DFE1 !important;
    }

    /* Form & Selectbox Styling (BaseWeb, ARIA & Streamlit Overrides to guarantee 100% High Contrast) */
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSelectbox"] > div > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input {
        background-color: #FFFFFF !important;
        color: #17232D !important;
        border-color: #D9DFE1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.90rem !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div {
        color: #17232D !important;
    }
    div[data-baseweb="select"] svg,
    div[data-testid="stSelectbox"] svg {
        fill: #17232D !important;
        background: transparent !important;
    }
    /* Popovers, menus and options (BaseWeb + ARIA role-based) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    ul[data-baseweb="menu"] > li,
    [role="listbox"],
    [role="listbox"] > li,
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #17232D !important;
        border-color: #D9DFE1 !important;
    }
    ul[data-baseweb="menu"],
    [role="listbox"] {
        border: 1px solid #D9DFE1 !important;
        border-radius: 6px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12) !important;
    }
    li[data-baseweb="menu-item"],
    li[data-baseweb="menu-item"] > div,
    [role="option"],
    [role="option"] > div,
    [role="option"] span {
        background-color: #FFFFFF !important;
        color: #17232D !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }
    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"]:hover > div,
    li[data-baseweb="menu-item"][aria-selected="true"],
    li[data-baseweb="menu-item"][aria-selected="true"] > div,
    [role="option"]:hover,
    [role="option"]:hover > div,
    [role="option"][aria-selected="true"],
    [role="option"][aria-selected="true"] > div {
        background-color: #EAECE9 !important;
        color: #17232D !important;
        font-weight: 600 !important;
    }

    /* Textarea adjustment */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #17232D !important;
        border: 1px solid #D9DFE1 !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #17232D !important;
        box-shadow: none !important;
    }

    /* Primary Action Buttons */
    .stButton button[kind="primary"],
    .stButton button[kind="primary"] * {
        background-color: #17232D !important;
        color: #FFFFFF !important;
        border: 1px solid #17232D !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[kind="primary"]:hover * {
        background-color: #E66B36 !important;
        border-color: #E66B36 !important;
        color: #FFFFFF !important;
    }

    .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #17232D !important;
        border: 1px solid #D9DFE1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Data and Model Persistence Loaders ---
MODEL_FILES = {
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
def load_pipeline(feature_tech: str, algorithm_name: str):
    filename = MODEL_FILES.get((feature_tech, algorithm_name), "best_model.pkl")
    path = os.path.join(os.path.dirname(__file__), "models", filename)
    if os.path.exists(path):
        return joblib.load(path)
    fallback = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    if os.path.exists(fallback):
        return joblib.load(fallback)
    raise FileNotFoundError("Trained pipeline models not found. Please run src/evaluation.py first.")


@st.cache_data
def load_evaluation_metrics():
    csv_path = os.path.join(os.path.dirname(__file__), "reports", "results", "results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_corpus_sample(n=10):
    from data_loader import load_raw_data, clean_data
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    return clean.head(n)


# --- Sidebar Navigation & Brand Shell ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #D9DFE1; margin-bottom: 20px;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #E66B36; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
            TELECOM SECURITY LAB
        </div>
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.5rem; font-weight: 800; color: #17232D; letter-spacing: -0.03em;">
            SIGNAL
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.80rem; color: #68747B; margin-top: 2px;">
            SMS Intelligence & Spam Detection Workstation
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "WORKSTATION NAVIGATION",
        [
            "Message Analyzer",
            "Model Performance",
            "Dataset Explorer",
            "Research Comparison",
            "About Project"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<br><hr style='border: none; border-top: 1px solid #D9DFE1; margin: 20px 0;'>", unsafe_allow_html=True)

    # Bottom Telemetry Panel
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #68747B; line-height: 1.8;">
        <div><strong>CORPUS:</strong> UCI SMS SPAM (N=5,160)</div>
        <div><strong>SPLIT:</strong> STRATIFIED 80:20</div>
        <div><strong>LATENCY:</strong> &le; 12 MS (LINEAR SVM)</div>
        <div><strong>PIPELINES:</strong> 8 REPRODUCIBLE MODELS</div>
        <div><strong>STATUS:</strong> <span style="color: #27866B; font-weight: 600;">ACTIVE / READY</span></div>
        <div style="margin-top: 8px; color: #94a3b8;">BUILD: v2.4-RM-ENG</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: MESSAGE ANALYZER
# ==============================================================================
if nav_selection == "Message Analyzer":
    # Header Bar
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                Primary Workspace
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">Message Analyzer</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                Evaluate individual SMS strings in real-time using selected NLP vectorization and classification models.
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            WORKSTATION ONLINE<br>
            <span style="color: #27866B; font-weight: 600;">● ZERO DATA LEAKAGE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("### Analyse an SMS")
        st.markdown("<p style='font-size: 0.88rem; color: #68747B; margin-top: -8px;'>Configure model pipeline, insert SMS text, and execute classification.</p>", unsafe_allow_html=True)

        # Pipeline Configuration Bar
        c_pipe1, c_pipe2 = st.columns(2)
        with c_pipe1:
            feature_choice = st.selectbox(
                "Feature Extraction",
                ["TF-IDF", "CountVectorizer"],
                index=0,
                help="TF-IDF applies sublinear document weighting; CountVectorizer computes raw token frequency."
            )
        with c_pipe2:
            model_choice = st.selectbox(
                "Classifier Model",
                [
                    "Linear Support Vector Machine",
                    "Bernoulli Naive Bayes",
                    "Logistic Regression",
                    "Multinomial Naive Bayes"
                ],
                index=0,
                help="Supervised learning classifier architecture."
            )

        # Preset Academic Samples
        sample_catalog = {
            "-- Select sample or compose custom text --": "",
            "🚨 Phishing / Banking Urgent Claim": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Claim code KL341.",
            "🚨 Lottery Cash Offer": "URGENT! Your mobile account has won a guaranteed £1000 cash prize. Reply CLAIM to 87121 immediately. Terms apply £1.50/msg.",
            "✅ Peer Meeting Notice": "Hey mate, are you coming over to the university library this afternoon to finish the Research Methodology slides?",
            "✅ Academic Department Circular": "Dear Students, please submit your 7th Semester RM lab file by Friday 5 PM on the department portal."
        }

        selected_sample_key = st.selectbox(
            "Load Test Case",
            list(sample_catalog.keys())
        )
        sample_text = sample_catalog[selected_sample_key]

        # Session state for message input
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""

        if sample_text:
            st.session_state["user_text"] = sample_text

        text_input = st.text_area(
            "SMS Message Text",
            value=st.session_state["user_text"],
            placeholder="Type or paste SMS text here to inspect...",
            height=130,
            key="sms_raw_input"
        )

        char_len = len(text_input)
        word_len = len(text_input.split())

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #68747B; margin-top: -6px; margin-bottom: 14px;">
            <span>CHARACTERS: {char_len} / 160</span>
            <span>WORDS: {word_len}</span>
        </div>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            analyze_clicked = st.button("Analyze Message", type="primary", use_container_width=True)
        with btn_col2:
            clear_clicked = st.button("Clear", type="secondary", use_container_width=True)

        if clear_clicked:
            st.session_state["user_text"] = ""
            st.rerun()

    with col_output:
        st.markdown("### Classification Verdict")
        st.markdown("<p style='font-size: 0.88rem; color: #68747B; margin-top: -8px;'>Real-time inference telemetry and probability distribution.</p>", unsafe_allow_html=True)

        if analyze_clicked:
            if not text_input.strip():
                st.warning("Please enter an SMS text message to analyze.")
            else:
                with st.spinner("Processing text normalization & linear inference..."):
                    time.sleep(0.08)  # Subtle realistic analysis transition
                    pipeline = load_pipeline(feature_choice, model_choice)
                    start_t = time.perf_counter()
                    pred = pipeline.predict([text_input])[0]
                    latency_ms = (time.perf_counter() - start_t) * 1000

                    probs = None
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            probs = pipeline.predict_proba([text_input])[0]
                        except Exception:
                            pass

                # Render Verdict Card
                if pred == 1:
                    st.markdown(f"""
                    <div class="verdict-box-spam">
                        <span class="verdict-badge-spam">SPAM DETECTED</span>
                        <h2 style="color: #E66B36; margin: 4px 0 8px 0; font-size: 1.6rem;">THREAT LEVEL: HIGH</h2>
                        <p style="color: #17232D; font-size: 0.90rem; margin: 0; line-height: 1.5;">
                            This message matches characteristic fraudulent patterns, unsolicited marketing, or credential phishing hooks.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-box-ham">
                        <span class="verdict-badge-ham">SAFE (HAM)</span>
                        <h2 style="color: #27866B; margin: 4px 0 8px 0; font-size: 1.6rem;">AUTHENTIC MESSAGE</h2>
                        <p style="color: #17232D; font-size: 0.90rem; margin: 0; line-height: 1.5;">
                            Linguistic analysis indicates regular conversational or informational message content. No unauthorized threat triggers detected.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Telemetry Grid
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Pipeline</div>
                        <div style="font-size: 0.88rem; font-weight: 700; color: #17232D;">{model_choice.split()[0]}</div>
                        <div class="metric-panel-caption">{feature_choice}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t2:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Inference Latency</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #17232D;">{latency_ms:.1f} ms</div>
                        <div class="metric-panel-caption">Single Sample</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t3:
                    prob_val = f"{probs[1]*100:.1f}%" if probs is not None else "N/A"
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Spam Probability</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {'#E66B36' if pred==1 else '#27866B'};">{prob_val}</div>
                        <div class="metric-panel-caption">Platt Calibrated</div>
                    </div>
                    """, unsafe_allow_html=True)

                if probs is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("**Classification Margin Distribution:**")
                    st.progress(float(probs[1]))
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
                        <span>HAM: {probs[0]*100:.1f}%</span>
                        <span>SPAM: {probs[1]*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Normalized Tokens
                st.markdown("<br>", unsafe_allow_html=True)
                cleaned_tokens = preprocess_text(text_input)
                st.markdown("**NLP Root Token Representation:**")
                st.markdown(f"""
                <div class="tech-tag" style="display: block; padding: 10px 14px; background: #FFFFFF; font-size: 0.82rem; border: 1px solid #D9DFE1; color: #17232D; margin-bottom: 16px;">
                    {cleaned_tokens or '[All tokens filtered as standard English stopwords]'}
                </div>
                """, unsafe_allow_html=True)

                # --- Cross-Model Consensus Radar (All 8 Pipelines) ---
                st.markdown("##### 🔍 Multi-Model Consensus (All 8 Pipelines)")
                consensus_rows = []
                spam_votes = 0
                for (feat, algo), filename in MODEL_FILES.items():
                    p = load_pipeline(feat, algo)
                    p_pred = p.predict([text_input])[0]
                    p_prob = p.predict_proba([text_input])[0][1] if hasattr(p, "predict_proba") else None
                    if p_pred == 1:
                        spam_votes += 1
                    consensus_rows.append({
                        "Algorithm": algo,
                        "Vectorizer": feat,
                        "Verdict": "🚨 SPAM" if p_pred == 1 else "✅ SAFE",
                        "Spam Probability": f"{p_prob*100:.1f}%" if p_prob is not None else "N/A"
                    })

                if spam_votes >= 5:
                    status_html = f"<span style='color: #E66B36; font-weight: 700;'>🚨 HIGH THREAT CONSENSUS ({spam_votes}/8 Models Flagged SPAM)</span>"
                elif spam_votes > 0:
                    status_html = f"<span style='color: #E66B36; font-weight: 700;'>⚠️ DISCORDANT / PARTIAL THREAT ({spam_votes}/8 Models Flagged SPAM)</span>"
                else:
                    status_html = f"<span style='color: #27866B; font-weight: 700;'>✅ UNANIMOUS SAFE (0/8 Models Flagged SPAM)</span>"

                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #D9DFE1; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; font-size: 0.84rem; line-height: 1.6;">
                    <div><strong>Consensus Radar:</strong> {status_html}</div>
                    <div style="color: #68747B; font-size: 0.78rem; margin-top: 2px;">
                        Models diverge on conversational smishing: Bernoulli NB triggers on binary keywords ('fee', 'pay'), while TF-IDF Linear SVM discounts them due to frequent benign terms.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                df_consensus = pd.DataFrame(consensus_rows)
                st.dataframe(df_consensus, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="empty-state-box">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">📡</div>
                <div style="font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.05rem; color: #17232D; margin-bottom: 6px;">
                    Workstation Awaiting Input
                </div>
                <div style="font-size: 0.85rem; max-width: 320px; margin: 0 auto;">
                    Select an academic test sample or paste SMS text on the left, then click <strong>Analyze Message</strong> to initiate evaluation.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: MODEL PERFORMANCE
# ==============================================================================
elif nav_selection == "Model Performance":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                Empirical Evaluation
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">Model Performance</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                Quantitative evaluation across 8 experimental combinations (2 Feature Extractors &times; 4 Classifiers) on 1,032 test messages.
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            VALIDATION METHOD<br>
            <span style="color: #27866B; font-weight: 600;">STRATIFIED 80:20 TEST SET</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Peak Accuracy</div>
            <div class="metric-panel-value">98.16%</div>
            <div class="metric-panel-caption">Linear SVM (E4, E8)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Peak Precision</div>
            <div class="metric-panel-value" style="color: #27866B;">97.39%</div>
            <div class="metric-panel-caption">CountVec + Linear SVM (3 FP)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Peak Recall</div>
            <div class="metric-panel-value" style="color: #E66B36;">89.06%</div>
            <div class="metric-panel-caption">CountVec + MNB (114 / 128 Spam)</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Top F1-Score</div>
            <div class="metric-panel-value">92.24%</div>
            <div class="metric-panel-caption">TF-IDF + Linear SVM (Balanced)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full Factorial Table
    st.markdown("### Factorial Evaluation Matrix (All 8 Experiments)")
    df_results = load_evaluation_metrics()
    if df_results is not None:
        filter_tech = st.radio(
            "Filter Experiments:",
            ["All 8 Pipelines", "TF-IDF Only", "CountVectorizer Only"],
            horizontal=True
        )

        display_df = df_results.copy()
        if filter_tech == "TF-IDF Only":
            display_df = display_df[display_df["Feature_Extraction"] == "TF-IDF"]
        elif filter_tech == "CountVectorizer Only":
            display_df = display_df[display_df["Feature_Extraction"] == "CountVectorizer"]

        # Format percentages
        for c in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"

        st.dataframe(
            display_df[[
                "Experiment", "Feature_Extraction", "Algorithm",
                "Accuracy", "Precision", "Recall", "F1-Score",
                "FP (False Alarm)", "FN (Missed Spam)"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Results file reports/results/results_summary.csv not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    st.markdown("### Performance Visualizations")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        f1_chart = os.path.join(os.path.dirname(__file__), "reports", "figures", "f1_comparison.png")
        if os.path.exists(f1_chart):
            st.image(f1_chart, caption="Figure 1: Spam F1-Score Comparison across Experiments", use_container_width=True)

    with col_g2:
        feat_chart = os.path.join(os.path.dirname(__file__), "reports", "figures", "feature_extraction_comparison.png")
        if os.path.exists(feat_chart):
            st.image(feat_chart, caption="Figure 2: TF-IDF vs. CountVectorizer Aggregated Comparison", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 8-Panel Confusion Matrix Grid")
    all_cm = os.path.join(os.path.dirname(__file__), "reports", "figures", "confusion_matrices_all8.png")
    if os.path.exists(all_cm):
        st.image(all_cm, caption="Figure 3: Confusion Matrices for All 8 Experimental Pipelines (904 Ham, 128 Spam)", use_container_width=True)


# ==============================================================================
# PAGE 3: DATASET EXPLORER
# ==============================================================================
elif nav_selection == "Dataset Explorer":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                Corpus Diagnostics
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">Dataset Explorer</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                Official UCI SMS Spam Collection dataset properties, deduplication audit, and class distributions.
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            PROVENANCE<br>
            <span style="color: #27866B; font-weight: 600;">UCI REPOSITORY ID: 228</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Unique</div>
            <div class="metric-panel-value">5,160</div>
            <div class="metric-panel-caption">Cleaned Messages</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Ham (Safe)</div>
            <div class="metric-panel-value" style="color: #27866B;">4,518</div>
            <div class="metric-panel-caption">87.56% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Spam (Threat)</div>
            <div class="metric-panel-value" style="color: #E66B36;">642</div>
            <div class="metric-panel-caption">12.44% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Deduplicated</div>
            <div class="metric-panel-value">414</div>
            <div class="metric-panel-caption">Duplicates Dropped</div>
        </div>
        """, unsafe_allow_html=True)
    with d5:
        st.markdown("""
        <div class="metric-panel">
            <div class="metric-panel-label">Missing Values</div>
            <div class="metric-panel-value" style="color: #27866B;">0</div>
            <div class="metric-panel-caption">Complete Records</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cd1, col_cd2 = st.columns(2)
    with col_cd1:
        st.markdown("### Natural Class Distribution")
        class_fig = os.path.join(os.path.dirname(__file__), "reports", "figures", "class_distribution.png")
        if os.path.exists(class_fig):
            st.image(class_fig, use_container_width=True)

    with col_cd2:
        st.markdown("### Message Length Characteristics")
        len_fig = os.path.join(os.path.dirname(__file__), "reports", "figures", "message_length_distribution.png")
        if os.path.exists(len_fig):
            st.image(len_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Preprocessing Transformation Impact (Corpus Level)")
    df_prep = pd.DataFrame([
        {"Corpus Metric": "Character Volume", "Raw Corpus": "408,360", "Preprocessed Root": "246,088", "Reduction": "-39.74%"},
        {"Corpus Metric": "Word Tokens", "Raw Corpus": "79,521", "Preprocessed Root": "44,883", "Reduction": "-43.56%"},
        {"Corpus Metric": "Unique Vocabulary", "Raw Corpus": "13,627", "Preprocessed Root": "8,001", "Reduction": "-41.29%"}
    ])
    st.dataframe(df_prep, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Cleaned Corpus Sample Records")
    samples_df = load_corpus_sample(10)
    st.dataframe(samples_df[["label", "message"]], use_container_width=True)


# ==============================================================================
# PAGE 4: RESEARCH COMPARISON
# ==============================================================================
elif nav_selection == "Research Comparison":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                Benchmark Validation
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">Research Comparison</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                Direct comparative variance analysis against IEEE TENCON 2023 (Robert G. de Luna et al.).
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            REFERENCE VENUE<br>
            <span style="color: #27866B; font-weight: 600;">IEEE TENCON 2023</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Published IEEE Reference Benchmarks (Table VII in de Luna et al.)")
    st.markdown("""
    *Reference Paper:* **A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)**  
    *Authors:* Robert G. de Luna, Karina L. Enriquez, Aira Mae Española, et al. (Polytechnic University of the Philippines).
    """)

    df_paper_tbl = pd.DataFrame([
        {
            "Algorithm": "Bernoulli Naive Bayes (BNB)",
            "IEEE TENCON 2023 Reported": "Acc: 96.6333% | Prec: 98.4431% | Rec: 95.5174% | F1: 92.7835%",
            "Our Factorial Result (CountVec)": "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%",
            "Our Factorial Result (TF-IDF)": "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%",
            "Academic Finding": "Top model in the IEEE study. Binarized presence/absence modeling yields identical high accuracy across both feature spaces."
        },
        {
            "Algorithm": "Random Forest Classifier (RFC)",
            "IEEE TENCON 2023 Reported": "Acc: 96.0297% | Prec: 98.5228% | Rec: 94.8699% | F1: 91.1408%",
            "Our Factorial Result (CountVec)": "N/A (Focus on 4 core linear/probabilistic models)",
            "Our Factorial Result (TF-IDF)": "N/A (Focus on 4 core linear/probabilistic models)",
            "Academic Finding": "Important Correction: 96.0297% in Table VII belongs to Random Forest, NOT Multinomial Naive Bayes."
        },
        {
            "Algorithm": "Logistic Regression (LR)",
            "IEEE TENCON 2023 Reported": "Acc: 95.8709% | Prec: 98.7711% | Rec: 93.7399% | F1: 89.2555%",
            "Our Factorial Result (CountVec)": "Acc: 97.87% | Prec: 97.32% | Rec: 85.16% | F1: 90.83%",
            "Our Factorial Result (TF-IDF)": "Acc: 97.97% | Prec: 96.52% | Rec: 86.72% | F1: 91.36%",
            "Academic Finding": "Solid linear baseline across both investigations. Sublinear TF-IDF scaling improved F1 to 91.36%."
        },
        {
            "Algorithm": "Linear Support Vector Machine",
            "IEEE TENCON 2023 Reported": "Acc: 94.5684% (Default CV, unoptimized in Table VII)",
            "Our Factorial Result (CountVec)": "Acc: 98.16% | Prec: 97.39% | Rec: 87.50% | F1: 92.18%",
            "Our Factorial Result (TF-IDF)": "Acc: 98.16% | Prec: 96.58% | Rec: 88.28% | F1: 92.24%",
            "Academic Finding": "Omitted from Table VII optimization in the IEEE study. Hyperparameter tuning elevated Linear SVM to our #1 top model."
        }
    ])
    st.dataframe(df_paper_tbl, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    paper_chart = os.path.join(os.path.dirname(__file__), "reports", "figures", "paper_vs_our_comparison.png")
    if os.path.exists(paper_chart):
        st.image(paper_chart, caption="Figure 4: Comparative Accuracy across IEEE TENCON 2023 and Our Dual Feature Extractions", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    ### Critical Methodological Divergences:
    1. **Artificial Partitioning vs. Authentic Imbalance:**  
       The IEEE authors undersampled ham messages to force an approximate **60:40 Ham-to-Spam partition** (1,930 ham, 1,218 spam). Our project retained the real-world **87.6% Ham : 12.4% Spam** natural distribution. Artificially increasing spam density inflates model recall (95.5% vs 88.3%) because spam patterns appear at triple their real-world frequency.
    2. **Feature Extraction Philosophy:**  
       The IEEE authors selected CountVectorizer based on an initial MNB test. Our factorial investigation demonstrates that **TF-IDF with sublinear scaling achieves superior performance for Linear SVM and Logistic Regression**, mitigating high-frequency generic word noise.
    3. **Controlled Equivalence Disclaimer:**  
       This comparison serves as an **external benchmark validation**, not a direct controlled ablation, due to distinct regional corpora and sampling procedures.
    """)


# ==============================================================================
# PAGE 5: ABOUT PROJECT
# ==============================================================================
elif nav_selection == "About Project":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                Academic Specifications
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">About Project</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                B.Tech Computer Science and Engineering · Research Methodologies (RM) Subject.
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            INSTITUTIONAL PROFILE<br>
            <span style="color: #27866B; font-weight: 600;">SEMESTER VII CSE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="editorial-card">
        <h3 style="margin-top: 0;">Abstract</h3>
        <p style="font-size: 0.90rem; line-height: 1.6; color: #17232D;">
            Short Message Service (SMS) remains a critical telecommunication channel, yet mobile subscribers are increasingly targeted by fraudulent text spam. This study presents a comparative empirical investigation of supervised Machine Learning algorithms using the official UCI SMS Spam Collection dataset. We implement a 2 &times; 4 full factorial experimental design evaluating two feature extraction methods (TF-IDF and CountVectorizer) across four classification algorithms: Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, and Linear Support Vector Machine. Linear SVM with TF-IDF achieved the highest overall performance with 98.16% Accuracy and 92.24% F1-Score, while Bernoulli Naive Bayes attained 97.37% precision with only 3 false alarms out of 904 legitimate messages.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_ab1, c_ab2 = st.columns(2)
    with c_ab1:
        st.markdown("""
        <div class="editorial-card">
            <h4 style="margin-top: 0;">Research Objectives</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #17232D; padding-left: 18px;">
                <li>Execute an end-to-end NLP & machine learning classification pipeline.</li>
                <li>Implement a 2 &times; 4 full factorial comparison of TF-IDF vs. CountVectorizer.</li>
                <li>Enforce mathematical binarization for Bernoulli Naive Bayes.</li>
                <li>Prevent data leakage via isolated Scikit-learn Pipeline constructs.</li>
                <li>Conduct external benchmark validation against IEEE TENCON 2023.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_ab2:
        st.markdown("""
        <div class="editorial-card">
            <h4 style="margin-top: 0;">Technology Stack</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #17232D; padding-left: 18px;">
                <li><strong>Language:</strong> Python 3.14</li>
                <li><strong>Core ML:</strong> Scikit-learn 1.9</li>
                <li><strong>NLP & Tokenization:</strong> NLTK 3.10 (Porter Stemmer)</li>
                <li><strong>Data Processing:</strong> Pandas 3.0 & NumPy 2.4</li>
                <li><strong>Workstation UI:</strong> Streamlit 1.61</li>
                <li><strong>Visualizations:</strong> Seaborn & Matplotlib</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="editorial-card">
        <h4 style="margin-top: 0;">Academic References</h4>
        <ol style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; line-height: 1.7; color: #17232D; padding-left: 18px;">
            <li>R. G. de Luna, K. L. Enriquez, A. M. Española, et al., "A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)," in Proc. 2023 IEEE Region 10 Conference (TENCON), Chiang Mai, Thailand, 2023, pp. 52–57.</li>
            <li>T. A. Almeida, J. M. G. Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," in Proc. 11th ACM Symposium on Document Engineering (DocEng), 2011, pp. 259–262.</li>
            <li>N. Krishnaveni and V. Radha, "Comparison of Naïve Bayes and SVM Classifier for Detection of Spam SMS using Natural Language Processing," Int. J. Semantic Comput., vol. 11, no. 2, pp. 2260–2265, 2021.</li>
            <li>N. Kudupudi and S. Nair, "Spam message detection using logistic regression," Int. J. Adv. Comput. Sci. Appl., vol. 9, no. 9, pp. 815–818, 2021.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div style="text-align: center; font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #68747B; border-top: 1px solid #D9DFE1; padding-top: 18px; margin-top: 36px;">
    SIGNAL · SMS INTELLIGENCE LAB · 7TH SEM B.TECH CSE · RESEARCH METHODOLOGIES
</div>
""", unsafe_allow_html=True)
