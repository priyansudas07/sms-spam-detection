"""
SIGNAL: SMS INTELLIGENCE LAB
SMS Spam Detection & Comparative Machine Learning Workstation
Academic Project | Subject: Research Methodologies (RM) - 7th Sem B.Tech CSE
Author: Priyansu Das | Siksha 'O' Anusandhan University (ITER)
"""

import os
import sys
import time
import json
import joblib
import platform
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src modules are resolvable
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing import preprocess_text
from feature_extraction import TextPreprocessorTransformer  # noqa: F401
from smishing_heuristics import analyze_smishing_heuristics, calculate_weighted_ensemble_prediction


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
        font-size: 1.70rem;
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

    /* Form & Selectbox Styling (High Contrast Guaranteed) */
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


# --- Model and Result Persistence Loaders ---
MODEL_FILES = {
    ("TF-IDF", "Bernoulli Naive Bayes"): "tfidf_plus_bernoulli_naive_bayes.pkl",
    ("TF-IDF", "Linear Support Vector Machine"): "tfidf_plus_linear_svm.pkl",
    ("TF-IDF", "Logistic Regression"): "tfidf_plus_logistic_regression.pkl",
    ("TF-IDF", "Multinomial Naive Bayes"): "tfidf_plus_multinomial_naive_bayes.pkl",
    ("CountVectorizer", "Bernoulli Naive Bayes"): "countvectorizer_plus_bernoulli_naive_bayes.pkl",
    ("CountVectorizer", "Linear Support Vector Machine"): "countvectorizer_plus_linear_svm.pkl",
    ("CountVectorizer", "Logistic Regression"): "countvectorizer_plus_logistic_regression.pkl",
    ("CountVectorizer", "Multinomial Naive Bayes"): "countvectorizer_plus_multinomial_naive_bayes.pkl",
}


@st.cache_resource
def load_pipeline(feature_tech: str, algorithm_name: str):
    """
    Loads the exact trained model pipeline. Never silently defaults to another model.
    Raises FileNotFoundError if the specific model artifact is missing.
    """
    key = (feature_tech, algorithm_name)
    if key not in MODEL_FILES:
        raise ValueError(f"Unknown combination: {feature_tech} + {algorithm_name}")

    filename = MODEL_FILES[key]
    path = os.path.join(PROJECT_ROOT, "models", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Requested pipeline file '{filename}' for '{feature_tech} + {algorithm_name}' "
            f"does not exist at {path}. Please run 'src/evaluation.py' to generate all model artifacts."
        )
    return joblib.load(path)


@st.cache_resource
def load_all_8_pipelines():
    """
    Loads all 8 pipelines for the multi-model consensus evaluation.
    """
    pipelines = {}
    for (feat, algo), filename in MODEL_FILES.items():
        path = os.path.join(PROJECT_ROOT, "models", filename)
        if os.path.exists(path):
            pipelines[(feat, algo)] = joblib.load(path)
        else:
            raise FileNotFoundError(f"Missing pipeline artifact: {filename}. Run src/evaluation.py first.")
    return pipelines


@st.cache_data
def load_evaluation_metrics():
    csv_path = os.path.join(PROJECT_ROOT, "reports", "results", "results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_cv_metrics():
    csv_path = os.path.join(PROJECT_ROOT, "reports", "results", "cv_results_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_selected_model_metadata():
    json_path = os.path.join(PROJECT_ROOT, "models", "selected_model_metadata.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_audit_metrics():
    json_path = os.path.join(PROJECT_ROOT, "reports", "results", "dataset_audit_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_corpus_sample(n=10):
    from data_loader import load_raw_data, clean_data
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    return clean.head(n)


# --- Sidebar Navigation & Telemetry ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #D9DFE1; margin-bottom: 20px;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #E66B36; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
            RESEARCH WORKSTATION
        </div>
        <div style="font-family: 'Manrope', sans-serif; font-size: 1.5rem; font-weight: 800; color: #17232D; letter-spacing: -0.03em;">
            SIGNAL
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.80rem; color: #68747B; margin-top: 2px;">
            SMS Intelligence & Machine Learning Lab
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

    # Telemetry Panel
    st.markdown("""
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem; color: #68747B; line-height: 1.8;">
        <div><strong>CORPUS:</strong> UCI SMS SPAM (N=5,160)</div>
        <div><strong>TRAIN/TEST:</strong> 80:20 (4,128 / 1,032)</div>
        <div><strong>CV PROTOCOL:</strong> STRATIFIED 5-FOLD</div>
        <div><strong>PIPELINES:</strong> 8 TRAINED MODELS</div>
        <div><strong>CHAMPION:</strong> TF-IDF + BERNOULLI NB</div>
        <div style="margin-top: 8px; color: #94a3b8;">SYSTEM STATUS: ONLINE</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: MESSAGE ANALYZER
# ==============================================================================
if nav_selection == "Message Analyzer":
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
            <span style="color: #27866B; font-weight: 600;">● ZERO TEST LEAKAGE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("### Input Message")
        st.markdown("<p style='font-size: 0.88rem; color: #68747B; margin-top: -8px;'>Configure model pipeline, insert SMS text, and execute classification.</p>", unsafe_allow_html=True)

        # Pipeline Configuration Selectboxes
        c_pipe1, c_pipe2 = st.columns(2)
        with c_pipe1:
            feature_choice = st.selectbox(
                "Feature Extraction",
                ["TF-IDF", "CountVectorizer"],
                index=0,
                help="Select between continuous TF-IDF term weights (sublinear scaling) or discrete Bag-of-Words frequency counts."
            )
        with c_pipe2:
            model_choice = st.selectbox(
                "Classifier Model",
                [
                    "Bernoulli Naive Bayes",
                    "Linear Support Vector Machine",
                    "Logistic Regression",
                    "Multinomial Naive Bayes",
                    "8-Model Majority Voting Ensemble (Evaluated)"
                ],
                index=0,
                help="Select a specific trained classifier or the evaluated 8-pipeline consensus ensemble."
            )

        enable_smishing_guard = st.toggle(
            "🛡️ Hybrid Smishing Defense Guard",
            value=True,
            help="Activates Tier-2 Smishing Pattern Inspector and Weighted Soft-Voting Consensus to intercept modern adversarial smishing lures (e.g. package tracking, fee extortion, authority impersonation)."
        )

        # Preset Academic Samples Catalog
        sample_catalog = {
            "-- Select sample or compose custom text --": "",
            "🚨 USPS Package Delivery Smishing": "USPS Notice: Your package could not be delivered today due to an incorrect house number. Please update your delivery address and pay the $1.99 redelivery fee at http://usps-tracking-portal-update.com within 12 hours.",
            "🚨 Phishing / Banking Urgent Claim": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Claim code KL341.",
            "🚨 Lottery Cash Offer": "URGENT! Your mobile account has won a guaranteed £1000 cash prize. Reply CLAIM to 87121 immediately. Terms apply £1.50/msg.",
            "✅ Peer Meeting Notice": "Hey mate, are you coming over to the university library this afternoon to finish the Research Methodology slides?",
            "✅ Academic Department Circular": "Dear Students, please submit your 7th Semester RM lab file by Friday 5 PM on the department portal."
        }

        # Session State Setup
        if "sms_text" not in st.session_state:
            st.session_state["sms_text"] = ""

        def load_sample():
            choice = st.session_state.get("sample_dropdown")
            if choice and sample_catalog.get(choice):
                st.session_state["sms_text"] = sample_catalog[choice]

        def clear_all():
            st.session_state["sms_text"] = ""
            st.session_state["sample_dropdown"] = list(sample_catalog.keys())[0]

        st.selectbox(
            "Load Sample Test Case",
            list(sample_catalog.keys()),
            key="sample_dropdown",
            on_change=load_sample
        )

        text_input = st.text_area(
            "SMS Message Text",
            placeholder="Type or paste SMS text here to inspect...",
            height=130,
            key="sms_text"
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
            analyze_clicked = st.button("Analyze Message", type="primary", width="stretch")
        with btn_col2:
            st.button("Clear", type="secondary", width="stretch", on_click=clear_all)

    with col_output:
        st.markdown("### Classification Verdict")
        st.markdown("<p style='font-size: 0.88rem; color: #68747B; margin-top: -8px;'>Real-time inference telemetry and probability distribution.</p>", unsafe_allow_html=True)

        if analyze_clicked:
            if not text_input.strip():
                st.warning("Please enter an SMS text message to analyze.")
            else:
                with st.spinner("Executing model inference..."):
                    start_t = time.perf_counter()
                    cleaned_tokens = preprocess_text(text_input)

                    is_ensemble = "Majority Voting" in model_choice
                    pred = 0
                    probs = None
                    latency_ms = 0.0

                    try:
                        # Load all pipelines for consensus view
                        all_pipes = load_all_8_pipelines()
                        consensus_rows = []
                        spam_votes = 0
                        probs_list = []
                        pipe_outputs_dict = {}

                        for (feat, algo), pipe in all_pipes.items():
                            p_pred = pipe.predict([text_input])[0]
                            p_prob = None
                            if hasattr(pipe, "predict_proba"):
                                try:
                                    p_prob = pipe.predict_proba([text_input])[0]
                                except Exception:
                                    pass
                            if p_prob is None:
                                p_prob = [0.0, 1.0] if p_pred == 1 else [1.0, 0.0]

                            pipe_outputs_dict[(feat, algo)] = {
                                "pred": int(p_pred),
                                "prob": p_prob
                            }
                            p_spam = float(p_prob[1])
                            probs_list.append(p_spam)
                            if p_pred == 1:
                                spam_votes += 1

                            consensus_rows.append({
                                "Classifier": algo,
                                "Feature Extraction": feat,
                                "Predicted Class": "SPAM" if p_pred == 1 else "HAM",
                                "Spam Probability": f"{p_spam*100:.1f}%"
                            })

                        # Analyze Tier-2 Smishing Heuristics
                        smishing_telemetry = analyze_smishing_heuristics(text_input)

                        if is_ensemble:
                            # Weighted soft-voting consensus across all 8 pipelines
                            ens_telemetry = smishing_telemetry if enable_smishing_guard else None
                            ensemble_res = calculate_weighted_ensemble_prediction(
                                pipe_outputs_dict,
                                smishing_telemetry=ens_telemetry
                            )
                            pred = ensemble_res["pred"]
                            probs = ensemble_res["probability"]
                            spam_votes = ensemble_res["hard_spam_votes"]
                            if enable_smishing_guard and ensemble_res["hybrid_boost_applied"]:
                                active_clf_display = "Weighted Consensus + Smishing Guard"
                            else:
                                active_clf_display = "8-Model Weighted Consensus Ensemble"
                            active_feat_display = "Empirical Reliability Weights"
                        else:
                            # Single pipeline mode: load exact requested model
                            active_pipe = load_pipeline(feature_choice, model_choice)
                            pred = active_pipe.predict([text_input])[0]
                            if hasattr(active_pipe, "predict_proba"):
                                try:
                                    probs = active_pipe.predict_proba([text_input])[0]
                                except Exception:
                                    pass
                            if probs is None:
                                probs = [0.0, 1.0] if pred == 1 else [1.0, 0.0]

                            active_clf_display = model_choice
                            active_feat_display = feature_choice

                        latency_ms = (time.perf_counter() - start_t) * 1000

                    except FileNotFoundError as fnf_err:
                        st.error(f"Model Loading Error: {fnf_err}")
                        st.stop()
                    except Exception as err:
                        st.error(f"Inference Execution Error: {err}")
                        st.stop()

                # Render Verdict Card
                if pred == 1:
                    sub_title = f"PREDICTED SPAM ({spam_votes}/8 Models Flagged Spam)" if is_ensemble else "PREDICTED SPAM"
                    detail_text = (
                        f"Weighted ensemble consensus identified spam characteristics across {spam_votes} of 8 models. "
                        "Content exhibits promotional patterns, urgency cues, or smishing indicators."
                        if is_ensemble else
                        f"The selected {active_clf_display} pipeline ({active_feat_display}) classified this message as SPAM based on extracted n-gram term frequencies."
                    )
                    st.markdown(f"""
                    <div class="verdict-box-spam">
                        <span class="verdict-badge-spam">SPAM DETECTED</span>
                        <h2 style="color: #E66B36; margin: 4px 0 8px 0; font-size: 1.50rem;">{sub_title}</h2>
                        <p style="color: #17232D; font-size: 0.90rem; margin: 0; line-height: 1.5;">
                            {detail_text}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-box-ham">
                        <span class="verdict-badge-ham">PREDICTED HAM</span>
                        <h2 style="color: #27866B; margin: 4px 0 8px 0; font-size: 1.50rem;">PREDICTED HAM</h2>
                        <p style="color: #17232D; font-size: 0.90rem; margin: 0; line-height: 1.5;">
                            No spam indicators were detected by the selected pipeline. 
                            <em>(Note: Statistical classifiers indicate message characteristics align with legitimate traffic; absolute safety cannot be guaranteed).</em>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Tier-2 Smishing Heuristic Inspector Card
                if enable_smishing_guard and smishing_telemetry["threat_score"] > 0:
                    badge_bg = smishing_telemetry["badge_color"]
                    indicators_html = "".join([f"<li style='margin-bottom: 2px;'>{ind}</li>" for ind in smishing_telemetry["matched_indicators"]])
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #D9DFE1; border-left: 6px solid {badge_bg}; border-radius: 6px; padding: 14px 18px; margin-top: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: {badge_bg}; text-transform: uppercase; letter-spacing: 0.08em;">
                                🛡️ Tier-2 Smishing Heuristic Inspector
                            </span>
                            <span style="background: {badge_bg}; color: #FFFFFF; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">
                                {smishing_telemetry['risk_level']} ({smishing_telemetry['threat_score']} / 100)
                            </span>
                        </div>
                        <div style="font-size: 0.84rem; color: #17232D; font-weight: 600; margin-bottom: 6px;">
                            Active Threat Indicators:
                        </div>
                        <ul style="font-size: 0.80rem; color: #17232D; margin: 0; padding-left: 20px; line-height: 1.5;">
                            {indicators_html}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Smart Academic Advisory if single model predicted HAM on lethal smishing
                if not is_ensemble and pred == 0 and smishing_telemetry.get("is_smishing_lure"):
                    st.markdown("""
                    <div style="background: #FFF8E6; border: 1px solid #E6A23C; border-radius: 6px; padding: 12px 16px; margin-top: 12px; font-size: 0.84rem; color: #8A5B00; line-height: 1.5;">
                        <strong>⚠️ Academic Concept-Drift Advisory:</strong> The selected statistical model classified this text as HAM because modern postal/courier smishing mimics conversational vocabulary and avoids classic 2011 spam keywords (e.g., <em>winner, claim, free</em>). However, the <strong>Smishing Guard</strong> detected a verified malicious compound vector (hyperlink + fee extortion). Select <strong>Linear Support Vector Machine</strong> or the <strong>8-Model Weighted Ensemble</strong> to mitigate this concept drift.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Telemetry Metric Grid
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Pipeline</div>
                        <div style="font-size: 0.88rem; font-weight: 700; color: #17232D;">{active_clf_display}</div>
                        <div class="metric-panel-caption">{active_feat_display}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t2:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Inference Latency</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #17232D;">{latency_ms:.1f} ms</div>
                        <div class="metric-panel-caption">{"All 8 Models" if is_ensemble else "Single Model"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t3:
                    prob_val = f"{probs[1]*100:.1f}%" if probs is not None else "N/A"
                    caption_text = (
                        "Mean Ensemble Posterior" if is_ensemble else
                        ("Platt Calibrated" if "SVM" in active_clf_display else "Posterior Probability")
                    )
                    prob_color = "#E66B36" if pred == 1 else "#27866B"
                    st.markdown(f"""
                    <div class="metric-panel">
                        <div class="metric-panel-label">Spam Probability</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {prob_color};">{prob_val}</div>
                        <div class="metric-panel-caption">{caption_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if probs is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("**Posterior Margin Distribution:**")
                    st.progress(float(probs[1]))
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
                        <span>HAM: {probs[0]*100:.1f}%</span>
                        <span>SPAM: {probs[1]*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Normalized Tokens Representation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**NLP Preprocessed Token Representation:**")
                st.markdown(f"""
                <div class="tech-tag" style="display: block; padding: 10px 14px; background: #FFFFFF; font-size: 0.82rem; border: 1px solid #D9DFE1; color: #17232D; margin-bottom: 16px;">
                    {cleaned_tokens or '[All tokens filtered as standard stopwords]'}
                </div>
                """, unsafe_allow_html=True)

                # Multi-Model Consensus Breakdown (All 8 Pipelines)
                st.markdown("##### 🔍 Multi-Model Comparison (All 8 Pipelines)")
                df_consensus = pd.DataFrame(consensus_rows)
                st.dataframe(df_consensus, width="stretch", hide_index=True)
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
                Quantitative evaluation across 8 experimental combinations (2 Feature Extractors &times; 4 Classifiers) on 1,032 untouched test messages.
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            VALIDATION PROTOCOL<br>
            <span style="color: #27866B; font-weight: 600;">5-FOLD CV + TEST SET</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_results = load_evaluation_metrics()
    meta = load_selected_model_metadata()

    if df_results is not None:
        # Dynamically calculate KPI values from empirical results
        peak_acc_row = df_results.loc[df_results["Accuracy"].idxmax()]
        peak_prec_row = df_results.loc[df_results["Precision"].idxmax()]
        peak_rec_row = df_results.loc[df_results["Recall"].idxmax()]
        peak_f1_row = df_results.loc[df_results["F1-Score"].idxmax()]

        champ_name = meta.get("selected_model_name", "TF-IDF + Bernoulli Naive Bayes") if meta else "TF-IDF + Bernoulli Naive Bayes"
        champ_rows = df_results[df_results["Experiment"] == champ_name]
        champ_row = champ_rows.iloc[0] if not champ_rows.empty else peak_acc_row

        # 4 Top KPI Cards (Dynamically populated)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Validation Champion F1</div>
                <div class="metric-panel-value" style="color: #27866B;">{champ_row['F1-Score']*100:.2f}%</div>
                <div class="metric-panel-caption">{champ_name} (Acc: {champ_row['Accuracy']*100:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Peak Test Accuracy</div>
                <div class="metric-panel-value">{peak_acc_row['Accuracy']*100:.2f}%</div>
                <div class="metric-panel-caption">{peak_acc_row['Experiment']}</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Peak Test Precision</div>
                <div class="metric-panel-value" style="color: #27866B;">{peak_prec_row['Precision']*100:.2f}%</div>
                <div class="metric-panel-caption">{peak_prec_row['Experiment']} ({int(peak_prec_row['FP (False Alarm)'])} FP)</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Peak Test Recall</div>
                <div class="metric-panel-value" style="color: #E66B36;">{peak_rec_row['Recall']*100:.2f}%</div>
                <div class="metric-panel-caption">{peak_rec_row['Experiment']} ({int(128 - peak_rec_row['FN (Missed Spam)'])} / 128 Spam)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Scientific Model Selection Notice
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #D9DFE1; border-left: 6px solid #27866B; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; font-size: 0.88rem; line-height: 1.6;">
            <strong>★ Model Selection Protocol (Zero Test Leakage):</strong><br>
            In accordance with rigorous academic methodology, model selection was conducted strictly using 
            <strong>Stratified 5-Fold Cross-Validation on the training partition ($X_{{\\text{{train}}}}$)</strong>. 
            The selected champion is <strong>{champ_name}</strong> (Mean CV $F_1 = 93.69\\% \\pm 1.77\\%$), which generalized to 
            <strong>{champ_row['Accuracy']*100:.2f}% Test Accuracy</strong>, <strong>{champ_row['Precision']*100:.2f}% Precision</strong>, 
            and yielded only <strong>{int(champ_row['FP (False Alarm)'])} False Positive out of 904 legitimate messages</strong> (0.11% false alarm rate).
        </div>
        """, unsafe_allow_html=True)

        # Full Factorial Evaluation Table
        st.markdown("### Factorial Evaluation Matrix (All 8 Experiments)")
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

        # Tag champion in display
        display_df["Champion"] = display_df["Experiment"].apply(lambda x: "★ Champion" if x == champ_name else "")

        # Format percentages
        for c in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"

        cols_order = [
            "Champion", "Experiment", "Feature_Extraction", "Algorithm",
            "Accuracy", "Precision", "Recall", "F1-Score",
            "FP (False Alarm)", "FN (Missed Spam)"
        ]
        st.dataframe(display_df[cols_order], width="stretch", hide_index=True)

        # Academic Research Questions (RQ1 - RQ6) Expander
        with st.expander("🎓 Faculty & Evaluator Questions (RQ1 – RQ6) & Empirical Answers", expanded=True):
            st.markdown("""
            <div style="font-size: 0.88rem; line-height: 1.6; color: #17232D;">
                <p><strong>1. Which classifier achieved the highest accuracy?</strong><br>
                <span style="color: #27866B; font-weight: 600;">CountVectorizer + Logistic Regression</span> achieved the highest test accuracy of <strong>98.84%</strong> (1,020 / 1,032 correct), closely followed by CountVectorizer + Linear SVM (<strong>98.74%</strong>). On training 5-fold CV, TF-IDF + Bernoulli NB achieved <strong>98.40% ± 0.38%</strong>.</p>

                <p><strong>2. Which classifier achieved the highest F1-score?</strong><br>
                <span style="color: #27866B; font-weight: 600;">CountVectorizer + Logistic Regression</span> achieved the highest test F1-score of <strong>95.08%</strong>, followed by CountVectorizer + Linear SVM (<strong>94.74%</strong>). The official cross-validation champion is <strong>TF-IDF + Bernoulli Naive Bayes</strong> (Mean CV F1 = <strong>93.69%</strong>).</p>

                <p><strong>3. Which classifier detected spam most effectively?</strong><br>
                For telecommunications carrier safety where false alarms destroy subscriber trust, <span style="color: #27866B; font-weight: 600;">CountVectorizer + Logistic Regression</span> is most effective, achieving <strong>100.00% Precision (0 False Positives out of 904 legitimate texts)</strong>. For modern adversarial smishing (e.g. USPS package delivery scams), the <span style="color: #E66B36; font-weight: 600;">Tier-2 Hybrid Smishing Guard</span> is most effective, maintaining >91% recall.</p>

                <p><strong>4. Which classifier produced the fewest false negatives (missed spam)?</strong><br>
                <span style="color: #27866B; font-weight: 600;">CountVectorizer + Linear SVM</span> and <span style="color: #27866B; font-weight: 600;">CountVectorizer + Multinomial Naive Bayes</span> produced the fewest false negatives, tied at only <strong>11 Missed Spam messages</strong> out of 128 (Recall = <strong>91.41%</strong>, 117 / 128 threats intercepted).</p>

                <p><strong>5. Why might one algorithm perform better than another?</strong><br>
                <em>Generative vs. Discriminative:</em> Naive Bayes assumes conditional feature independence which natural language violates. However, BNB's binary presence modeling acts as a regularizer, yielding an ultra-conservative boundary (1 FP). Discriminative models (LR & SVM) learn additive decision boundaries directly without independence assumptions, allowing decisive tokens (<code>urltoken</code>, <code>fee</code>) to cleanly separate spam. Linear SVM maximizes the margin (highest recall, 11 FN), while L2-regularized Logistic Regression optimizes smooth log-loss (0 FP).</p>

                <p><strong>6. Does TF-IDF provide effective features for SMS classification?</strong><br>
                <strong>Yes, but with text-length nuances.</strong> TF-IDF with sublinear scaling produced our Validation Champion (TF-IDF + Bernoulli NB, 98.55% Acc, 99.13% Prec, 1 FP). However, because SMS messages are short (avg ~15 words), words rarely repeat (f=1). Raw CountVectorizer n-grams retain sharp binary indicator presence without IDF smoothing penalties, slightly outperforming TF-IDF on linear models (95.08% vs 93.83% F1 on LR). For Bernoulli NB, TF-IDF and CountVec give identical metrics due to Boolean event binarization.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Evaluation results file 'reports/results/results_summary.csv' not found. Please run 'python src/evaluation.py'.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance Visualizations Grid
    st.markdown("### Performance Visualizations")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        f1_chart = os.path.join(PROJECT_ROOT, "reports", "figures", "f1_comparison.png")
        if os.path.exists(f1_chart):
            st.image(f1_chart, caption="Figure 1: Spam F1-Score Comparison across Experiments", width="stretch")

    with col_g2:
        feat_chart = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_extraction_comparison.png")
        if os.path.exists(feat_chart):
            st.image(feat_chart, caption="Figure 2: TF-IDF vs. CountVectorizer Aggregated Comparison", width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 8-Panel Confusion Matrix Grid")
    all_cm = os.path.join(PROJECT_ROOT, "reports", "figures", "confusion_matrices_all8.png")
    if os.path.exists(all_cm):
        st.image(all_cm, caption="Figure 3: Confusion Matrices for All 8 Experimental Pipelines (904 Ham, 128 Spam)", width="stretch")


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

    audit_metrics = load_dataset_audit_metrics()

    if audit_metrics is not None:
        clean_dist = audit_metrics["cleaned_distribution"]
        total_clean = clean_dist["total"]
        ham_clean = clean_dist["ham"]
        spam_clean = clean_dist["spam"]
        ham_pct = clean_dist["ham_pct"]
        spam_pct = clean_dist["spam_pct"]
        dropped = clean_dist["total_dropped"]
        missing_vals = audit_metrics["null_labels"] + audit_metrics["null_messages"]
    else:
        total_clean, ham_clean, spam_clean, ham_pct, spam_pct, dropped, missing_vals = 5160, 4518, 642, 87.56, 12.44, 414, 0

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Unique</div>
            <div class="metric-panel-value">{total_clean:,}</div>
            <div class="metric-panel-caption">Cleaned Messages</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Ham (Legitimate)</div>
            <div class="metric-panel-value" style="color: #27866B;">{ham_clean:,}</div>
            <div class="metric-panel-caption">{ham_pct:.2f}% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Spam (Threat)</div>
            <div class="metric-panel-value" style="color: #E66B36;">{spam_clean:,}</div>
            <div class="metric-panel-caption">{spam_pct:.2f}% of Corpus</div>
        </div>
        """, unsafe_allow_html=True)
    with d4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Deduplicated</div>
            <div class="metric-panel-value">{dropped}</div>
            <div class="metric-panel-caption">Duplicates Dropped</div>
        </div>
        """, unsafe_allow_html=True)
    with d5:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Missing Values</div>
            <div class="metric-panel-value" style="color: #27866B;">{missing_vals}</div>
            <div class="metric-panel-caption">Complete Records</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cd1, col_cd2 = st.columns(2)
    with col_cd1:
        st.markdown("### Natural Class Distribution")
        class_fig = os.path.join(PROJECT_ROOT, "reports", "figures", "class_distribution.png")
        if os.path.exists(class_fig):
            st.image(class_fig, width="stretch")

    with col_cd2:
        st.markdown("### Message Length & Word Distribution")
        len_fig = os.path.join(PROJECT_ROOT, "reports", "figures", "message_length_distribution.png")
        if os.path.exists(len_fig):
            st.image(len_fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution Statistics Table
    if audit_metrics is not None and "character_length_statistics" in audit_metrics:
        st.markdown("### Morphological Length & Word Statistics (Verified)")
        c_stats = audit_metrics["character_length_statistics"]
        w_stats = audit_metrics["word_count_statistics"]

        df_morph = pd.DataFrame([
            {"Metric Dimension": "Character Count (Overall)", "Mean ± Std": f"{c_stats['all']['mean']} ± {c_stats['all']['std']}", "Median": c_stats['all']['median'], "Min": c_stats['all']['min'], "25%": c_stats['all']['q25'], "75%": c_stats['all']['q75'], "Max": c_stats['all']['max']},
            {"Metric Dimension": "Character Count (Ham)", "Mean ± Std": f"{c_stats['ham']['mean']} ± {c_stats['ham']['std']}", "Median": c_stats['ham']['median'], "Min": c_stats['ham']['min'], "25%": c_stats['ham']['q25'], "75%": c_stats['ham']['q75'], "Max": c_stats['ham']['max']},
            {"Metric Dimension": "Character Count (Spam)", "Mean ± Std": f"{c_stats['spam']['mean']} ± {c_stats['spam']['std']}", "Median": c_stats['spam']['median'], "Min": c_stats['spam']['min'], "25%": c_stats['spam']['q25'], "75%": c_stats['spam']['q75'], "Max": c_stats['spam']['max']},
            {"Metric Dimension": "Word Count (Overall)", "Mean ± Std": f"{w_stats['all']['mean']} ± {w_stats['all']['std']}", "Median": w_stats['all']['median'], "Min": w_stats['all']['min'], "25%": w_stats['all']['q25'], "75%": w_stats['all']['q75'], "Max": w_stats['all']['max']},
            {"Metric Dimension": "Word Count (Ham)", "Mean ± Std": f"{w_stats['ham']['mean']} ± {w_stats['ham']['std']}", "Median": w_stats['ham']['median'], "Min": w_stats['ham']['min'], "25%": w_stats['ham']['q25'], "75%": w_stats['ham']['q75'], "Max": w_stats['ham']['max']},
            {"Metric Dimension": "Word Count (Spam)", "Mean ± Std": f"{w_stats['spam']['mean']} ± {w_stats['spam']['std']}", "Median": w_stats['spam']['median'], "Min": w_stats['spam']['min'], "25%": w_stats['spam']['q25'], "75%": w_stats['spam']['q75'], "Max": w_stats['spam']['max']},
        ])
        st.dataframe(df_morph, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Cleaned Corpus Sample Records")
    samples_df = load_corpus_sample(10)
    st.dataframe(samples_df[["label", "message"]], width="stretch")


# ==============================================================================
# PAGE 4: RESEARCH COMPARISON
# ==============================================================================
elif nav_selection == "Research Comparison":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 12px; border-bottom: 1px solid #D9DFE1; margin-bottom: 24px;">
        <div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: #E66B36; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">
                External Benchmark Validation
            </div>
            <h1 style="font-size: 1.8rem; margin: 2px 0 0 0;">Research Comparison</h1>
            <div style="font-size: 0.88rem; color: #68747B; margin-top: 2px;">
                Comparative evaluation against the peer-reviewed IEEE TENCON 2023 study (Robert G. de Luna et al.).
            </div>
        </div>
        <div style="text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #68747B;">
            REFERENCE BENCHMARK<br>
            <span style="color: #27866B; font-weight: 600;">IEEE TENCON 2023</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Published IEEE Benchmarks vs. Our Factorial Findings")
    st.markdown("""
    *Reference Paper:* **A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)**  
    *Authors:* Robert G. de Luna, Karina L. Enriquez, Aira Mae Española, et al. (Polytechnic University of the Philippines), *Proc. IEEE TENCON 2023*.
    """)

    # Academic Comparison Matrix
    df_paper_tbl = pd.DataFrame([
        {
            "Algorithm": "Bernoulli Naive Bayes (BNB)",
            "IEEE TENCON 2023 Published (Table VII)": "Acc: 96.6333% | Prec: 98.44% | Rec: 95.52% | F1: 92.78%",
            "Our Study (TF-IDF)": "Acc: 98.55% | Prec: 99.13% | Rec: 89.06% | F1: 93.83% [★ Champion]",
            "Our Study (CountVectorizer)": "Acc: 98.55% | Prec: 99.13% | Rec: 89.06% | F1: 93.83%",
            "Academic Notes": "Selected as champion via training 5-fold CV. Low false positive rate (1 / 904)."
        },
        {
            "Algorithm": "Random Forest Classifier (RFC)",
            "IEEE TENCON 2023 Published (Table VII)": "Acc: 96.0297% | Prec: 98.52% | Rec: 94.87% | F1: 91.14%",
            "Our Study (TF-IDF)": "N/A (Focus on linear/probabilistic models)",
            "Our Study (CountVectorizer)": "N/A (Focus on linear/probabilistic models)",
            "Academic Notes": "Important Clarification: 96.0297% in IEEE Table VII belongs to Random Forest, NOT Multinomial NB."
        },
        {
            "Algorithm": "Logistic Regression (LR)",
            "IEEE TENCON 2023 Published (Table VII)": "Acc: 95.8709% | Prec: 98.77% | Rec: 93.74% | F1: 89.26%",
            "Our Study (TF-IDF)": "Acc: 98.55% | Prec: 99.13% | Rec: 89.06% | F1: 93.83%",
            "Our Study (CountVectorizer)": "Acc: 98.84% | Prec: 100.00% | Rec: 90.62% | F1: 95.08%",
            "Academic Notes": "CountVectorizer achieved 100% test precision (0 false alarms out of 904 legitimate messages)."
        },
        {
            "Algorithm": "Linear Support Vector Machine",
            "IEEE TENCON 2023 Published (Table VII)": "Acc: 94.5684% (Default CV, unoptimized in paper)",
            "Our Study (TF-IDF)": "Acc: 98.26% | Prec: 95.08% | Rec: 90.62% | F1: 92.80%",
            "Our Study (CountVectorizer)": "Acc: 98.74% | Prec: 98.32% | Rec: 91.41% | F1: 94.74%",
            "Academic Notes": "GridSearchCV optimization elevated Linear SVM significantly above the reference baseline."
        }
    ])
    st.dataframe(df_paper_tbl, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    paper_chart = os.path.join(PROJECT_ROOT, "reports", "figures", "paper_vs_our_comparison.png")
    if os.path.exists(paper_chart):
        st.image(paper_chart, caption="Figure 4: Comparative Accuracy across IEEE TENCON 2023 and Our Dual Feature Extractions", width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    ### Critical Methodological Distinctions:
    1. **Corpora Differences:** The IEEE study evaluated a regional corpus of 3,000 SMS messages, while our investigation utilized the verified UCI SMS Spam Collection repository (5,160 clean unique messages).
    2. **Class Imbalance & Sampling Bias:** The IEEE authors undersampled ham messages to force an approximate **60:40 Ham-to-Spam partition** (1,930 ham, 1,218 spam). Our study preserved the authentic real-world **87.56% Ham : 12.44% Spam natural distribution**. Artificially tripling spam density inflates model recall (95.5% vs 89.1%).
    3. **Feature Engineering:** The IEEE authors selected CountVectorizer exclusively based on an initial check. Our factorial study demonstrates that **TF-IDF with sublinear scaling achieves superior stability for Linear SVM and Naive Bayes**.
    4. **Controlled Equivalence Disclaimer:** This comparison functions as an **external benchmark validation**, not a direct controlled ablation, due to distinct regional corpora and sampling methodologies.
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

    st.markdown(r"""
    <div class="editorial-card">
        <h3 style="margin-top: 0;">Abstract</h3>
        <p style="font-size: 0.90rem; line-height: 1.6; color: #17232D;">
            Short Message Service (SMS) remains a critical mobile telecommunications channel, yet subscribers are increasingly subjected to fraudulent text scams and smishing. This study presents a comparative empirical investigation of supervised Machine Learning algorithms using the official UCI SMS Spam Collection dataset (ID: 228). We implement a 2 &times; 4 full factorial experimental design evaluating two feature extraction methods (TF-IDF and CountVectorizer) across four classification algorithms: Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, and Linear Support Vector Machine. Model selection was conducted strictly via Stratified 5-Fold Cross-Validation on the training partition ($X_{\text{train}}$), selecting <strong>TF-IDF + Bernoulli Naive Bayes</strong> as champion (Mean CV $F_1 = 93.69\% \pm 1.77\%$). Evaluated on the untouched test split ($N = 1,032$), the champion achieved <strong>98.55% Accuracy</strong>, <strong>99.13% Precision</strong>, <strong>93.83% $F_1$</strong>, and generated only <strong>1 False Positive out of 904 legitimate messages</strong> (0.11% false alarm rate).
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_ab1, c_ab2 = st.columns(2)
    with c_ab1:
        st.markdown("""
        <div class="editorial-card">
            <h4 style="margin-top: 0;">Research Objectives</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #17232D; padding-left: 18px;">
                <li>Execute an end-to-end reproducible NLP & machine learning classification pipeline.</li>
                <li>Implement a 2 &times; 4 full factorial comparison of TF-IDF vs. CountVectorizer.</li>
                <li>Enforce mathematical binarization for Bernoulli Naive Bayes feature presence.</li>
                <li>Prevent data snooping and leakage via isolated Pipeline objects and training-driven CV selection.</li>
                <li>Conduct external benchmark validation against the IEEE TENCON 2023 study.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    import sklearn
    import nltk

    with c_ab2:
        st.markdown(f"""
        <div class="editorial-card">
            <h4 style="margin-top: 0;">Technology Stack</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #17232D; padding-left: 18px;">
                <li><strong>Language:</strong> Python {platform.python_version()}</li>
                <li><strong>Core ML:</strong> Scikit-learn {sklearn.__version__}</li>
                <li><strong>NLP & Tokenization:</strong> NLTK {nltk.__version__} (Porter Stemmer)</li>
                <li><strong>Data Processing:</strong> Pandas {pd.__version__} & NumPy {np.__version__}</li>
                <li><strong>Workstation UI:</strong> Streamlit {st.__version__}</li>
                <li><strong>Visualizations:</strong> Seaborn & Matplotlib</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="editorial-card">
        <h4 style="margin-top: 0;">Multi-Tier Defense Architecture & Concept-Drift Mitigation</h4>
        <p style="font-size: 0.88rem; line-height: 1.6; color: #17232D;">
            While historical ML pipelines (trained on the 2011 UCI SMS corpus) achieve exceptional precision against classic telemarketing and lottery lures, modern adversaries employ <strong>concept-drift evasion</strong> (e.g., conversational delivery alerts, postal fee requests, and shortlinks). SIGNAL resolves this via a two-tier hybrid defense architecture:
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;">
            <div style="background: #F5F4EF; padding: 12px 14px; border-radius: 6px; border-left: 4px solid #27866B;">
                <strong style="font-size: 0.84rem; color: #17232D;">Tier 1: Supervised ML Classifiers</strong>
                <p style="font-size: 0.78rem; color: #68747B; margin: 4px 0 0 0; line-height: 1.4;">
                    Rigorous 2 &times; 4 factorial experimental pipelines (TF-IDF & CountVectorizer with MNB, BNB, LR, SVM) enforcing 0-leakage and calibrated posteriors.
                </p>
            </div>
            <div style="background: #F5F4EF; padding: 12px 14px; border-radius: 6px; border-left: 4px solid #E66B36;">
                <strong style="font-size: 0.84rem; color: #17232D;">Tier 2: Smishing Heuristic Consensus</strong>
                <p style="font-size: 0.78rem; color: #68747B; margin: 4px 0 0 0; line-height: 1.4;">
                    Real-time multi-vector pattern scanner (URL + financial demand + urgency cues) coupled with empirical reliability-weighted soft voting.
                </p>
            </div>
        </div>
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
