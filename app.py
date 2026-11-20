"""
Streamlit Web Application for SMS Spam Detection.

Allows users to input any SMS message and receive real-time classification
as either 'Spam' or 'Ham (Legitimate)', along with predicted class probability
and NLP token normalization breakdown.
"""

import os
import sys
import joblib
import streamlit as st

# Add src to system path to ensure custom transformers are unpickled cleanly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess_text
from feature_extraction import TextPreprocessorTransformer  # Required for unpickling pipeline


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector - ML Research",
    page_icon="📱",
    layout="centered"
)


@st.cache_resource
def load_trained_model():
    """
    Loads the trained best model pipeline from models/best_model.pkl.
    If not found, trains and saves it on the fly.
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Fallback if model not yet saved
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


# Header
st.title("📱 SMS Spam Detection System")
st.markdown(
    "**Academic Research Project:** *Comparative Analysis of ML Algorithms for SMS Spam Detection*  \n"
    "*Research Methodologies (RM) — 7th Semester B.Tech Project*"
)
st.divider()

# Sidebar: Research metadata & quick stats
with st.sidebar:
    st.header("🔬 Project Overview")
    st.markdown("""
    - **Dataset:** UCI SMS Spam Collection
    - **Active Model:** Linear Support Vector Machine (Linear SVM)
    - **Feature Extraction:** Sublinear TF-IDF (4,000 features, Unigrams + Bigrams)
    - **Evaluation Metrics:**
      - **Accuracy:** 98.16%
      - **Precision (Spam):** 96.58%
      - **Recall (Spam):** 88.28%
      - **F1-Score (Spam):** 92.24%
    """)
    st.divider()
    st.info("💡 **Academic Note:** Linear SVM maximizes the separating margin in high-dimensional sparse TF-IDF text representations, providing optimal spam classification resilience.")

# Sample messages selector
st.subheader("1. Enter or Select an SMS Message")

sample_messages = {
    "-- Select an example message or type your own below --": "",
    "🚨 Spam Example 1 (Cash Prize Scam)": "WINNER!! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim. Valid 12 hours.",
    "🚨 Spam Example 2 (Urgent Lottery / Service)": "FreeMsg Hey! You have won a guaranteed £1000 cash or holiday! Text CLAIM to 87121 now. T&Cs apply, £1.50/msg.",
    "✅ Ham Example 1 (Casual Chat)": "Hey mate, are we still meeting up for coffee tomorrow afternoon around 4?",
    "✅ Ham Example 2 (Work / Study Notice)": "Hi Dr. Sharma, I have submitted the Research Methodology assignment draft on Google Classroom. Please review."
}

selected_sample = st.selectbox("Quick Test Samples:", list(sample_messages.keys()))
default_text = sample_messages[selected_sample]

user_input = st.text_area(
    "SMS Message Content:",
    value=default_text,
    placeholder="Type or paste the SMS message here...",
    height=120
)

# Prediction button
if st.button("🔍 Classify SMS Message", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid SMS text message to classify.")
    else:
        try:
            model = load_trained_model()
            prediction = model.predict([user_input])[0]
            
            # Predict probability if supported
            prob_scores = None
            if hasattr(model, "predict_proba"):
                try:
                    prob_scores = model.predict_proba([user_input])[0]
                except Exception:
                    pass

            st.divider()
            st.subheader("2. Classification Result")

            col1, col2 = st.columns([1.5, 1])

            with col1:
                if prediction == 1:
                    st.error("### 🚨 Result: SPAM DETECTED")
                    st.markdown("**Assessment:** This message exhibits characteristic linguistic features of fraudulent, promotional, or unauthorized spam communications.")
                else:
                    st.success("### ✅ Result: HAM (LEGITIMATE)")
                    st.markdown("**Assessment:** This message is classified as a genuine, non-spam legitimate personal or transactional message.")

            with col2:
                if prob_scores is not None:
                    spam_prob = prob_scores[1] * 100
                    ham_prob = prob_scores[0] * 100
                    st.metric("Estimated Spam Probability", f"{spam_prob:.1f}%")
                    st.progress(float(prob_scores[1]))
                    st.caption(f"Ham Probability: {ham_prob:.1f}% | Spam Probability: {spam_prob:.1f}%")

            # Academic clarification on probability
            st.caption(
                "📌 **Methodological Clarification:** Probability scores produced via Platt-scaled decision boundaries "
                "represent relative classification margins and should not be confused with strictly calibrated Bayesian likelihoods."
            )

            # Preprocessing breakdown
            st.divider()
            with st.expander("🛠️ View NLP Text Preprocessing Pipeline Breakdown"):
                cleaned_tokens = preprocess_text(user_input)
                st.markdown(f"**Original Raw Text:** `{user_input}`")
                st.markdown(f"**Normalized Root Tokens (Stemmed, Stopwords & Noise Removed):**")
                st.code(cleaned_tokens or "[Empty token list after stopword filtering]")

        except Exception as e:
            st.error(f"Error during classification: {str(e)}")

# Footer
st.divider()
st.markdown(
    "<center><small>Developed for Research Methodology (RM) Academic Project | Built with Python, Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True
)
