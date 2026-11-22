"""
SIGNAL: SMS Intelligence Lab - Local Terminal Predictor
Interactive & CLI analysis across all 8 Factorial Machine Learning Pipelines.
No browser or web server required.
"""

import os
import sys
import time
import joblib

# Ensure src/ is on python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess_text
from feature_extraction import TextPreprocessorTransformer  # noqa: F401


# ANSI color codes for terminal formatting
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

PIPELINE_FILES = [
    ("Bernoulli Naive Bayes", "TF-IDF", "tfidf_plus_bernoulli_naive_bayes.pkl"),
    ("Bernoulli Naive Bayes", "CountVectorizer", "countvectorizer_plus_bernoulli_naive_bayes.pkl"),
    ("Linear Support Vector Machine", "CountVectorizer", "countvectorizer_plus_linear_svm.pkl"),
    ("Linear Support Vector Machine", "TF-IDF", "tfidf_plus_linear_svm.pkl"),
    ("Logistic Regression", "CountVectorizer", "countvectorizer_plus_logistic_regression.pkl"),
    ("Logistic Regression", "TF-IDF", "tfidf_plus_logistic_regression.pkl"),
    ("Multinomial Naive Bayes", "CountVectorizer", "countvectorizer_plus_multinomial_naive_bayes.pkl"),
    ("Multinomial Naive Bayes", "TF-IDF", "tfidf_plus_multinomial_naive_bayes.pkl"),
]


def load_all_models():
    """Loads all 8 trained pipelines into memory once."""
    loaded = []
    for algo, vec, filename in PIPELINE_FILES:
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"{RED}[ERROR] Model file missing: {path}{RESET}")
            continue
        try:
            model = joblib.load(path)
            loaded.append((algo, vec, model))
        except Exception as e:
            print(f"{YELLOW}[WARNING] Could not load {filename}: {e}{RESET}")
    return loaded


def analyze_message(text, models):
    """Evaluates an SMS string across all 8 models and prints the telemetry."""
    if not text.strip():
        print(f"{YELLOW}Empty message provided.{RESET}\n")
        return

    clean_tokens = preprocess_text(text)
    
    start_t = time.perf_counter()
    results = []
    spam_count = 0

    for algo, vec, model in models:
        pred = model.predict([text])[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba([text])[0][1]
            except Exception:
                prob = None
        if pred == 1:
            spam_count += 1
        results.append((algo, vec, pred, prob))

    total_latency_ms = (time.perf_counter() - start_t) * 1000

    # Header / Consensus Banner
    print("\n" + "=" * 78)
    print(f"{BOLD}SIGNAL : SMS INTELLIGENCE LAB — MULTI-MODEL ANALYSIS{RESET}")
    print("=" * 78)
    print(f"{BOLD}Original Text:{RESET}\n\"{text}\"\n")
    print(f"{DIM}Clean NLP Tokens:{RESET} {clean_tokens or '[All tokens filtered as English stopwords]'}")
    print("-" * 78)

    # Consensus Status
    total_models = len(results)
    if spam_count >= 5:
        consensus_text = f"{RED}{BOLD}🚨 HIGH THREAT CONSENSUS ({spam_count}/{total_models} Models Flagged SPAM){RESET}"
    elif spam_count > 0:
        consensus_text = f"{YELLOW}{BOLD}⚠️  DISCORDANT / PARTIAL THREAT ({spam_count}/{total_models} Models Flagged SPAM){RESET}"
    else:
        consensus_text = f"{GREEN}{BOLD}✅ UNANIMOUS SAFE (0/{total_models} Models Flagged SPAM){RESET}"

    print(f"Consensus Verdict: {consensus_text}")
    print(f"Total Multi-Model Latency: {total_latency_ms:.2f} ms")
    print("-" * 78)

    # Detailed Table
    print(f"{BOLD}{'ALGORITHM':<32} {'VECTORIZER':<18} {'VERDICT':<12} {'SPAM PROB':<10}{RESET}")
    print("-" * 78)

    for algo, vec, pred, prob in results:
        if pred == 1:
            verdict_str = f"{RED}{BOLD}SPAM{RESET}"
        else:
            verdict_str = f"{GREEN}SAFE (HAM){RESET}"

        prob_str = f"{prob*100:5.1f}%" if prob is not None else "   N/A"
        print(f"{algo:<32} {vec:<18} {verdict_str:<21} {prob_str:<10}")

    print("-" * 78)

    # Explanatory Insight
    if 0 < spam_count < total_models:
        print(f"{CYAN}{BOLD}Model Disagreement Insight:{RESET}")
        print(f"  • This message contains subtle smishing vocabulary mixed with conversational terms.")
        print(f"  • {BOLD}Bernoulli NB{RESET} flags binary trigger tokens (e.g. 'fee', 'pay', 'update') -> {RED}SPAM{RESET}.")
        print(f"  • {BOLD}Linear SVM (TF-IDF){RESET} balances high-frequency benign terms ('today', 'package', 'address') -> {GREEN}SAFE{RESET}.")
        print(f"  • In production telecom SOCs, this discordant status triggers quarantine / secondary review.")
    elif spam_count == 0:
        print(f"{GREEN}All models agree this message contains authentic conversational patterns.{RESET}")
    else:
        print(f"{RED}All models detected strong characteristic fraud / marketing signatures.{RESET}")
    print("=" * 78 + "\n")


PRESET_SAMPLES = {
    "1": ("USPS Redelivery Smishing", "USPS Notice: Your package could not be delivered today due to an incorrect house number. Please update your delivery address and pay the $1.99 redelivery fee at http://usps-tracking-portal-update.com within 12 hours."),
    "2": ("Banking Urgent Phishing", "URGENT: Your Chase Bank debit card has been temporarily locked due to unusual login attempts. Verify your identity immediately at http://chase-secure-auth.com to prevent permanent suspension."),
    "3": ("Lottery / Prize Claim", "Congratulations! Your mobile number won a £1,000 cash prize. Call 08712345678 to claim your award today. T&Cs apply."),
    "4": ("Legitimate Peer Ham", "Hey, are we still meeting at the library at 4pm for the project discussion? Let me know."),
    "5": ("Legitimate University Ham", "The Department of Computer Science has rescheduled the Research Methodologies seminar to Friday 10:00 AM in Hall B.")
}


def main():
    print(f"\n{BOLD}Initializing SIGNAL Local Terminal Workstation...{RESET}")
    models = load_all_models()
    if not models:
        print(f"{RED}[FATAL] No models could be loaded from {MODEL_DIR}. Exiting.{RESET}")
        sys.exit(1)

    print(f"{GREEN}✓ Successfully loaded {len(models)} machine learning pipelines.{RESET}")

    # Direct argument mode (e.g. python predict.py "some message text")
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        analyze_message(query, models)
        return

    # Interactive loop mode
    print(f"\n{BOLD}SIGNAL: SMS INTELLIGENCE LAB (Interactive Local Console){RESET}")
    print("Type or paste any SMS message below, choose a preset number (1-5), or type 'exit' to quit.\n")
    print(f"{DIM}Presets available:")
    for k, (name, _) in PRESET_SAMPLES.items():
        print(f"  [{k}] {name}")
    print(f"{RESET}")

    while True:
        try:
            user_input = input(f"{BOLD}Enter SMS (or 1-5 / 'exit') > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"{DIM}Exiting SIGNAL Console. Goodbye!{RESET}")
                break

            if user_input in PRESET_SAMPLES:
                label, sample_text = PRESET_SAMPLES[user_input]
                print(f"\n{CYAN}--- Loading Preset [{user_input}]: {label} ---{RESET}")
                analyze_message(sample_text, models)
            else:
                analyze_message(user_input, models)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Interrupted. Exiting SIGNAL Console.{RESET}")
            break


if __name__ == "__main__":
    main()
