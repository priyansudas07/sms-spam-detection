"""
Smishing Heuristics & Weighted Consensus Engine.
Part of SIGNAL — SMS Intelligence Lab.

Provides:
1. Real-time heuristic pattern detection for modern SMS Phishing (Smishing).
2. Multi-tier threat scoring for adversarial mimicry (e.g., postal scams, banking lures).
3. Weighted soft-probability consensus ensemble aggregating all 8 ML pipelines.
"""

import re
from typing import Dict, Any, List, Tuple


# Regex patterns targeting modern smishing anatomy
URL_REGEX = re.compile(r'https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|xyz|top|info|live|ru|cn|cc|to|tk|me|site|link)/\S*', re.IGNORECASE)
PAYMENT_REGEX = re.compile(r'[$£€₹]|\b(?:pay|payment|fee|redelivery fee|charge|cost|penalty|fine|refund|balance|transfer)\b', re.IGNORECASE)
URGENCY_REGEX = re.compile(r'\b(?:urgent|immediately|within \d+ hours?|within \d+ mins?|today|final notice|action required|suspended|locked|expires|expiring|restricted|unauthorized)\b', re.IGNORECASE)
IMPERSONATION_REGEX = re.compile(r'\b(?:usps|fedex|ups|dhl|royal mail|amazon|netflix|apple|paypal|chase|wellsfargo|citi|bank|irs|tax|customs|evri)\b', re.IGNORECASE)
DELIVERY_LURE_REGEX = re.compile(r'\b(?:package|parcel|shipment|delivery address|incorrect house number|redelivery|courier|dispatch|customs fee)\b', re.IGNORECASE)


def analyze_smishing_heuristics(raw_text: str) -> Dict[str, Any]:
    """
    Evaluates an SMS string for multi-vector smishing patterns:
    - Presence of external hyperlink / unverified shortlink
    - Financial demands (fees, payment, currency symbols)
    - Manufactured urgency and time-decay deadlines
    - Impersonated trusted authorities (postal, courier, banking)
    
    Returns structured threat telemetry.
    """
    matched_indicators: List[str] = []
    threat_score = 0

    has_url = bool(URL_REGEX.search(raw_text))
    has_payment = bool(PAYMENT_REGEX.search(raw_text))
    has_urgency = bool(URGENCY_REGEX.search(raw_text))
    has_impersonation = bool(IMPERSONATION_REGEX.search(raw_text))
    has_delivery = bool(DELIVERY_LURE_REGEX.search(raw_text))

    if has_url:
        matched_indicators.append("External Hyperlink / Unverified Web Domain")
        threat_score += 25

    if has_payment:
        matched_indicators.append("Financial Demand / Monetary Fee Request")
        threat_score += 20

    if has_urgency:
        matched_indicators.append("Manufactured Urgency / Time-Pressure Deadline")
        threat_score += 15

    if has_impersonation:
        matched_indicators.append("Brand / Authority Impersonation Lure")
        threat_score += 15

    if has_delivery:
        matched_indicators.append("Package / Delivery Exception Narrative")
        threat_score += 15

    # Multi-vector lethal compound combinations:
    # 1. URL + Monetary Demand (e.g. $1.99 redelivery fee at http://...)
    if has_url and has_payment:
        matched_indicators.append("High-Risk Compound: Hyperlink coupled with Financial Demand")
        threat_score += 20

    # 2. URL + Urgency + Delivery/Bank lure
    if has_url and (has_urgency or has_delivery or has_impersonation):
        matched_indicators.append("High-Risk Compound: Actionable Link with Coercive Context")
        threat_score += 15

    threat_score = min(threat_score, 100)

    if threat_score >= 60:
        risk_level = "CRITICAL THREAT"
        badge_color = "#E66B36"
    elif threat_score >= 35:
        risk_level = "ELEVATED RISK"
        badge_color = "#E6A23C"
    elif threat_score >= 20:
        risk_level = "MODERATE NOTICE"
        badge_color = "#3498DB"
    else:
        risk_level = "NEGLIGIBLE / BENIGN"
        badge_color = "#27866B"

    return {
        "has_url": has_url,
        "has_payment": has_payment,
        "has_urgency": has_urgency,
        "has_impersonation": has_impersonation,
        "has_delivery": has_delivery,
        "threat_score": threat_score,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "matched_indicators": matched_indicators,
        "is_smishing_lure": (threat_score >= 50 and has_url)
    }


# Empirical Model Reliability Weights
# Discriminative models (Linear SVM and Logistic Regression) receive higher weighting
# due to superior resilience against semantic obfuscation and concept drift.
MODEL_WEIGHTS: Dict[Tuple[str, str], float] = {
    ("CountVectorizer", "Linear Support Vector Machine"): 2.2,
    ("TF-IDF", "Linear Support Vector Machine"): 1.8,
    ("CountVectorizer", "Logistic Regression"): 2.0,
    ("TF-IDF", "Logistic Regression"): 1.5,
    ("TF-IDF", "Bernoulli Naive Bayes"): 1.0,
    ("CountVectorizer", "Bernoulli Naive Bayes"): 1.0,
    ("CountVectorizer", "Multinomial Naive Bayes"): 0.8,
    ("TF-IDF", "Multinomial Naive Bayes"): 0.8,
}


def calculate_weighted_ensemble_prediction(
    pipeline_outputs: Dict[Tuple[str, str], Dict[str, Any]],
    smishing_telemetry: Dict[str, Any] = None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes a weighted soft-probability consensus prediction across all 8 models.
    
    If smishing telemetry indicates a critical compound threat (URL + Payment/Urgency),
    applies adaptive Bayesian threat adjustment to safeguard against generative model blind spots.
    """
    total_weight = 0.0
    weighted_spam_prob_sum = 0.0
    hard_spam_votes = 0
    total_models = len(pipeline_outputs)

    model_breakdown = []

    for key, output in pipeline_outputs.items():
        feat_tech, algo_name = key
        pred = output.get("pred", 0)
        prob = output.get("prob")  # list [P(Ham), P(Spam)] or None
        
        weight = MODEL_WEIGHTS.get(key, 1.0)
        total_weight += weight

        if pred == 1:
            hard_spam_votes += 1

        # Use calibrated probability if present, otherwise binary vote
        if prob is not None and len(prob) == 2:
            p_spam = float(prob[1])
        else:
            p_spam = 1.0 if pred == 1 else 0.0

        weighted_spam_prob_sum += (p_spam * weight)

        model_breakdown.append({
            "feature_tech": feat_tech,
            "algorithm": algo_name,
            "pred": pred,
            "p_spam": p_spam,
            "weight": weight
        })

    raw_weighted_prob = weighted_spam_prob_sum / total_weight if total_weight > 0 else 0.0

    # Hybrid Defense Integration:
    # If the rule engine identified a verified lethal smishing pattern (URL + Payment Fee/Urgency),
    # combine with statistical probabilities to prevent unweighted minority dilution.
    final_prob = raw_weighted_prob
    if smishing_telemetry and smishing_telemetry.get("is_smishing_lure"):
        heuristic_boost = smishing_telemetry["threat_score"] / 100.0 * 0.25
        final_prob = min(raw_weighted_prob + heuristic_boost, 1.0)

    final_pred = 1 if final_prob >= threshold else 0

    return {
        "pred": final_pred,
        "probability": [1.0 - final_prob, final_prob],
        "raw_weighted_prob": round(raw_weighted_prob, 4),
        "final_prob": round(final_prob, 4),
        "hard_spam_votes": hard_spam_votes,
        "total_models": total_models,
        "threshold_used": threshold,
        "hybrid_boost_applied": bool(smishing_telemetry and smishing_telemetry.get("is_smishing_lure")),
        "model_breakdown": model_breakdown
    }
