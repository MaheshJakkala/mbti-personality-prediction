"""
predict.py
----------
Standalone inference module for MBTI personality prediction.
Loads trained models and vectorizer from disk and predicts MBTI type
from a raw text input string.

Usage:
    from src.predict import predict_mbti
    result = predict_mbti("I love reading alone and thinking deeply...")
    # Returns: "INFJ"
"""

import pickle
import os
from src.preprocess import clean_text, tokens_to_string

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _load(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def _map_prediction(pred, trait):
    mapping = {
        "IE": {0: "I", 1: "E"},
        "NS": {0: "N", 1: "S"},
        "TF": {0: "T", 1: "F"},
        "JP": {0: "J", 1: "P"},
    }
    return mapping[trait][int(pred)]


def predict_mbti(raw_text, verbose=False):
    """
    Predict the MBTI personality type of a given raw text.

    Args:
        raw_text (str): Any piece of text (social media posts, essay, speech, etc.)
        verbose (bool): If True, prints per-dimension probabilities.

    Returns:
        str: 4-letter MBTI type string e.g. "INFJ"
    """
    # Load artefacts
    vectorizer = _load("tfidf_vectorizer.pkl")
    ie_model = _load("ie_model.pkl")
    ns_model = _load("ns_model.pkl")
    tf_model = _load("tf_model.pkl")
    jp_model = _load("jp_model.pkl")

    # Preprocess
    tokens = clean_text(raw_text)
    cleaned = tokens_to_string(tokens)
    X = vectorizer.transform([cleaned])

    result = ""
    for trait, model in [("IE", ie_model), ("NS", ns_model), ("TF", tf_model), ("JP", jp_model)]:
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        letter = _map_prediction(pred, trait)
        result += letter

        if verbose:
            labels = {
                "IE": ["Introvert (I)", "Extrovert (E)"],
                "NS": ["Intuition (N)", "Sensing (S)"],
                "TF": ["Thinking (T)", "Feeling (F)"],
                "JP": ["Judging (J)", "Perceiving (P)"],
            }
            print(f"  {labels[trait][0]}: {proba[0]:.3f} | {labels[trait][1]}: {proba[1]:.3f}  → {letter}")

    if verbose:
        print(f"\n  🔍 Predicted MBTI Type: {result}")

    return result


def save_models(vectorizer, ie, ns, tf, jp):
    """
    Serialize trained models and vectorizer to the models/ directory.

    Args:
        vectorizer: Fitted TfidfVectorizer
        ie, ns, tf, jp: Trained classifiers for each dimension
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "tfidf_vectorizer.pkl": vectorizer,
        "ie_model.pkl": ie,
        "ns_model.pkl": ns,
        "tf_model.pkl": tf,
        "jp_model.pkl": jp,
    }
    for fname, obj in artifacts.items():
        with open(os.path.join(MODEL_DIR, fname), "wb") as f:
            pickle.dump(obj, f)
    print(f"✅ Saved {len(artifacts)} artifacts to {MODEL_DIR}/")
