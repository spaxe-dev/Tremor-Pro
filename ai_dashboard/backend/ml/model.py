"""
Inference module — loads the trained Random Forest and classifies windows.

This module is imported by the FastAPI backend (main.py) and provides
a simple API:
    from ml.model import get_classifier, classify_window, classify_batch

The model is loaded lazily on first use and cached in memory.
"""

import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from ml.features import (
    extract_features_single,
    extract_features_batch,
    CLASSES,
    IDX_TO_CLASS,
    FEATURE_NAMES,
)

# ──────────────────────────────────────────────
# Model loading (lazy singleton)
# ──────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "tremor_rf.joblib"

_cached_model: Optional[RandomForestClassifier] = None


def get_classifier() -> Optional[RandomForestClassifier]:
    """
    Load and cache the trained Random Forest model.
    Returns None if no model file exists.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not MODEL_PATH.exists():
        print(f"[ML] ⚠ No model found at {MODEL_PATH}")
        print(f"[ML]   Train one with: python -m ml.train_model")
        return None

    _cached_model = joblib.load(MODEL_PATH)
    print(f"[ML] ✓ Loaded Random Forest model ({MODEL_PATH.name})")
    print(f"[ML]   Trees: {_cached_model.n_estimators}, "
          f"Features: {_cached_model.n_features_in_}")
    return _cached_model


def is_model_available() -> bool:
    """Check if a trained model exists on disk."""
    return MODEL_PATH.exists()


# ──────────────────────────────────────────────
# Single-window classification
# ──────────────────────────────────────────────
def classify_window(
    b1: float,
    b2: float,
    b3: float,
    mean_norm: float = 0.0,
) -> dict:
    """
    Classify a single ESP32 window.

    Returns a dict with:
      - prediction: str ("no_tremor", "parkinsonian", "essential", "physiological")
      - tremor_type: str (same as prediction, or "none" for no_tremor)
      - is_tremor: bool
      - confidence: float (max probability)
      - probabilities: dict[str, float]
      - model_version: str
    """
    clf = get_classifier()
    if clf is None:
        return _fallback_classify(b1, b2, b3, mean_norm)

    features = extract_features_single(b1, b2, b3, mean_norm).reshape(1, -1)
    proba = clf.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = IDX_TO_CLASS[pred_idx]

    return {
        "prediction": pred_label,
        "tremor_type": pred_label if pred_label != "no_tremor" else "none",
        "is_tremor": pred_label != "no_tremor",
        "confidence": float(proba[pred_idx]),
        "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(proba)},
        "model_version": f"rf_v1_{date.today().isoformat()}",
    }


# ──────────────────────────────────────────────
# Batch classification (for session summary)
# ──────────────────────────────────────────────
def classify_batch(windows: list[dict]) -> dict:
    """
    Classify a batch of windows and return aggregate results.

    Input: list of dicts with keys b1, b2, b3, meanNorm
    Returns aggregate statistics over all windows.
    """
    clf = get_classifier()
    if clf is None or len(windows) == 0:
        return {"error": "No model loaded or empty input"}

    X = extract_features_batch(windows)
    proba = clf.predict_proba(X)            # (N, 4)
    preds = np.argmax(proba, axis=1)        # (N,)

    # Per-window results
    per_window = []
    for i in range(len(windows)):
        per_window.append({
            "prediction": IDX_TO_CLASS[int(preds[i])],
            "confidence": float(proba[i, preds[i]]),
        })

    # Aggregate
    unique, counts = np.unique(preds, return_counts=True)
    class_counts = {IDX_TO_CLASS[int(u)]: int(c) for u, c in zip(unique, counts)}
    dominant_class = IDX_TO_CLASS[int(unique[np.argmax(counts)])]
    mean_proba = {CLASSES[i]: float(proba[:, i].mean()) for i in range(len(CLASSES))}
    tremor_fraction = float((preds != 0).sum() / len(preds))

    return {
        "n_windows": len(windows),
        "dominant_prediction": dominant_class,
        "is_tremor": dominant_class != "no_tremor",
        "tremor_fraction": tremor_fraction,
        "class_counts": class_counts,
        "mean_probabilities": mean_proba,
        "per_window": per_window,
        "model_version": f"rf_v1_{date.today().isoformat()}",
    }


# ──────────────────────────────────────────────
# Fallback (when no model is trained yet)
# ──────────────────────────────────────────────
def _fallback_classify(b1: float, b2: float, b3: float,
                        mean_norm: float) -> dict:
    """
    Rule-based fallback when no trained model is available.
    Mirrors the ESP32 classify() logic.
    """
    from ml.features import label_window_rule_based
    label = label_window_rule_based(b1, b2, b3, mean_norm)
    total = b1 + b2 + b3
    # Fake confidence from band dominance
    bands = [b1, b2, b3]
    conf = max(bands) / (total + 1e-6) if total > 0.01 else 1.0

    return {
        "prediction": label,
        "tremor_type": label if label != "no_tremor" else "none",
        "is_tremor": label != "no_tremor",
        "confidence": float(min(conf, 1.0)),
        "probabilities": {c: (0.9 if c == label else 0.1 / 3) for c in CLASSES},
        "model_version": "rule_based_fallback",
    }
