"""
Feature extraction for tremor classification.

Takes raw window data from the ESP32 (band powers + meanNorm)
and produces a fixed-length feature vector for the Random Forest.

Features per window (7 total):
  0. b1          — band power 4–6 Hz (Parkinsonian)
  1. b2          — band power 6–8 Hz (Essential)
  2. b3          — band power 8–12 Hz (Physiological)
  3. total_power — b1 + b2 + b3
  4. meanNorm    — mean acceleration magnitude
  5. dom_ratio   — max(b1,b2,b3) / (min(b1,b2,b3) + ε)
  6. spectral_centroid — weighted average frequency band index
"""

import numpy as np

FEATURE_NAMES = [
    "b1", "b2", "b3",
    "total_power", "meanNorm",
    "dom_ratio", "spectral_centroid",
]

# Class labels
CLASSES = ["no_tremor", "parkinsonian", "essential", "physiological"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Threshold below which we consider "no tremor" for labeling
NOISE_FLOOR = 0.01


def extract_features_single(b1: float, b2: float, b3: float,
                             mean_norm: float = 0.0) -> np.ndarray:
    """
    Extract the feature vector from a single ESP32 window.
    Returns a 1D array of shape (7,).
    """
    eps = 1e-6
    total = b1 + b2 + b3
    bands = np.array([b1, b2, b3])
    dom_ratio = float(bands.max() / (bands.min() + eps))
    # Spectral centroid: 0 = low freq dominant, 2 = high freq dominant
    spectral_centroid = float(
        np.dot(bands, [0, 1, 2]) / (total + eps)
    )
    return np.array([
        b1, b2, b3,
        total, mean_norm,
        dom_ratio, spectral_centroid,
    ], dtype=np.float64)


def extract_features_batch(windows: list[dict]) -> np.ndarray:
    """
    Extract features from a list of window dicts.
    Each dict must have keys: b1, b2, b3, and optionally meanNorm.
    Returns a 2D array of shape (N, 7).
    """
    rows = []
    for w in windows:
        rows.append(extract_features_single(
            b1=float(w.get("b1", 0)),
            b2=float(w.get("b2", 0)),
            b3=float(w.get("b3", 0)),
            mean_norm=float(w.get("meanNorm", 0)),
        ))
    return np.array(rows)


def label_window_rule_based(b1: float, b2: float, b3: float,
                             mean_norm: float = 0.0) -> str:
    """
    Assign a label using deterministic rules (mirrors ESP32 classify()).
    Used for generating weak labels from existing data.
    """
    a1 = b1 if b1 > NOISE_FLOOR else 0
    a2 = b2 if b2 > NOISE_FLOOR else 0
    a3 = b3 if b3 > NOISE_FLOOR else 0
    total = a1 + a2 + a3

    if total < NOISE_FLOOR:
        return "no_tremor"

    # Voluntary movement check (same as ESP32)
    if mean_norm > 0.7 and total < 5:
        return "no_tremor"

    if a1 > a2 and a1 > a3 and a1 > 0.3:
        return "parkinsonian"
    elif a2 > a1 and a2 > a3 and a2 > 0.3:
        return "essential"
    elif a3 > a1 and a3 > a2 and a3 > 0.3:
        return "physiological"
    else:
        # Weak / mixed signal — fall back to dominant band
        bands = {"parkinsonian": a1, "essential": a2, "physiological": a3}
        if total > 0.05:
            return max(bands, key=bands.get)
        return "no_tremor"
