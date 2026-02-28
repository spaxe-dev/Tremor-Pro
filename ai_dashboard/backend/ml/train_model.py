"""
Training pipeline for the Random Forest tremor classifier.

Usage:
  # Generate synthetic data + train (from scratch)
  python -m ml.train_model

  # Augment with real session data from profiles.db, then train
  python -m ml.train_model --augment-from-db

  # Only generate and save synthetic CSV (for inspection)
  python -m ml.train_model --export-csv
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from ml.features import (
    extract_features_single,
    extract_features_batch,
    label_window_rule_based,
    CLASSES,
    CLASS_TO_IDX,
    FEATURE_NAMES,
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "tremor_rf.joblib"
DB_PATH = Path(os.environ.get("TREMOR_PROFILE_DB_PATH", "./profiles.db"))

# ──────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────
# Based on published tremor frequency literature:
#   - Parkinsonian resting tremor: 4–6 Hz, moderate–high power
#   - Essential tremor: 6–8 Hz, moderate power, postural
#   - Enhanced physiological: 8–12 Hz, low–moderate power
#   - No tremor: all bands at noise floor

PROFILES = {
    "no_tremor": {
        "b1_range": (0.0, 0.008),
        "b2_range": (0.0, 0.008),
        "b3_range": (0.0, 0.008),
        "norm_range": (0.01, 0.15),
    },
    "parkinsonian": {
        "b1_range": (0.5, 15.0),    # dominant: 4–6 Hz
        "b2_range": (0.01, 2.0),
        "b3_range": (0.01, 1.5),
        "norm_range": (0.05, 0.4),
    },
    "essential": {
        "b1_range": (0.01, 2.0),
        "b2_range": (0.5, 12.0),     # dominant: 6–8 Hz
        "b3_range": (0.01, 2.0),
        "norm_range": (0.05, 0.5),
    },
    "physiological": {
        "b1_range": (0.01, 1.5),
        "b2_range": (0.01, 2.0),
        "b3_range": (0.4, 10.0),     # dominant: 8–12 Hz
        "norm_range": (0.03, 0.35),
    },
}


def _sample_uniform(lo: float, hi: float, rng: np.random.Generator) -> float:
    return rng.uniform(lo, hi)


def _sample_log_uniform(lo: float, hi: float,
                         rng: np.random.Generator) -> float:
    """Sample from a log-uniform distribution (more realistic for power)."""
    if lo <= 0:
        lo = 1e-6
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def generate_synthetic_data(
    n_per_class: int = 1500,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on clinical tremor profiles.

    Returns (X, y) where X has shape (N, 7) and y has shape (N,).

    The generator creates realistic band power distributions for each
    tremor type and adds Gaussian noise for robustness.
    """
    rng = np.random.default_rng(seed)
    X_all, y_all = [], []

    for cls_name, profile in PROFILES.items():
        cls_idx = CLASS_TO_IDX[cls_name]
        for _ in range(n_per_class):
            # Use log-uniform for band powers (more realistic distribution)
            if cls_name == "no_tremor":
                b1 = _sample_uniform(*profile["b1_range"], rng)
                b2 = _sample_uniform(*profile["b2_range"], rng)
                b3 = _sample_uniform(*profile["b3_range"], rng)
            else:
                b1 = _sample_log_uniform(*profile["b1_range"], rng)
                b2 = _sample_log_uniform(*profile["b2_range"], rng)
                b3 = _sample_log_uniform(*profile["b3_range"], rng)

            norm = _sample_uniform(*profile["norm_range"], rng)

            # Add noise
            b1 = max(0, b1 + rng.normal(0, noise_std))
            b2 = max(0, b2 + rng.normal(0, noise_std))
            b3 = max(0, b3 + rng.normal(0, noise_std))
            norm = max(0, norm + rng.normal(0, noise_std * 0.5))

            feats = extract_features_single(b1, b2, b3, norm)
            X_all.append(feats)
            y_all.append(cls_idx)

    # Also generate "borderline" samples near decision boundaries
    # These help the RF learn tricky cases
    for _ in range(n_per_class // 3):
        # Borderline no-tremor / weak-tremor
        b1 = rng.uniform(0.005, 0.05)
        b2 = rng.uniform(0.005, 0.05)
        b3 = rng.uniform(0.005, 0.05)
        norm = rng.uniform(0.02, 0.2)
        feats = extract_features_single(b1, b2, b3, norm)
        X_all.append(feats)
        y_all.append(CLASS_TO_IDX["no_tremor"])

        # Borderline voluntary movement (high norm, low power)
        b1 = rng.uniform(0.01, 0.5)
        b2 = rng.uniform(0.01, 0.5)
        b3 = rng.uniform(0.01, 0.5)
        norm = rng.uniform(0.7, 1.5)
        feats = extract_features_single(b1, b2, b3, norm)
        X_all.append(feats)
        y_all.append(CLASS_TO_IDX["no_tremor"])

        # Mixed tremor (overlapping bands → classify by dominant)
        dominant_cls = rng.choice(["parkinsonian", "essential", "physiological"])
        base = rng.uniform(0.3, 3.0)
        if dominant_cls == "parkinsonian":
            b1, b2, b3 = base * rng.uniform(1.5, 3), base * rng.uniform(0.3, 0.9), base * rng.uniform(0.2, 0.8)
        elif dominant_cls == "essential":
            b1, b2, b3 = base * rng.uniform(0.3, 0.9), base * rng.uniform(1.5, 3), base * rng.uniform(0.2, 0.8)
        else:
            b1, b2, b3 = base * rng.uniform(0.2, 0.8), base * rng.uniform(0.3, 0.9), base * rng.uniform(1.5, 3)
        norm = rng.uniform(0.05, 0.4)
        feats = extract_features_single(b1, b2, b3, norm)
        X_all.append(feats)
        y_all.append(CLASS_TO_IDX[dominant_cls])

    X = np.array(X_all)
    y = np.array(y_all)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ──────────────────────────────────────────────
# Augment from real session data (profiles.db)
# ──────────────────────────────────────────────
def load_real_data_from_db() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load stored sessions from profiles.db and extract features
    using the rule-based labeler as weak supervision.
    """
    if not DB_PATH.exists():
        print(f"  ⚠ Database not found at {DB_PATH}")
        return None

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT raw_summary_json FROM sessions WHERE raw_summary_json IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("  ⚠ No sessions found in database")
        return None

    X_all, y_all = [], []
    for row in rows:
        try:
            raw = json.loads(row["raw_summary_json"])
        except (json.JSONDecodeError, KeyError):
            continue

        # Extract per-session band powers and create a sample
        fp = raw.get("frequency_profile", {})
        ip = raw.get("intensity_profile", {})
        bp = fp.get("band_power_mean", {})

        b1 = float(bp.get("hz_4_6", 0))
        b2 = float(bp.get("hz_6_8", 0))
        b3 = float(bp.get("hz_8_12", 0))
        rms = float(ip.get("rms_mean", 0))

        if b1 == 0 and b2 == 0 and b3 == 0:
            continue

        label = label_window_rule_based(b1, b2, b3, rms)
        feats = extract_features_single(b1, b2, b3, rms)
        X_all.append(feats)
        y_all.append(CLASS_TO_IDX[label])

    if not X_all:
        print("  ⚠ No usable sessions extracted")
        return None

    print(f"  ✓ Extracted {len(X_all)} samples from {len(rows)} sessions")
    return np.array(X_all), np.array(y_all)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 150,
    max_depth: int | None = 12,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the feature matrix X and labels y.

    Hyperparameters chosen for a small feature space (7 features):
      - 150 trees: enough for stable predictions, fast inference
      - max_depth=12: prevents overfitting on synthetic data
      - min_samples_leaf=3: regularization
      - class_weight='balanced': handles class imbalance
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING RANDOM FOREST")
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | Classes: {len(set(y))}")
    print(f"{'='*60}\n")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        min_samples_split=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"  Test Accuracy: {accuracy:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=CLASSES,
        digits=3,
    ))

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Feature importances
    print("\n  Feature Importances:")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {name:>20s}: {imp:.4f}  {bar}")

    # Confusion matrix
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    header = "".join(f"{c[:8]:>10s}" for c in CLASSES)
    print(f"    {'Predicted →':>20s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10d}" for v in row)
        print(f"    {CLASSES[i]:>20s}{row_str}")

    return clf


def save_model(clf: RandomForestClassifier) -> Path:
    """Save the trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\n  ✓ Model saved to {MODEL_PATH} ({size_kb:.0f} KB)")
    return MODEL_PATH


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train tremor RF classifier")
    parser.add_argument("--augment-from-db", action="store_true",
                        help="Augment synthetic data with real sessions from profiles.db")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export synthetic data to CSV for inspection")
    parser.add_argument("--n-per-class", type=int, default=1500,
                        help="Number of synthetic samples per class (default: 1500)")
    parser.add_argument("--n-estimators", type=int, default=150,
                        help="Number of trees (default: 150)")
    parser.add_argument("--max-depth", type=int, default=12,
                        help="Max tree depth (default: 12)")
    args = parser.parse_args()

    # Step 1: Generate synthetic data
    print("► Generating synthetic training data...")
    X_synth, y_synth = generate_synthetic_data(n_per_class=args.n_per_class)
    print(f"  ✓ {len(X_synth)} synthetic samples generated")
    print(f"    Class distribution: {dict(zip(*np.unique(y_synth, return_counts=True)))}")

    X, y = X_synth, y_synth

    # Step 2: Optionally augment with real data
    if args.augment_from_db:
        print("\n► Loading real session data from database...")
        real = load_real_data_from_db()
        if real is not None:
            X_real, y_real = real
            # Upsample real data (weight it 3× since it's real)
            X_real_up = np.tile(X_real, (3, 1))
            y_real_up = np.tile(y_real, 3)
            X = np.vstack([X, X_real_up])
            y = np.concatenate([y, y_real_up])
            print(f"  ✓ Total training set: {len(X)} samples")

    # Step 3: Export CSV (optional)
    if args.export_csv:
        import csv
        csv_path = MODEL_DIR / "training_data.csv"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES + ["label"])
            for xi, yi in zip(X, y):
                writer.writerow(list(xi) + [CLASSES[yi]])
        print(f"\n  ✓ CSV exported to {csv_path}")

    # Step 4: Train
    clf = train_random_forest(X, y,
                               n_estimators=args.n_estimators,
                               max_depth=args.max_depth)

    # Step 5: Save
    save_model(clf)

    print(f"\n{'='*60}")
    print("  DONE! Model is ready for inference.")
    print("  Start the backend with: uvicorn main:app --reload")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
