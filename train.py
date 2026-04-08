"""
train.py - Model Training Module
AI-Based Intrusion Detection System (IDS)

Trains Logistic Regression, Random Forest, and XGBoost classifiers,
evaluates them, and saves trained models + preprocessing artifacts to disk.

Usage:
    python train.py                        # train on synthetic data
    python train.py --data path/to/file    # train on NSL-KDD file
    python train.py --scaler minmax        # use MinMaxScaler
"""

import argparse
import logging
import time
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logging.warning("XGBoost not installed. Skipping XGBClassifier.")

from preprocess import (
    generate_synthetic_nslkdd,
    load_nslkdd,
    full_preprocess_pipeline,
)
from utils import (
    compute_metrics,
    save_model,
    save_artifacts,
    build_metrics_table,
    export_results_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return model instances with tuned hyperparameters."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    return models


# ─────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────

def train_all_models(preprocessed: dict, verbose: bool = True) -> dict:
    """
    Train all models, evaluate on test set, save to disk.

    Parameters
    ----------
    preprocessed : dict
        Output from full_preprocess_pipeline().
    verbose : bool
        Whether to print classification reports.

    Returns
    -------
    results : dict
        {model_name: metrics_dict}
    """
    X_train = preprocessed["X_train"]
    X_test  = preprocessed["X_test"]
    y_train = preprocessed["y_train"]
    y_test  = preprocessed["y_test"]
    le      = preprocessed["label_encoder"]
    classes = list(le.classes_)

    models  = get_models()
    results = {}

    logger.info(f"=== Training {len(models)} models ===")
    logger.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"  Classes: {classes}")

    for name, model in models.items():
        logger.info(f"\n{'─'*45}")
        logger.info(f"  Training: {name}")
        t0 = time.time()

        # ── Fit ──────────────────────────────────
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        # ── Evaluate ─────────────────────────────
        y_pred  = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, classes=classes)
        metrics["train_time_s"] = train_time

        # ── Log results ──────────────────────────
        logger.info(f"  ✓ Done in {train_time}s")
        logger.info(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
        logger.info(f"    Precision: {metrics['precision']*100:.2f}%")
        logger.info(f"    Recall:    {metrics['recall']*100:.2f}%")
        logger.info(f"    F1-Score:  {metrics['f1_score']*100:.2f}%")
        if verbose:
            logger.info(f"\nClassification Report:\n{metrics['report']}")

        # ── Save model ───────────────────────────
        save_model(model, name)
        results[name] = metrics

    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IDS Model Trainer")
    parser.add_argument("--data",   type=str,   default=None,
                        help="Path to NSL-KDD .csv/.txt dataset")
    parser.add_argument("--scaler", type=str,   default="standard",
                        choices=["standard", "minmax"],
                        help="Feature scaling method")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split ratio (default 0.2)")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Synthetic sample count if no --data given")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────
    if args.data:
        df = load_nslkdd(args.data)
    else:
        logger.info("No dataset path provided – using synthetic NSL-KDD data.")
        df = generate_synthetic_nslkdd(n_samples=args.samples)

    # ── Preprocess ───────────────────────────────
    preprocessed = full_preprocess_pipeline(
        df, scaler_method=args.scaler, test_size=args.test_size
    )

    # ── Train ────────────────────────────────────
    results = train_all_models(preprocessed)

    # ── Save artifacts ───────────────────────────
    artifacts = {
        "label_encoder":   preprocessed["label_encoder"],
        "feature_encoders": preprocessed["feature_encoders"],
        "scaler":          preprocessed["scaler"],
        "feature_cols":    preprocessed["feature_cols"],
        "classes":         list(preprocessed["label_encoder"].classes_),
        "results":         results,
    }
    save_artifacts(artifacts)

    # ── Print summary table ──────────────────────
    table = build_metrics_table(results)
    print("\n" + "═" * 60)
    print("            MODEL PERFORMANCE SUMMARY")
    print("═" * 60)
    print(table.to_string())
    print("═" * 60)

    # ── Export JSON ──────────────────────────────
    export_results_json(results, "models/results.json")
    logger.info("\nAll models trained and saved successfully. ✓")


if __name__ == "__main__":
    main()