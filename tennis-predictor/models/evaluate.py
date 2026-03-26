"""
evaluate.py
-----------
Standalone evaluation script: loads the saved best_model.pkl and a features
CSV (defaulting to data/processed/features.csv), prints a full metrics report,
and generates diagnostic plots (calibration curve, ROC curve, confusion matrix).

Usage:
    python models/evaluate.py
    python models/evaluate.py --features-csv path/to/custom_features.csv
"""

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
CHARTS_DIR = MODELS_DIR / "charts"
TEST_CUTOFF_YEAR = 2023


def load_model(model_path: Path) -> tuple:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"], bundle["name"]


def load_test_data(features_csv: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(features_csv)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["date"].dt.year >= TEST_CUTOFF_YEAR]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["target"]
    return X_test, y_test


def plot_roc_curve(y_true, y_proba, model_name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    out = CHARTS_DIR / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve to %s", out)


def plot_calibration(y_true, y_proba, model_name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=10
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    out = CHARTS_DIR / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve to %s", out)


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["P2 wins", "P1 wins"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    out = CHARTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved best model.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    args = parser.parse_args()

    if not BEST_MODEL_PATH.exists():
        logger.error("No saved model at %s. Run models/train.py first.", BEST_MODEL_PATH)
        return

    model, feature_cols, model_name = load_model(BEST_MODEL_PATH)
    X_test, y_test = load_test_data(args.features_csv, feature_cols)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(f"  Evaluation: {model_name}  (test set ≥ {TEST_CUTOFF_YEAR})")
    print("=" * 50)
    print(f"  Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Log Loss   : {log_loss(y_test, y_proba):.4f}")
    print(f"  ROC-AUC    : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Brier Score: {brier_score_loss(y_test, y_proba):.4f}")
    print("=" * 50 + "\n")

    plot_roc_curve(y_test, y_proba, model_name)
    plot_calibration(y_test, y_proba, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)


if __name__ == "__main__":
    main()
