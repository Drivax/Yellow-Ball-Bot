"""
train.py
--------
Trains and compares four classifiers to predict tennis match winners:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Training respects temporal order — the test set contains only matches from
2023 onward so that the model is never evaluated on data used for training.
Hyperparameters are tuned with Optuna (XGBoost and LightGBM) and the best
overall model is saved to models/best_model.pkl.

Usage:
    python models/train.py
    python models/train.py --n-trials 50 --no-optuna
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
MODELS_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
CHARTS_DIR = MODELS_DIR / "charts"

# ---------------------------------------------------------------------------
# Feature columns used for training
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "elo_p1", "elo_p2", "elo_diff",
    "h2h_win_rate_p1",
    "surface_win_rate_p1_52w", "surface_win_rate_p2_52w",
    "avg_rank_p1", "avg_rank_p2", "rank_diff",
    "form_p1", "form_p2", "form_diff",
    "tourney_win_rate_p1", "tourney_win_rate_p2",
    "age_diff",
    "days_since_last_p1", "days_since_last_p2",
    "surface_enc", "round_enc", "tourney_level_enc",
]
TARGET_COL = "target"
TEST_CUTOFF_YEAR = 2023


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(features_csv: Path = FEATURES_CSV) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load features, split chronologically into train and test."""
    df = pd.read_csv(features_csv)
    df["date"] = pd.to_datetime(df["date"])

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in feature CSV: {missing_cols}")

    train_df = df[df["date"].dt.year < TEST_CUTOFF_YEAR].copy()
    test_df = df[df["date"].dt.year >= TEST_CUTOFF_YEAR].copy()

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df[TARGET_COL]

    logger.info(
        "Train: %d rows (%d-%d), Test: %d rows (%d+)",
        len(X_train),
        train_df["date"].dt.year.min(),
        train_df["date"].dt.year.max(),
        len(X_test),
        TEST_CUTOFF_YEAR,
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
    }
    logger.info(
        "[%s] Accuracy=%.4f | LogLoss=%.4f | AUC=%.4f | Brier=%.4f",
        name, metrics["accuracy"], metrics["log_loss"],
        metrics["roc_auc"], metrics["brier_score"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ])


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)


def build_xgboost(**kwargs) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def build_lightgbm(**kwargs) -> lgb.LGBMClassifier:
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30) -> dict:
    """Tune XGBoost hyperparameters using Optuna."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = build_xgboost(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGBoost best AUC (CV): %.4f", study.best_value)
    return study.best_params


def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30) -> dict:
    """Tune LightGBM hyperparameters using Optuna."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        }
        model = build_lightgbm(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("LightGBM best AUC (CV): %.4f", study.best_value)
    return study.best_params


# ---------------------------------------------------------------------------
# Feature importance plots
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], model_name: str) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf")
        if inner is None or not hasattr(inner, "coef_"):
            return
        importances = np.abs(inner.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    out_path = CHARTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance chart to %s", out_path)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train tennis match prediction models.")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model.")
    parser.add_argument("--no-optuna", action="store_true", help="Skip hyperparameter tuning.")
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        logger.error("Features CSV not found at %s. Run preprocessing/feature_engineering.py first.", FEATURES_CSV)
        return

    X_train, y_train, X_test, y_test = load_data()

    use_optuna = OPTUNA_AVAILABLE and not args.no_optuna

    # --- Build models ---
    models: dict = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest": build_random_forest(),
    }

    if use_optuna:
        logger.info("Tuning XGBoost with Optuna (%d trials)…", args.n_trials)
        xgb_params = tune_xgboost(X_train, y_train, n_trials=args.n_trials)
        logger.info("Tuning LightGBM with Optuna (%d trials)…", args.n_trials)
        lgb_params = tune_lightgbm(X_train, y_train, n_trials=args.n_trials)
        models["XGBoost"] = build_xgboost(**xgb_params)
        models["LightGBM"] = build_lightgbm(**lgb_params)
    else:
        models["XGBoost"] = build_xgboost()
        models["LightGBM"] = build_lightgbm()

    # --- Train & evaluate ---
    results = []
    trained_models: dict = {}

    for name, model in tqdm(models.items(), desc="Training models"):
        logger.info("Training %s…", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = model
        plot_feature_importance(model, FEATURE_COLS, name)

    # --- Summary table ---
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    logger.info("\n%s", results_df.to_string(index=False))

    # --- Save best model ---
    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": FEATURE_COLS, "name": best_name}, f)
    logger.info("Saved best model (%s) to %s", best_name, BEST_MODEL_PATH)

    # --- Save evaluation results ---
    eval_path = MODELS_DIR / "evaluation_results.csv"
    results_df.to_csv(eval_path, index=False)
    logger.info("Saved evaluation results to %s", eval_path)


if __name__ == "__main__":
    main()
