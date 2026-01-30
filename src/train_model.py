"""Train ML models to predict Super Bowl setlist inclusion probability."""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from .config import DATA_PROCESSED_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# === CONFIGURATION ===

# Columns to exclude from features (identifiers and targets)
EXCLUDE_COLUMNS = [
    "song_name",
    "song_id",
    # Target-related columns
    "flagship_appearances",
    "flagship_shows_total",
    "flagship_appearance_rate",
    "appeared_in_flagship_show",
    "superbowl_likelihood_score",
    "combined_score",
    "likely_setlist_candidate",
    # Holdout columns (would leak validation data)
    "holdout_appeared_in_pr",
    "holdout_pr_appearance_rate",
    # PR residency columns (used for holdout split)
    "performed_at_pr_residency",
    "pr_residency_frequency",
]

# Categorical columns to one-hot encode
CATEGORICAL_COLUMNS = ["duration_bucket", "position_category"]

# Target configuration
TOP_PERCENTILE = 0.15  # Top 15% = positive class


def load_and_prepare_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load training data and prepare features/target.

    Returns:
        - df: Original dataframe
        - X: Feature matrix
        - y: Binary target
        - feature_names: List of feature column names
    """
    logger.info("Loading training data...")
    df = pd.read_csv(DATA_PROCESSED_DIR / "training_data.csv")
    logger.info(f"Loaded {len(df)} songs with {len(df.columns)} columns")

    # Create binary target from combined_score
    threshold = df["combined_score"].quantile(1 - TOP_PERCENTILE)
    df["target"] = (df["combined_score"] >= threshold).astype(int)
    logger.info(f"Target threshold (top {TOP_PERCENTILE*100:.0f}%): {threshold:.2f}")
    logger.info(f"Positive class: {df['target'].sum()} songs ({df['target'].mean()*100:.1f}%)")

    # Define feature columns
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS + ["target"]]
    logger.info(f"Initial feature columns: {len(feature_cols)}")

    # Separate numeric and categorical
    numeric_cols = [col for col in feature_cols if col not in CATEGORICAL_COLUMNS]

    # Create feature dataframe
    X_df = df[feature_cols].copy()

    # One-hot encode categorical columns
    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in X_df.columns:
            dummies = pd.get_dummies(X_df[cat_col], prefix=cat_col, dummy_na=True)
            X_df = pd.concat([X_df.drop(cat_col, axis=1), dummies], axis=1)

    # Handle missing values
    X_df = X_df.fillna(0)

    # Get final feature names
    feature_names = X_df.columns.tolist()
    logger.info(f"Final feature count (after encoding): {len(feature_names)}")

    # Convert to numpy arrays
    X = X_df.values.astype(float)
    y = df["target"].values

    return df, X, y, feature_names


def create_holdout_split(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation split based on PR residency holdout.

    Songs that appeared in PR residency are used for validation.
    """
    # Use holdout_appeared_in_pr for validation set
    val_mask = df["holdout_appeared_in_pr"] == 1
    train_mask = ~val_mask

    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    logger.info(f"Training set: {len(X_train)} songs")
    logger.info(f"Validation set (PR residency): {len(X_val)} songs")
    logger.info(f"Validation positive rate: {y_val.mean()*100:.1f}%")

    return X_train, X_val, y_train, y_val, train_mask, val_mask


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Any]:
    """Train multiple models for comparison."""

    # Standardize features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}

    # 1. Logistic Regression (baseline)
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = {
        "model": lr,
        "scaler": scaler,
        "needs_scaling": True
    }

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = {
        "model": rf,
        "scaler": None,
        "needs_scaling": False
    }

    # 3. Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = {
        "model": gb,
        "scaler": None,
        "needs_scaling": False
    }

    return models


def evaluate_model(
    model_info: Dict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """Evaluate a single model on validation set."""

    model = model_info["model"]

    # Apply scaling if needed
    if model_info["needs_scaling"]:
        X_eval = model_info["scaler"].transform(X_val)
    else:
        X_eval = X_val

    # Predictions
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    # Metrics
    metrics = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0
    }

    return metrics


def cross_validate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Run 5-fold cross-validation on all models."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model_info in models.items():
        logger.info(f"Cross-validating {name}...")

        model = model_info["model"]

        # For scaled models, we need to handle this differently
        if model_info["needs_scaling"]:
            # Scale full dataset
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_cv = X_scaled
        else:
            X_cv = X

        # Cross-validation scores
        f1_scores = cross_val_score(model, X_cv, y, cv=cv, scoring="f1")
        auc_scores = cross_val_score(model, X_cv, y, cv=cv, scoring="roc_auc")

        cv_results[name] = {
            "cv_f1_mean": f1_scores.mean(),
            "cv_f1_std": f1_scores.std(),
            "cv_auc_mean": auc_scores.mean(),
            "cv_auc_std": auc_scores.std()
        }

    return cv_results


def get_feature_importance(
    model_info: Dict,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """Extract feature importances from model."""

    model = model_info["model"]

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models - use absolute coefficient values
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    # Create dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df


def generate_predictions(
    best_model_info: Dict,
    X: np.ndarray,
    df: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """Generate predictions for all songs using best model."""

    model = best_model_info["model"]

    # Apply scaling if needed
    if best_model_info["needs_scaling"]:
        X_pred = best_model_info["scaler"].transform(X)
    else:
        X_pred = X

    # Get probabilities
    probabilities = model.predict_proba(X_pred)[:, 1]

    # Create predictions dataframe
    predictions = pd.DataFrame({
        "song_name": df["song_name"],
        "song_id": df["song_id"],
        "ml_probability": probabilities,
        "ml_rank": pd.Series(probabilities).rank(ascending=False).astype(int),
        "combined_score": df["combined_score"],
        "combined_rank": df["combined_score"].rank(ascending=False).astype(int),
        "is_latest_album": df["is_latest_album"],
        "times_performed_live": df["times_performed_live"],
        "cultural_significance": df["cultural_significance"],
        "in_halftime_trailer": df["in_halftime_trailer"]
    })

    # Sort by ML probability
    predictions = predictions.sort_values("ml_probability", ascending=False)

    return predictions


def save_model(
    best_model_info: Dict,
    best_model_name: str,
    feature_names: List[str],
    metrics: Dict,
    cv_results: Dict
) -> None:
    """Save trained model and model card."""

    # Save model
    model_path = MODELS_DIR / "setlist_predictor.pkl"
    model_data = {
        "model": best_model_info["model"],
        "scaler": best_model_info["scaler"],
        "needs_scaling": best_model_info["needs_scaling"],
        "feature_names": feature_names,
        "model_name": best_model_name,
        "trained_at": datetime.now().isoformat()
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved model to: {model_path}")

    # Create model card
    model_card = {
        "model_name": best_model_name,
        "task": "Binary classification - Super Bowl setlist inclusion",
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "source": "data/processed/training_data.csv",
            "total_samples": 342,
            "positive_class_definition": f"Top {TOP_PERCENTILE*100:.0f}% by combined_score"
        },
        "features": {
            "count": len(feature_names),
            "names": feature_names
        },
        "validation_metrics": {
            "holdout": "Puerto Rico residency shows",
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "auc_roc": round(metrics["auc_roc"], 4)
        },
        "cross_validation": {
            "folds": 5,
            "f1_mean": round(cv_results[best_model_name]["cv_f1_mean"], 4),
            "f1_std": round(cv_results[best_model_name]["cv_f1_std"], 4),
            "auc_mean": round(cv_results[best_model_name]["cv_auc_mean"], 4),
            "auc_std": round(cv_results[best_model_name]["cv_auc_std"], 4)
        }
    }

    card_path = MODELS_DIR / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    logger.info(f"Saved model card to: {card_path}")


def compare_with_constraint_predictions(predictions: pd.DataFrame) -> None:
    """Compare ML predictions with constraint-based optimizer predictions."""

    constraint_path = DATA_PROCESSED_DIR / "predicted_setlist.json"

    if not constraint_path.exists():
        logger.warning("Constraint-based predictions not found, skipping comparison")
        return

    with open(constraint_path) as f:
        constraint_pred = json.load(f)

    constraint_songs = set(s["song_name"] for s in constraint_pred["setlist"])

    # Get top 8 by ML probability (matching constraint output count)
    ml_top8 = set(predictions.head(8)["song_name"].tolist())

    # Calculate overlap
    overlap = constraint_songs & ml_top8

    print("\n" + "=" * 70)
    print("COMPARISON: ML vs CONSTRAINT-BASED PREDICTIONS")
    print("=" * 70)

    print(f"\nConstraint-based setlist ({len(constraint_songs)} songs):")
    for song in sorted(constraint_songs):
        in_ml = "✓" if song in ml_top8 else " "
        print(f"  [{in_ml}] {song}")

    print(f"\nML top 8 predictions:")
    for i, row in predictions.head(8).iterrows():
        in_constraint = "✓" if row["song_name"] in constraint_songs else " "
        print(f"  [{in_constraint}] {row['song_name']:<30} (prob: {row['ml_probability']:.3f})")

    print(f"\n" + "-" * 50)
    print(f"Agreement: {len(overlap)}/8 songs ({len(overlap)/8*100:.0f}%)")
    print(f"Overlapping songs: {sorted(overlap)}")
    print("=" * 70)


def print_results(
    all_metrics: Dict[str, Dict],
    cv_results: Dict[str, Dict],
    importance_df: pd.DataFrame,
    predictions: pd.DataFrame
) -> None:
    """Print comprehensive results summary."""

    print("\n" + "=" * 70)
    print("MODEL TRAINING RESULTS")
    print("=" * 70)

    # Model comparison table
    print("\n" + "-" * 50)
    print("MODEL COMPARISON (Holdout Validation)")
    print("-" * 50)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 65)

    for name, metrics in all_metrics.items():
        print(f"{name:<25} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1']:>10.3f} {metrics['auc_roc']:>10.3f}")

    # Cross-validation results
    print("\n" + "-" * 50)
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print("-" * 50)
    print(f"{'Model':<25} {'F1 Mean':>10} {'F1 Std':>10} {'AUC Mean':>10} {'AUC Std':>10}")
    print("-" * 65)

    for name, cv in cv_results.items():
        print(f"{name:<25} {cv['cv_f1_mean']:>10.3f} {cv['cv_f1_std']:>10.3f} "
              f"{cv['cv_auc_mean']:>10.3f} {cv['cv_auc_std']:>10.3f}")

    # Feature importance
    print("\n" + "-" * 50)
    print("TOP 10 FEATURE IMPORTANCES (Best Model)")
    print("-" * 50)

    for i, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<35} {row['importance']:.4f} {bar}")

    # Top predictions
    print("\n" + "-" * 50)
    print("TOP 15 PREDICTED SONGS")
    print("-" * 50)
    print(f"{'Rank':<5} {'Song':<35} {'ML Prob':>10} {'Combined':>10}")
    print("-" * 60)

    for _, row in predictions.head(15).iterrows():
        flags = []
        if row["in_halftime_trailer"] == 1:
            flags.append("CONFIRMED")
        if row["is_latest_album"] == 1:
            flags.append("NEW")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"{int(row['ml_rank']):<5} {row['song_name']:<35} "
              f"{row['ml_probability']:>10.3f} {row['combined_score']:>10.1f}{flag_str}")

    print("\n" + "=" * 70)


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Training ML Models for Setlist Prediction")
    logger.info("=" * 60)

    # 1. Load and prepare data
    df, X, y, feature_names = load_and_prepare_data()

    # 2. Create holdout split
    X_train, X_val, y_train, y_val, train_mask, val_mask = create_holdout_split(df, X, y)

    # 3. Train models
    models = train_models(X_train, y_train, feature_names)

    # 4. Evaluate on holdout
    logger.info("Evaluating on PR residency holdout...")
    all_metrics = {}
    for name, model_info in models.items():
        metrics = evaluate_model(model_info, X_val, y_val, name)
        all_metrics[name] = metrics
        logger.info(f"{name}: F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}")

    # 5. Cross-validation
    logger.info("Running cross-validation...")
    cv_results = cross_validate_models(models, X, y)

    # 6. Select best model (by holdout AUC)
    best_model_name = max(all_metrics, key=lambda x: all_metrics[x]["auc_roc"])
    best_model_info = models[best_model_name]
    logger.info(f"Best model: {best_model_name}")

    # 7. Feature importance
    importance_df = get_feature_importance(best_model_info, feature_names, best_model_name)

    # 8. Generate predictions for all songs
    # Retrain on full dataset for final predictions
    logger.info("Retraining best model on full dataset...")
    if best_model_info["needs_scaling"]:
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X)
        best_model_info["model"].fit(X_full_scaled, y)
        best_model_info["scaler"] = scaler
    else:
        best_model_info["model"].fit(X, y)

    predictions = generate_predictions(best_model_info, X, df, feature_names)

    # Save predictions
    pred_path = DATA_PROCESSED_DIR / "model_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to: {pred_path}")

    # 9. Save model and model card
    save_model(
        best_model_info,
        best_model_name,
        feature_names,
        all_metrics[best_model_name],
        cv_results
    )

    # 10. Print results
    print_results(all_metrics, cv_results, importance_df, predictions)

    # 11. Compare with constraint-based predictions
    compare_with_constraint_predictions(predictions)


if __name__ == "__main__":
    main()
