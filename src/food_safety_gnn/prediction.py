"""Leakage-safe temporal splits, feature joins, and predictive evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from food_safety_gnn.provenance import atomic_write_json, utc_timestamp


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Chronological split cutoffs for next-inspection prediction."""

    train_end: str = "2016-12-31"
    validation_end: str = "2017-12-31"
    test_end: str = "2018-09-30"
    target_horizon_days: int = 365


@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier training settings selected before test evaluation."""

    random_state: int = 42
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_min_child_weight: int = 5
    mlp_hidden_layer_sizes: tuple[int, ...] = (128, 64)
    mlp_max_iter: int = 200


def assign_temporal_split(
    labels: pd.DataFrame, config: TemporalSplitConfig
) -> pd.DataFrame:
    """Assign train/validation/test using the anchor inspection date only.

    Features and embeddings for a row must still be constructed from information
    available strictly before that anchor date.
    """
    frame = labels.copy()
    frame["Inspection Date"] = pd.to_datetime(frame["Inspection Date"])
    train_end = pd.Timestamp(config.train_end)
    validation_end = pd.Timestamp(config.validation_end)
    test_end = pd.Timestamp(config.test_end)
    # Require the label window to mature before the source ends.
    maturity = pd.Timedelta(days=config.target_horizon_days)
    max_label_date = frame["next_inspection_date"].max()
    if pd.isna(max_label_date):
        raise ValueError("Labeled table has no next_inspection_date values.")

    def _split(row: pd.Series) -> str | None:
        anchor = row["Inspection Date"]
        if anchor + maturity > max_label_date:
            return None
        if anchor <= train_end:
            return "train"
        if anchor <= validation_end:
            return "validation"
        if anchor <= test_end:
            return "test"
        return None

    frame["split"] = frame.apply(_split, axis=1)
    return frame.dropna(subset=["split"]).reset_index(drop=True)


def join_embeddings(
    labels: pd.DataFrame, embeddings: pd.DataFrame
) -> pd.DataFrame:
    """Join snapshot embeddings to labeled anchors by entity_id."""
    embedding_cols = [
        column for column in embeddings.columns if column.startswith("emb_")
    ]
    if not embedding_cols:
        raise ValueError("Embedding table has no emb_* columns.")
    merged = labels.merge(
        embeddings[["entity_id", *embedding_cols]],
        on="entity_id",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No labeled rows joined to embeddings.")
    return merged


def _feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    embedding_cols = [column for column in frame.columns if column.startswith("emb_")]
    tabular_candidates = [
        "prior_inspection_count",
        "prior_fail_count",
        "prior_fail_rate",
        "days_since_last_inspection",
    ]
    tabular_cols = [column for column in tabular_candidates if column in frame.columns]
    feature_cols = embedding_cols + tabular_cols
    matrix = frame[feature_cols].astype(np.float32).to_numpy()
    return matrix, feature_cols


def _scale_pos_weight(y_train: np.ndarray) -> float:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if positives <= 0:
        return 1.0
    return negatives / positives


def train_classifiers(
    dataset: pd.DataFrame, config: ClassifierConfig
) -> dict[str, Any]:
    """Train XGBoost (primary) and a small MLP comparator on the training split only."""
    train = dataset.loc[dataset["split"].eq("train")].copy()
    validation = dataset.loc[dataset["split"].eq("validation")].copy()
    test = dataset.loc[dataset["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Temporal splits are empty. Adjust split cutoffs or ensure embeddings "
            "cover entities."
        )

    x_train, feature_names = _feature_matrix(train)
    y_train = train["target_label"].astype(int).to_numpy()
    x_validation, _ = _feature_matrix(validation)
    y_validation = validation["target_label"].astype(int).to_numpy()
    x_test, _ = _feature_matrix(test)
    y_test = test["target_label"].astype(int).to_numpy()

    pos_weight = _scale_pos_weight(y_train)
    models: dict[str, Any] = {
        "xgboost": XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            min_child_weight=config.xgb_min_child_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=pos_weight,
            random_state=config.random_state,
            n_jobs=-1,
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=config.mlp_hidden_layer_sizes,
                        max_iter=config.mlp_max_iter,
                        random_state=config.random_state,
                        early_stopping=True,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, Any] = {
        "feature_names": feature_names,
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_validation.mean()),
            "test_positive_rate": float(y_test.mean()),
            "scale_pos_weight": float(pos_weight),
        },
        "models": {},
    }

    best_model_name = None
    best_validation_pr_auc = -1.0
    fitted_models: dict[str, Any] = {}
    best_thresholds: dict[str, float] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        validation_proba = model.predict_proba(x_validation)[:, 1]
        threshold = _best_f1_threshold(y_validation, validation_proba)
        best_thresholds[name] = threshold
        validation_pred = (validation_proba >= threshold).astype(int)
        validation_metrics = _binary_metrics(
            y_validation, validation_pred, validation_proba
        )
        validation_metrics["decision_threshold"] = float(threshold)
        results["models"][name] = {"validation": validation_metrics}
        if validation_metrics["pr_auc"] > best_validation_pr_auc:
            best_validation_pr_auc = validation_metrics["pr_auc"]
            best_model_name = name

    assert best_model_name is not None
    selected = fitted_models[best_model_name]
    selected_threshold = best_thresholds[best_model_name]
    test_proba = selected.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= selected_threshold).astype(int)
    test_metrics = _binary_metrics(y_test, test_pred, test_proba)
    test_metrics["decision_threshold"] = float(selected_threshold)
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1])
    results["selected_model"] = best_model_name
    results["models"][best_model_name]["test"] = test_metrics
    results["confusion_matrix"] = {
        "labels": [0, 1],
        "matrix": matrix.tolist(),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "decision_threshold": float(selected_threshold),
    }
    results["test_predictions"] = pd.DataFrame(
        {
            "entity_id": test["entity_id"].to_numpy(),
            "Inspection Date": test["Inspection Date"].to_numpy(),
            "target_label": y_test,
            "probability": test_proba,
            "prediction": test_pred,
            "model": best_model_name,
            "decision_threshold": selected_threshold,
        }
    )
    results["fitted_models"] = fitted_models
    results["created_at"] = utc_timestamp()
    results["config"] = asdict(config)
    return results


def _best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Pick a validation threshold maximizing F1; lock before test evaluation."""
    candidates = np.unique(np.quantile(y_proba, np.linspace(0.05, 0.95, 19)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def export_prediction_artifacts(
    results: dict[str, Any], output_directory: Path
) -> dict[str, str]:
    """Persist metrics, predictions, and a compact run manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = output_directory / "test_predictions.parquet"
    metrics_path = output_directory / "metrics.json"
    results["test_predictions"].to_parquet(predictions_path, index=False)
    serializable = {
        key: value
        for key, value in results.items()
        if key not in {"test_predictions", "fitted_models"}
    }
    atomic_write_json(metrics_path, serializable)
    return {
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
    }
