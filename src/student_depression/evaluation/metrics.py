"""Comparable metrics for benchmark classifiers."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from student_depression.config import SETTINGS


def threshold_grid(
    start: float = SETTINGS.threshold.start,
    stop: float = SETTINGS.threshold.stop,
    step: float = SETTINGS.threshold.step,
) -> np.ndarray:
    """Return the inclusive validation-threshold grid configured by the paper."""
    if step <= 0:
        message = f"Threshold step must be positive, received {step}"
        raise ValueError(message)
    if start > stop:
        message = f"Threshold start must be <= stop, received {start} > {stop}"
        raise ValueError(message)

    steps = int(np.floor((stop - start) / step))
    thresholds = start + np.arange(steps + 1) * step
    if not np.isclose(thresholds[-1], stop) and thresholds[-1] < stop:
        thresholds = np.append(thresholds, stop)
    return np.round(thresholds, 10)


def positive_class_scores(
    estimator: Any,
    features: pd.DataFrame,
    positive_class: int = 1,
) -> np.ndarray:
    """Return positive-class probabilities or probability-like scores."""
    if hasattr(estimator, "predict_proba"):
        classes = list(estimator.classes_)
        if positive_class not in classes:
            message = f"Positive class {positive_class!r} not present in fitted estimator"
            raise ValueError(message)
        positive_class_index = classes.index(positive_class)
        return np.asarray(estimator.predict_proba(features)[:, positive_class_index])

    if hasattr(estimator, "decision_function"):
        decisions = np.asarray(estimator.decision_function(features))
        if decisions.ndim > 1:
            decisions = decisions[:, -1]
        return 1.0 / (1.0 + np.exp(-decisions))

    message = "Estimator must expose predict_proba or decision_function for threshold search"
    raise ValueError(message)


def predictions_at_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert positive-class scores into binary predictions."""
    return (np.asarray(scores) >= threshold).astype(int)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def classification_metrics_from_scores(
    target: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Evaluate binary predictions made at one threshold."""
    predictions = predictions_at_threshold(scores, threshold)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "specificity": _safe_ratio(true_negative, true_negative + false_positive),
        "negative_predictive_value": _safe_ratio(true_negative, true_negative + false_negative),
        "f1": f1_score(target, predictions, zero_division=0),
        "f2": fbeta_score(target, predictions, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(target, scores),
        "average_precision": average_precision_score(target, scores),
        "mcc": matthews_corrcoef(target, predictions),
        "cohen_kappa": cohen_kappa_score(target, predictions),
        "brier_loss": brier_score_loss(target, scores),
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_positive": float(true_positive),
    }


def select_threshold(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[float, dict[str, float]]:
    """Choose the validation threshold that maximizes F2, then configured tie-breakers."""
    scores = positive_class_scores(estimator, features)
    metric_names = (SETTINGS.threshold.optimize, *SETTINGS.threshold.tie_breakers)
    best_threshold = SETTINGS.threshold.start
    best_metrics = classification_metrics_from_scores(target, scores, best_threshold)
    best_key = tuple(best_metrics[name] for name in metric_names)

    for threshold in threshold_grid()[1:]:
        metrics = classification_metrics_from_scores(target, scores, float(threshold))
        key = tuple(metrics[name] for name in metric_names)
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key

    return best_threshold, best_metrics


def classification_metrics(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate one fitted binary classifier on the shared holdout set."""
    scores = positive_class_scores(estimator, features)
    return classification_metrics_from_scores(target, scores, threshold)
