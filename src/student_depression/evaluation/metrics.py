"""Comparable metrics for benchmark classifiers."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Evaluate one fitted binary classifier on the shared holdout set."""
    predictions = estimator.predict(features)
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": float("nan"),
    }

    # Probability-based ROC AUC gives a threshold-independent comparison.
    if hasattr(estimator, "predict_proba") and 1 in estimator.classes_:
        positive_class_index = list(estimator.classes_).index(1)
        probabilities = estimator.predict_proba(features)[:, positive_class_index]
        metrics["roc_auc"] = roc_auc_score(target, probabilities)
    return metrics
