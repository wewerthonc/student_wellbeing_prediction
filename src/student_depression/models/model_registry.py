"""Registry and pipeline construction for benchmark classifiers."""

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from student_depression.config import SETTINGS
from student_depression.config.loader import ModelSettings
from student_depression.preprocessing import (
    build_feature_selector,
    build_preprocessor,
    engineer_features,
)

ModelFactory = Callable[..., ClassifierMixin]

# YAML refers to these stable names instead of importing arbitrary Python objects.
MODEL_REGISTRY: dict[str, ModelFactory] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
}


def create_model(settings: ModelSettings) -> ClassifierMixin:
    """Instantiate one configured model with the shared random seed."""
    try:
        factory = MODEL_REGISTRY[settings.name]
    except KeyError as error:
        message = f"Configured model is not registered: {settings.name}"
        raise ValueError(message) from error

    parameters = dict(settings.parameters)
    parameters.setdefault("random_state", SETTINGS.split.random_state)
    try:
        return factory(**parameters)
    except TypeError as error:
        message = f"Invalid parameters for model '{settings.name}': {parameters}"
        raise ValueError(message) from error


def build_model_pipeline(settings: ModelSettings) -> Pipeline:
    """Wrap one classifier in the shared feature and preprocessing pipeline."""
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(engineer_features, validate=False),
            ),
            ("preprocessing", build_preprocessor()),
            ("feature_selection", build_feature_selector()),
            ("model", create_model(settings)),
        ]
    )


def build_benchmark_pipelines() -> dict[str, Pipeline]:
    """Build independent pipelines for every enabled YAML model."""
    # Each model receives a fresh preprocessor so fitting one cannot mutate another.
    return {
        settings.name: build_model_pipeline(settings)
        for settings in SETTINGS.models
        if settings.enabled
    }
