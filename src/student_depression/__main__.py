"""Run the configured model benchmark."""

from time import perf_counter

import pandas as pd
from sklearn.pipeline import Pipeline

from student_depression.data import (
    ensure_dataset,
    load_dataset,
    split_development_validation_test,
    split_features_target,
)
from student_depression.evaluation import classification_metrics, select_threshold
from student_depression.models import build_benchmark_pipelines
from student_depression.utils import configure_logging


def _fit_pipeline_with_optional_validation(
    pipeline: Pipeline,
    features_development: pd.DataFrame,
    target_development: pd.Series,
    features_validation: pd.DataFrame,
    target_validation: pd.Series,
) -> None:
    """Fit sklearn pipelines, giving PyTorch models transformed validation data."""
    final_estimator = pipeline.steps[-1][1]
    if not getattr(final_estimator, "uses_validation_data", False):
        pipeline.fit(features_development, target_development)
        return

    feature_pipeline = pipeline[:-1]
    transformed_development = feature_pipeline.fit_transform(
        features_development,
        target_development,
    )
    transformed_validation = feature_pipeline.transform(features_validation)
    final_estimator.fit(
        transformed_development,
        target_development,
        validation_data=(transformed_validation, target_validation),
    )


def main() -> None:
    """Train configured models and compare them on one holdout set."""
    logger = configure_logging()

    data_path = ensure_dataset()
    data = load_dataset(data_path)
    features, target = split_features_target(data)

    (
        features_development,
        features_validation,
        features_test,
        target_development,
        target_validation,
        target_test,
    ) = split_development_validation_test(features, target)
    logger.info(
        "Split sizes: development=%s validation=%s test=%s",
        len(features_development),
        len(features_validation),
        len(features_test),
    )

    results = []
    for model_name, pipeline in build_benchmark_pipelines().items():
        logger.info("Training %s", model_name)
        started_at = perf_counter()
        _fit_pipeline_with_optional_validation(
            pipeline,
            features_development,
            target_development,
            features_validation,
            target_validation,
        )
        elapsed_seconds = perf_counter() - started_at

        threshold, validation_metrics = select_threshold(
            pipeline,
            features_validation,
            target_validation,
        )
        metrics = classification_metrics(pipeline, features_test, target_test, threshold=threshold)
        results.append(
            {
                "model": model_name,
                "validation_f2": validation_metrics["f2"],
                **metrics,
                "fit_seconds": elapsed_seconds,
            }
        )

    # F2 is the threshold-selection objective; the full table keeps tradeoffs visible.
    benchmark = pd.DataFrame(results).set_index("model").sort_values("f2", ascending=False)
    logger.info("Benchmark results:\n%s", benchmark.round(3).to_string())


if __name__ == "__main__":
    main()
