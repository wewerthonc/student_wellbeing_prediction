"""Run the configured model benchmark."""

from time import perf_counter

import pandas as pd

from student_depression.data import (
    ensure_dataset,
    load_dataset,
    split_features_target,
    split_train_test,
)
from student_depression.evaluation import classification_metrics
from student_depression.models import build_benchmark_pipelines
from student_depression.utils import configure_logging


def main() -> None:
    """Train configured models and compare them on one holdout set."""
    logger = configure_logging()

    data_path = ensure_dataset()
    data = load_dataset(data_path)
    features, target = split_features_target(data)

    # A shared split makes scores directly comparable across every model.
    features_train, features_test, target_train, target_test = split_train_test(features, target)

    results = []
    for model_name, pipeline in build_benchmark_pipelines().items():
        logger.info("Training %s", model_name)
        started_at = perf_counter()
        pipeline.fit(features_train, target_train)
        elapsed_seconds = perf_counter() - started_at

        metrics = classification_metrics(pipeline, features_test, target_test)
        results.append(
            {
                "model": model_name,
                **metrics,
                "fit_seconds": elapsed_seconds,
            }
        )

    # ROC AUC is the primary ranking metric; the full table keeps tradeoffs visible.
    benchmark = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    logger.info("Benchmark results:\n%s", benchmark.round(3).to_string())


if __name__ == "__main__":
    main()
