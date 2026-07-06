"""Leakage-safe dataset splitting."""

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from student_depression.config import SETTINGS


def split_features_target(
    data: pd.DataFrame,
    target_column: str = SETTINGS.dataset.target_column,
    identifier_columns: Sequence[str] = SETTINGS.dataset.identifier_columns,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and target while dropping row identifiers."""
    if target_column not in data.columns:
        message = f"Target column not found: {target_column}"
        raise ValueError(message)

    target = data[target_column].copy()
    if target.isna().any():
        message = f"Target contains {int(target.isna().sum())} missing values"
        raise ValueError(message)

    columns_to_drop = [target_column, *[name for name in identifier_columns if name in data]]
    features = data.drop(columns=columns_to_drop).copy()
    return features, target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = SETTINGS.split.test_size,
    random_state: int = SETTINGS.split.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""
    if not 0 < test_size < 1:
        message = f"test_size must be between 0 and 1, received {test_size}"
        raise ValueError(message)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def split_development_validation_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = SETTINGS.split.test_size,
    validation_size: float = SETTINGS.split.validation_size,
    random_state: int = SETTINGS.split.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create the stratified 64/16/20 development/validation/test split."""
    if not 0 < test_size < 1:
        message = f"test_size must be between 0 and 1, received {test_size}"
        raise ValueError(message)
    if not 0 < validation_size < 1:
        message = f"validation_size must be between 0 and 1, received {validation_size}"
        raise ValueError(message)
    if test_size + validation_size >= 1:
        message = "test_size and validation_size must leave development data"
        raise ValueError(message)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    validation_fraction_of_train = validation_size / (1.0 - test_size)
    features_development, features_validation, target_development, target_validation = (
        train_test_split(
            features_train,
            target_train,
            test_size=validation_fraction_of_train,
            random_state=random_state,
            stratify=target_train,
        )
    )
    return (
        features_development,
        features_validation,
        features_test,
        target_development,
        target_validation,
        target_test,
    )
