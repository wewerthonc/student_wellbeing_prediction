"""Load typed project settings from YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Dataset source and schema settings."""

    url: str
    target_column: str
    identifier_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Train/test split settings."""

    test_size: float
    validation_size: float
    random_state: int
    cv_folds: int


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    """Column transformation settings."""

    categorical_min_frequency: int | float | None


@dataclass(frozen=True, slots=True)
class FeatureSelectionSettings:
    """Mutual-information feature-selection settings."""

    default_k: int | str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Configuration for one benchmark model."""

    name: str
    enabled: bool
    feature_selection_k: int | str | None
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Complete project settings loaded from YAML."""

    dataset: DatasetSettings
    split: SplitSettings
    preprocessing: PreprocessingSettings
    feature_selection: FeatureSelectionSettings
    models: tuple[ModelSettings, ...]


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        message = f"Configuration must be a YAML mapping: {config_path}"
        raise ValueError(message)
    return payload


def _parse_feature_selection_k(value: Any) -> int | str:
    if value == "all":
        return "all"
    if isinstance(value, bool):
        raise ValueError
    k = int(value)
    if k < 1:
        raise ValueError
    return k


def _parse_optional_feature_selection_k(value: Any) -> int | str | None:
    if value is None:
        return None
    return _parse_feature_selection_k(value)


def _parse_categorical_min_frequency(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError
    min_frequency = float(value)
    if min_frequency <= 0:
        raise ValueError
    if min_frequency.is_integer():
        return int(min_frequency)
    if min_frequency >= 1:
        raise ValueError
    return min_frequency


def load_settings(config_path: Path = CONFIG_FILE) -> ProjectSettings:
    """Load and validate project settings from a YAML file."""
    payload = _read_yaml(config_path)
    try:
        dataset = payload["dataset"]
        split = payload["split"]
        preprocessing = payload["preprocessing"]
        feature_selection = payload["feature_selection"]
        models = payload["models"]
        if not isinstance(preprocessing, dict) or not isinstance(feature_selection, dict):
            raise TypeError
        if not isinstance(models, dict):
            raise TypeError

        default_k = feature_selection.get("default_k", feature_selection.get("k"))

        model_settings = []
        for name, configuration in models.items():
            if not isinstance(configuration, dict):
                raise TypeError
            parameters = configuration.get("parameters", {})
            if not isinstance(parameters, dict):
                raise TypeError
            model_settings.append(
                ModelSettings(
                    name=str(name),
                    enabled=bool(configuration.get("enabled", True)),
                    feature_selection_k=_parse_optional_feature_selection_k(
                        configuration.get("feature_selection_k")
                    ),
                    parameters=dict(parameters),
                )
            )

        settings = ProjectSettings(
            dataset=DatasetSettings(
                url=str(dataset["url"]),
                target_column=str(dataset["target_column"]),
                identifier_columns=tuple(dataset["identifier_columns"]),
            ),
            split=SplitSettings(
                test_size=float(split["test_size"]),
                validation_size=float(split["validation_size"]),
                random_state=int(split["random_state"]),
                cv_folds=int(split["cv_folds"]),
            ),
            preprocessing=PreprocessingSettings(
                categorical_min_frequency=_parse_categorical_min_frequency(
                    preprocessing.get("categorical_min_frequency")
                ),
            ),
            feature_selection=FeatureSelectionSettings(
                default_k=_parse_feature_selection_k(default_k)
            ),
            models=tuple(model_settings),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f"Invalid project configuration: {config_path}"
        raise ValueError(message) from error

    if not 0 < settings.split.test_size < 1:
        message = "split.test_size must be between 0 and 1"
        raise ValueError(message)
    if not 0 < settings.split.validation_size < 1:
        message = "split.validation_size must be between 0 and 1"
        raise ValueError(message)
    if settings.split.test_size + settings.split.validation_size >= 1:
        message = "split.test_size and split.validation_size must leave development data"
        raise ValueError(message)
    if settings.split.cv_folds < 2:
        message = "split.cv_folds must be at least 2"
        raise ValueError(message)
    if not any(model.enabled for model in settings.models):
        message = "At least one benchmark model must be enabled"
        raise ValueError(message)
    return settings
