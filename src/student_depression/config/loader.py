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
    random_state: int


@dataclass(frozen=True, slots=True)
class FeatureSelectionSettings:
    """Mutual-information feature-selection settings."""

    k: int


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Configuration for one benchmark model."""

    name: str
    enabled: bool
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    """Complete project settings loaded from YAML."""

    dataset: DatasetSettings
    split: SplitSettings
    feature_selection: FeatureSelectionSettings
    models: tuple[ModelSettings, ...]


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        message = f"Configuration must be a YAML mapping: {config_path}"
        raise ValueError(message)
    return payload


def load_settings(config_path: Path = CONFIG_FILE) -> ProjectSettings:
    """Load and validate project settings from a YAML file."""
    payload = _read_yaml(config_path)
    try:
        dataset = payload["dataset"]
        split = payload["split"]
        feature_selection = payload["feature_selection"]
        models = payload["models"]
        if not isinstance(models, dict):
            raise TypeError

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
                random_state=int(split["random_state"]),
            ),
            feature_selection=FeatureSelectionSettings(k=int(feature_selection["k"])),
            models=tuple(model_settings),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f"Invalid project configuration: {config_path}"
        raise ValueError(message) from error

    if not 0 < settings.split.test_size < 1:
        message = "split.test_size must be between 0 and 1"
        raise ValueError(message)
    if settings.feature_selection.k < 1:
        message = "feature_selection.k must be at least 1"
        raise ValueError(message)
    if not any(model.enabled for model in settings.models):
        message = "At least one benchmark model must be enabled"
        raise ValueError(message)
    return settings
