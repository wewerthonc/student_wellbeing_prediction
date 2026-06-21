"""YAML-backed project configuration."""

from .loader import CONFIG_FILE, ProjectSettings, load_settings

SETTINGS = load_settings()

__all__ = ["CONFIG_FILE", "SETTINGS", "ProjectSettings", "load_settings"]
