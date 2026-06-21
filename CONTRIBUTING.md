# Contributing

Keep contributions focused, reproducible, and cautious about conclusions drawn from sensitive
mental-health data.

## Local workflow

1. Create a focused branch from the current default branch.
2. Run `uv sync` to install the locked project and development dependencies.
3. Make reusable changes inside the appropriate `src/student_depression/` module.
4. Run the quality checks before committing:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run the relevant training command or notebook when changing data, preprocessing, models, or metrics.
Do not commit downloaded datasets, model artifacts, generated figures, virtual environments, IDE
metadata, or notebook checkpoints.

## Design guidelines

- Keep runtime settings in `config/config.yaml`, loading logic in `config/loader.py`, and filesystem
  knowledge in `paths.py`.
- Keep downloading and splitting separate from preprocessing.
- Register estimators in `model_registry.py` and configure them in YAML so benchmarks share the same
  pipeline and holdout split.
- Fit preprocessing and feature selection only on training data or training folds.
- Add type hints and concise docstrings to public functions.
- Fail clearly when required data or configuration is invalid; avoid silent fallbacks.

## Commits and pull requests

- Use short, imperative commit messages.
- Explain what changed, why it changed, and how it was verified.
- Keep unrelated refactors in separate changes.
- Document assumptions, target-leakage risks, fairness limitations, and changes to model behavior.
- Never present project output as medical advice or a diagnostic result.
