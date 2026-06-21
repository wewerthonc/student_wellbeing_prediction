# Student Depression

A modular data-science project for exploring the Student Depression Dataset and training reproducible
scikit-learn classifiers. The code separates data access, preprocessing, model construction, and
evaluation so each concern can evolve independently.

> [!IMPORTANT]
> This project is for research and learning. Predictions are not clinical diagnoses and must not be
> used as a substitute for qualified mental-health care.

## Setup with uv

Install [uv](https://docs.astral.sh/uv/) and synchronize the locked environment:

```bash
uv sync
```

The first benchmark run downloads the public Kaggle dataset into the ignored `data/raw/` directory:

```bash
uv run student-depression
```

Models are enabled and parameterized in `config/config.yaml`. Every enabled model runs in one
benchmark:

```yaml
models:
  logistic_regression:
    enabled: true
    parameters:
      class_weight: balanced
      max_iter: 1000
```

Start the exploratory notebook environment with:

```bash
uv run jupyter lab
```

## Architecture

```text
src/
└── student_depression/
    ├── __init__.py
    ├── __main__.py                  # Command-line training workflow
    ├── config/
    │   ├── config.yaml              # Reproducible project settings
    │   └── loader.py                # Typed YAML configuration loader
    ├── paths.py                     # Central filesystem locations
    ├── data/
    │   ├── load_data.py             # Downloading and CSV loading
    │   └── split_data.py            # Target and stratified splits
    ├── preprocessing/
    │   ├── preprocessors.py         # Feature engineering and transformers
    │   └── feature_selection.py     # Variance-based selection
    ├── models/
    │   └── model_registry.py        # Model factories and complete pipelines
    ├── evaluation/
    │   ├── metrics.py               # Holdout metrics and reports
    │   └── cross_validation.py      # Stratified cross-validation
    ├── utils/
    │   └── logging.py               # Shared logging configuration
    └── visualization/
        └── plots.py                 # Exploratory plots
```

The benchmark currently configures logistic regression, random forest, and decision tree.
`model_registry.py` maps their stable YAML names to sklearn factories, leaving one clear place to add
future implementations. Every model uses an independent copy of the same feature engineering,
imputation, encoding, scaling, and mutual-information feature-selection pipeline. The number of
selected features is controlled by `feature_selection.k` in YAML. All models are evaluated on one
shared holdout set, making their metrics directly comparable.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.
