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

The first benchmark run downloads the public Kaggle dataset into the ignored `data/raw/` directory
and runs the selected experiment configuration:

```bash
uv run student-depression
```

Models are enabled and parameterized in `config/config.yaml`. The default configuration follows the
experiment protocol:

- stratified development/validation/test split of 64%/16%/20% with `random_state=42`
- median/mode imputation, `RobustScaler`, one-hot encoding with `min_frequency=10`
- deterministic feature engineering and mutual-information feature selection inside each pipeline
- threshold search from 0.10 to 0.90 on validation, optimizing F2 with F1 and balanced accuracy as
  tie-breakers
- final test reporting for the prevalence baseline, logistic regression, SGD logistic, decision tree,
  Random Forest, Extra Trees, Balanced Random Forest, Hist Gradient Boosting, XGBoost, and PyTorch MLP


```yaml
models:
  mlp_pytorch:
    enabled: true
    feature_selection_k: 40
    parameters:
      hidden_layers: [128, 64]
      dropout: [0.25, 0.20]
      device: cuda
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

The benchmark configures the selected experiment model set. `model_registry.py` maps stable YAML
names to sklearn, imbalanced-learn, XGBoost, and PyTorch factories. Every model uses an independent
copy of the same feature engineering, imputation, encoding, scaling, and mutual-information
feature-selection pipeline. The number of selected features can be set per model with
`feature_selection_k`, falling back to `feature_selection.default_k`. All models are evaluated on the
same validation and test splits, making their thresholded metrics directly comparable.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes.
