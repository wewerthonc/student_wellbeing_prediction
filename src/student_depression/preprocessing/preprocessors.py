"""Feature engineering and column-wise sklearn preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_ENGINEERING_COLUMNS = {
    "Academic Pressure",
    "CGPA",
    "Financial Stress",
    "Job Satisfaction",
    "Study Satisfaction",
    "Work Pressure",
    "Work/Study Hours",
}


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw values and create domain-inspired numerical interactions."""
    missing_columns = REQUIRED_ENGINEERING_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        message = f"Required feature columns are missing: {missing}"
        raise ValueError(message)

    data = dataframe.copy()
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    sleep_midpoint = {
        "Less than 5 hours": 4.5,
        "5-6 hours": 5.5,
        "7-8 hours": 7.5,
        "More than 8 hours": 8.5,
        "Others": np.nan,
    }
    sleep_risk = {
        "Less than 5 hours": 2.0,
        "5-6 hours": 1.0,
        "7-8 hours": 0.0,
        "More than 8 hours": 1.0,
        "Others": np.nan,
    }
    diet_risk = {"Healthy": 0.0, "Moderate": 1.0, "Unhealthy": 2.0, "Others": np.nan}
    yes_no = {"No": 0.0, "Yes": 1.0}

    if "Sleep Duration" in data:
        data["Sleep Hours Midpoint"] = data["Sleep Duration"].map(sleep_midpoint)
        data["Sleep Risk Score"] = data["Sleep Duration"].map(sleep_risk)
    if "Dietary Habits" in data:
        data["Diet Risk Score"] = data["Dietary Habits"].map(diet_risk)
    if "Have you ever had suicidal thoughts ?" in data:
        data["Suicidal Thoughts Binary"] = data["Have you ever had suicidal thoughts ?"].map(yes_no)
    if "Family History of Mental Illness" in data:
        data["Family History Binary"] = data["Family History of Mental Illness"].map(yes_no)

    total_pressure = data["Academic Pressure"] + data["Work Pressure"]
    total_satisfaction = data["Study Satisfaction"] + data["Job Satisfaction"]

    data["Total Pressure"] = total_pressure
    data["Total Satisfaction"] = total_satisfaction
    data["Pressure Satisfaction Ratio"] = (total_pressure + 1.0) / (total_satisfaction + 1.0)
    data["Stress Academic Index"] = data["Academic Pressure"] + data["Financial Stress"]
    data["Load Pressure Interaction"] = data["Work/Study Hours"] * data["Academic Pressure"]
    data["CGPA Pressure Interaction"] = data["CGPA"] * data["Academic Pressure"]

    return data


def build_preprocessor() -> ColumnTransformer:
    """Build dtype-aware numerical and categorical preprocessing."""
    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Mutual information must inspect continuous and encoded values together.
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        verbose_feature_names_out=False,
    )
