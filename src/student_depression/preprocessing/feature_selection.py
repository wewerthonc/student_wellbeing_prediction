"""Feature-selection components for model pipelines."""

from functools import partial

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from student_depression.config import SETTINGS


def build_feature_selector(
    k: int = SETTINGS.feature_selection.k,
    random_state: int = SETTINGS.split.random_state,
) -> SelectKBest:
    """Select the k transformed features with the most target information."""
    if k < 1:
        message = f"The number of selected features must be positive: {k}"
        raise ValueError(message)

    # Fixing the seed makes mutual-information estimates reproducible across models.
    score_function = partial(mutual_info_classif, random_state=random_state)
    return SelectKBest(score_func=score_function, k=k)
