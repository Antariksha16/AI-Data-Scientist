import pandas as pd

from ml_pipeline.target_suggester import (
    suggest_target_columns,
)


def create_cleaning_plan(
    df: pd.DataFrame,
    target_column: str,
    high_missing_threshold: float = 0.60,
) -> dict:
    """
    Create a proposed cleaning plan without modifying the dataset.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if not 0 <= high_missing_threshold <= 1:
        raise ValueError(
            "high_missing_threshold must be between 0 and 1."
        )

    candidate_features = [
        column
        for column in df.columns
        if column != target_column
    ]

    feature_df = df[candidate_features]

    feature_metadata = suggest_target_columns(
        feature_df,
        top_k=len(candidate_features),
    )

    metadata_by_column = {
        item["column"]: item
        for item in feature_metadata
    }

    drop_reasons = {}

    for column in candidate_features:
        series = df[column]

        missing_ratio = series.isnull().mean()
        unique_count = series.nunique(dropna=True)

        reasons = []

        if unique_count <= 1:
            reasons.append("constant or empty column")

        if metadata_by_column[column]["identifier_like"]:
            reasons.append("identifier-like column")

        if missing_ratio > high_missing_threshold:
            reasons.append(
                "missing values exceed configured threshold"
            )

        if reasons:
            drop_reasons[column] = reasons

    drop_columns = [
        column
        for column in candidate_features
        if column in drop_reasons
    ]

    kept_features = [
        column
        for column in candidate_features
        if column not in drop_reasons
    ]

    imputation_plan = {}
    review_columns = {}

    for column in kept_features:
        series = df[column]

        missing_count = int(series.isnull().sum())
        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else 0
        )

        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(series.dtype):
                imputation_plan[column] = "median"
            else:
                imputation_plan[column] = "most_frequent"

        is_categorical = (
            pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        )

        if (
            is_categorical
            and unique_count > 50
            and unique_ratio > 0.50
        ):
            review_columns[column] = (
                "high-cardinality categorical feature"
            )

    return {
        "target_column": target_column,
        "candidate_feature_count": len(candidate_features),
        "high_missing_threshold": high_missing_threshold,
        "drop_columns": drop_columns,
        "drop_reasons": drop_reasons,
        "kept_feature_count": len(kept_features),
        "kept_features": kept_features,
        "imputation_plan": imputation_plan,
        "review_columns": review_columns,
    }

def apply_column_removal(
    df: pd.DataFrame,
    cleaning_plan: dict,
) -> pd.DataFrame:
    """
    Apply only the column-removal decisions from a cleaning plan.

    A copy is returned so the original DataFrame is not modified.
    """

    target_column = cleaning_plan.get("target_column")
    drop_columns = cleaning_plan.get("drop_columns", [])

    if target_column is None:
        raise ValueError(
            "Cleaning plan does not contain a target column."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if target_column in drop_columns:
        raise ValueError(
            "The selected target cannot be removed."
        )

    missing_columns = [
        column
        for column in drop_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found in dataset: {missing_columns}"
        )

    cleaned_df = df.drop(
        columns=drop_columns,
    ).copy()

    return cleaned_df