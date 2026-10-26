from typing import Any

import pandas as pd

from ml_pipeline.analyzer import analyze_target
from ml_pipeline.cleaner import create_cleaning_plan
from ml_pipeline.leakage import (
    detect_potential_leakage,
)
from ml_pipeline.outliers import analyze_outliers
from ml_pipeline.preprocessor import (
    build_preprocessor,
)


def prepare_dataset_for_modeling(
    df: pd.DataFrame,
    target_column: str,
    allow_potential_leakage: bool = False,
) -> dict[str, Any]:
    """
    Prepare a supervised-learning dataset for model training.

    Returns X, y, a reusable preprocessor, and preprocessing
    decision reports.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    target_analysis = analyze_target(
        df,
        target_column,
    )

    target_missing_rows = int(
        df[target_column].isnull().sum()
    )

    training_df = df.loc[
        df[target_column].notna()
    ].copy()

    if training_df.empty:
        raise ValueError(
            "The selected target contains no usable values."
        )

    target_unique_values = int(
        training_df[target_column].nunique(
            dropna=True
        )
    )

    if target_unique_values < 2:
        raise ValueError(
            "The selected target must contain at least "
            "two unique values."
        )

    cleaning_plan = create_cleaning_plan(
        training_df,
        target_column,
    )

    # Record leakage evidence from all raw candidate features.
    raw_leakage_report = detect_potential_leakage(
        training_df,
        target_column,
    )

    # Review columns are not currently allowed into training.
    review_columns = set(
        cleaning_plan.get(
            "review_columns",
            {},
        ).keys()
    )

    # Only retained, non-review features enter the model.
    model_feature_columns = [
        column
        for column in cleaning_plan["kept_features"]
        if column not in review_columns
    ]

    # Leakage in excluded columns is recorded, but only leakage
    # in actual model features should block training.
    leakage_input_df = training_df[
        model_feature_columns + [target_column]
    ]

    leakage_report = detect_potential_leakage(
        leakage_input_df,
        target_column,
    )

    if (
        leakage_report["potential_leakage_count"] > 0
        and not allow_potential_leakage
    ):
        leakage_columns = leakage_report[
            "potential_leakage_columns"
        ]

        raise ValueError(
            "Potential target leakage requires review: "
            f"{leakage_columns}"
        )

    outlier_report = analyze_outliers(
        training_df,
        cleaning_plan,
    )

    preprocessor, preprocessing_metadata = (
        build_preprocessor(
            training_df,
            cleaning_plan,
            outlier_report,
        )
    )

    X = training_df.drop(
        columns=[target_column]
    ).copy()

    y = training_df[target_column].copy()

    report = {
        "target_analysis": target_analysis,
        "original_rows": len(df),
        "training_rows": len(training_df),
        "target_missing_rows_removed": (
            target_missing_rows
        ),
        "original_columns": len(df.columns),
        "candidate_feature_count": len(X.columns),
        "cleaning_plan": cleaning_plan,
        "raw_leakage_report": raw_leakage_report,
        "leakage_report": leakage_report,
        "outlier_report": outlier_report,
        "preprocessing_metadata": (
            preprocessing_metadata
        ),
    }

    return {
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "report": report,
    }