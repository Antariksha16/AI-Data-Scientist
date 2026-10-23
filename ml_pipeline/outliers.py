import pandas as pd


def analyze_outliers(
    df: pd.DataFrame,
    cleaning_plan: dict,
    iqr_multiplier: float = 1.5,
    robust_scaler_threshold: float = 0.05,
    minimum_unique_values: int = 10,
) -> dict:
    """
    Analyze numerical feature outliers using the IQR method.

    This function does not remove rows. It recommends robust
    scaling when the outlier percentage is sufficiently high.
    """

    if iqr_multiplier <= 0:
        raise ValueError(
            "iqr_multiplier must be greater than zero."
        )

    if not 0 <= robust_scaler_threshold <= 1:
        raise ValueError(
            "robust_scaler_threshold must be between 0 and 1."
        )

    target_column = cleaning_plan.get("target_column")

    if target_column is None:
        raise ValueError(
            "Cleaning plan does not contain a target column."
        )

    kept_features = cleaning_plan.get(
        "kept_features",
        [],
    )

    review_columns = set(
        cleaning_plan.get(
            "review_columns",
            {},
        ).keys()
    )

    model_features = [
        column
        for column in kept_features
        if column not in review_columns
    ]

    numerical_features = [
        column
        for column in model_features
        if (
            column in df.columns
            and pd.api.types.is_numeric_dtype(
                df[column].dtype
            )
            and not pd.api.types.is_bool_dtype(
                df[column].dtype
            )
        )
    ]

    outlier_summary = {}
    skipped_discrete_features = []
    skipped_zero_iqr_features = []

    for column in numerical_features:
        values = df[column].dropna()

        unique_count = int(values.nunique())

        if unique_count < minimum_unique_values:
            skipped_discrete_features.append(column)
            continue

        first_quartile = float(values.quantile(0.25))
        third_quartile = float(values.quantile(0.75))
        iqr = third_quartile - first_quartile

        if iqr == 0:
            skipped_zero_iqr_features.append(column)
            continue

        lower_bound = (
            first_quartile - iqr_multiplier * iqr
        )
        upper_bound = (
            third_quartile + iqr_multiplier * iqr
        )

        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())
        outlier_percentage = (
            outlier_count / len(values)
            if len(values) > 0
            else 0
        )

        recommended_scaler = (
            "robust"
            if outlier_percentage
            >= robust_scaler_threshold
            else "standard"
        )

        outlier_summary[column] = {
            "first_quartile": round(
                first_quartile,
                4,
            ),
            "third_quartile": round(
                third_quartile,
                4,
            ),
            "iqr": round(iqr, 4),
            "lower_bound": round(
                lower_bound,
                4,
            ),
            "upper_bound": round(
                upper_bound,
                4,
            ),
            "outlier_count": outlier_count,
            "outlier_percentage": round(
                outlier_percentage * 100,
                2,
            ),
            "recommended_scaler": recommended_scaler,
        }

    robust_scaler_features = [
        column
        for column, result in outlier_summary.items()
        if result["recommended_scaler"] == "robust"
    ]

    standard_scaler_features = [
        column
        for column, result in outlier_summary.items()
        if result["recommended_scaler"] == "standard"
    ]

    return {
        "target_column": target_column,
        "numerical_features_analyzed": numerical_features,
        "outlier_summary": outlier_summary,
        "robust_scaler_features": robust_scaler_features,
        "standard_scaler_features": standard_scaler_features,
        "skipped_discrete_features": (
            skipped_discrete_features
        ),
        "skipped_zero_iqr_features": (
            skipped_zero_iqr_features
        ),
        "rows_removed": 0,
        "reason": (
            "Outliers are retained; scaler recommendations "
            "are used instead of automatic row deletion."
        ),
    }