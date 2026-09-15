import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


def build_preprocessor(
    df: pd.DataFrame,
    cleaning_plan: dict,
    outlier_analysis: dict | None = None,
) -> tuple[ColumnTransformer, dict]:
    """
    Build a reusable preprocessing pipeline.

    Numerical features use either StandardScaler or
    RobustScaler according to the outlier analysis.
    """

    target_column = cleaning_plan.get("target_column")

    if target_column is None:
        raise ValueError(
            "Cleaning plan does not contain a target column."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
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

    missing_features = [
        column
        for column in model_features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features not found in dataset: {missing_features}"
        )

    numerical_features = []
    categorical_features = []
    unsupported_features = []

    for column in model_features:
        series = df[column]

        if (
            pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical_features.append(column)

        elif pd.api.types.is_numeric_dtype(series.dtype):
            numerical_features.append(column)

        else:
            unsupported_features.append(column)

    recommended_robust_features = set()

    if outlier_analysis is not None:
        analysis_target = outlier_analysis.get(
            "target_column"
        )

        if analysis_target != target_column:
            raise ValueError(
                "Cleaning plan and outlier analysis "
                "use different targets."
            )

        recommended_robust_features = set(
            outlier_analysis.get(
                "robust_scaler_features",
                [],
            )
        )

    robust_numeric_features = [
        column
        for column in numerical_features
        if column in recommended_robust_features
    ]

    standard_numeric_features = [
        column
        for column in numerical_features
        if column not in recommended_robust_features
    ]

    transformers = []

    if standard_numeric_features:
        standard_numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "standard_numerical",
                standard_numerical_pipeline,
                standard_numeric_features,
            )
        )

    if robust_numeric_features:
        robust_numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    RobustScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "robust_numerical",
                robust_numerical_pipeline,
                robust_numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable features are available for preprocessing."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessing_metadata = {
        "target_column": target_column,
        "model_features": model_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "standard_scaler_features": (
            standard_numeric_features
        ),
        "robust_scaler_features": (
            robust_numeric_features
        ),
        "excluded_review_features": sorted(
            review_columns
        ),
        "unsupported_features": unsupported_features,
    }

    return preprocessor, preprocessing_metadata