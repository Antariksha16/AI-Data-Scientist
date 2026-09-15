import math
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.analyzer import analyze_target


SUPPORTED_PROBLEM_TYPES = {
    "classification",
    "regression",
}


def _safe_number(
    value: Any,
    decimal_places: int,
) -> float | None:
    """Return a finite, rounded, JSON-safe number."""

    if value is None or pd.isna(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(number, decimal_places)


def _safe_value(value: Any) -> Any:
    """Convert a scalar target value into a JSON-safe value."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _resolve_problem_type(
    detected_problem_type: str | None,
    problem_type: str | None,
) -> tuple[str, str]:
    """Resolve the target problem type and record its source."""

    if problem_type is not None:
        normalized_problem_type = problem_type.strip().lower()

        if normalized_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise ValueError(
                "problem_type must be either 'classification' "
                "or 'regression'."
            )

        return normalized_problem_type, "user_override"

    normalized_detected_type = str(
        detected_problem_type or ""
    ).strip().lower()

    if normalized_detected_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "The target problem type could not be resolved as "
            "classification or regression. Provide an explicit "
            "problem_type override."
        )

    return normalized_detected_type, "target_analyzer"


def _classification_target_statistics(
    observed_target: pd.Series,
    config: EDAConfig,
) -> tuple[dict, list[str], list[str]]:
    """Generate classification distribution and imbalance details."""

    value_counts = observed_target.value_counts(
        dropna=False
    )
    observed_count = len(observed_target)
    class_count = len(value_counts)

    if class_count == 0:
        target_structure = "empty"
    elif class_count == 1:
        target_structure = "single_class"
    elif class_count == 2:
        target_structure = "binary"
    else:
        target_structure = "multiclass"

    displayed_counts = value_counts.head(
        config.max_categories_displayed
    )

    class_distribution = [
        {
            "class_value": _safe_value(value),
            "count": int(count),
            "percentage": round(
                int(count) / observed_count * 100,
                2,
            )
            if observed_count > 0
            else 0.0,
        }
        for value, count in displayed_counts.items()
    ]

    majority_class = (
        value_counts.index[0]
        if class_count > 0
        else None
    )
    majority_count = (
        int(value_counts.iloc[0])
        if class_count > 0
        else 0
    )
    minority_class = (
        value_counts.index[-1]
        if class_count > 0
        else None
    )
    minority_count = (
        int(value_counts.iloc[-1])
        if class_count > 0
        else 0
    )

    minority_to_majority_ratio = (
        minority_count / majority_count
        if majority_count > 0
        else 0.0
    )

    if class_count < 2:
        imbalance_level = "not_applicable"
    elif (
        minority_to_majority_ratio
        <= config.severe_imbalance_ratio
    ):
        imbalance_level = "severe"
    elif (
        minority_to_majority_ratio
        <= config.moderate_imbalance_ratio
    ):
        imbalance_level = "moderate"
    elif (
        minority_to_majority_ratio
        <= config.mild_imbalance_ratio
    ):
        imbalance_level = "mild"
    else:
        imbalance_level = "balanced"

    probabilities = (
        value_counts.to_numpy(dtype=float)
        / observed_count
        if observed_count > 0
        else np.array([], dtype=float)
    )

    entropy = (
        float(
            -np.sum(
                probabilities
                * np.log(probabilities)
            )
        )
        if probabilities.size > 0
        else 0.0
    )

    normalized_entropy = (
        entropy / math.log(class_count)
        if class_count > 1
        else 0.0
    )

    effective_class_count = (
        math.exp(entropy)
        if class_count > 0
        else 0.0
    )

    class_ratios = (
        value_counts / observed_count
        if observed_count > 0
        else value_counts.astype(float)
    )

    rare_class_mask = (
        class_ratios < config.rare_category_threshold
    )

    rare_classes = [
        {
            "class_value": _safe_value(value),
            "count": int(value_counts.loc[value]),
            "percentage": round(
                float(class_ratios.loc[value]) * 100,
                2,
            ),
        }
        for value in value_counts.index[rare_class_mask][
            : config.max_categories_displayed
        ]
    ]

    warnings = []
    recommendations = []

    if class_count < 2:
        warnings.append(
            "The target has fewer than two observed classes and "
            "cannot train a classifier."
        )
    else:
        recommendations.append(
            "Use stratified train/test splitting and stratified "
            "cross-validation when class counts permit."
        )

    if imbalance_level in {"mild", "moderate", "severe"}:
        warnings.append(
            f"The target has {imbalance_level} class imbalance."
        )
        recommendations.append(
            "Compare class-weighted models and evaluate per-class "
            "performance instead of relying on accuracy alone."
        )

    if imbalance_level in {"moderate", "severe"}:
        recommendations.append(
            "Review resampling only inside training folds to avoid "
            "validation leakage."
        )

    if class_count > config.max_categories_displayed:
        warnings.append(
            "The target has high class cardinality; the displayed "
            "distribution is truncated."
        )

    if minority_count == 1 and class_count > 1:
        warnings.append(
            "At least one class has only one row, so standard "
            "stratified splitting is not possible."
        )

    if target_structure == "binary":
        recommended_metrics = [
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ]
    else:
        recommended_metrics = [
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class_precision_recall",
            "log_loss",
        ]

    statistics = {
        "target_structure": target_structure,
        "class_count": class_count,
        "class_distribution": class_distribution,
        "classes_truncated": (
            class_count
            > config.max_categories_displayed
        ),
        "undisplayed_class_count": max(
            class_count
            - config.max_categories_displayed,
            0,
        ),
        "majority_class": _safe_value(majority_class),
        "majority_class_count": majority_count,
        "majority_class_percentage": round(
            majority_count / observed_count * 100,
            2,
        )
        if observed_count > 0
        else 0.0,
        "minority_class": _safe_value(minority_class),
        "minority_class_count": minority_count,
        "minority_class_percentage": round(
            minority_count / observed_count * 100,
            2,
        )
        if observed_count > 0
        else 0.0,
        "minority_to_majority_ratio": round(
            minority_to_majority_ratio,
            config.decimal_places,
        ),
        "imbalance_level": imbalance_level,
        "is_imbalanced": imbalance_level
        in {"mild", "moderate", "severe"},
        "entropy": _safe_number(
            entropy,
            config.decimal_places,
        ),
        "normalized_entropy": _safe_number(
            normalized_entropy,
            config.decimal_places,
        ),
        "effective_class_count": _safe_number(
            effective_class_count,
            config.decimal_places,
        ),
        "rare_class_count": int(rare_class_mask.sum()),
        "rare_classes": rare_classes,
        "rare_classes_truncated": int(
            rare_class_mask.sum()
        )
        > config.max_categories_displayed,
        "minimum_class_count": minority_count,
        "maximum_stratified_cv_folds": min(
            minority_count,
            10,
        )
        if class_count > 1 and minority_count >= 2
        else 0,
        "recommended_metrics": recommended_metrics,
        "positive_class_requires_user_confirmation": (
            target_structure == "binary"
        ),
        "automatic_resampling_performed": False,
    }

    return statistics, warnings, recommendations


def _regression_target_statistics(
    observed_target: pd.Series,
    config: EDAConfig,
) -> tuple[dict, list[str], list[str]]:
    """Generate numerical distribution details for a regression target."""

    numeric_target = pd.to_numeric(
        observed_target,
        errors="coerce",
    )

    non_numeric_count = int(
        numeric_target.isna().sum()
    )
    infinite_count = int(
        numeric_target.isin([np.inf, -np.inf]).sum()
    )

    finite_values = numeric_target.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    finite_count = len(finite_values)
    unique_count = int(finite_values.nunique())

    mean = finite_values.mean() if finite_count > 0 else None
    median = (
        finite_values.median()
        if finite_count > 0
        else None
    )
    standard_deviation = (
        finite_values.std()
        if finite_count > 1
        else None
    )
    variance = (
        finite_values.var()
        if finite_count > 1
        else None
    )
    minimum = finite_values.min() if finite_count > 0 else None
    maximum = finite_values.max() if finite_count > 0 else None

    first_quartile = (
        finite_values.quantile(0.25)
        if finite_count > 0
        else None
    )
    third_quartile = (
        finite_values.quantile(0.75)
        if finite_count > 0
        else None
    )
    iqr = (
        third_quartile - first_quartile
        if finite_count > 0
        else None
    )

    skewness = (
        finite_values.skew()
        if finite_count >= 3
        else None
    )
    kurtosis = (
        finite_values.kurtosis()
        if finite_count >= 4
        else None
    )

    lower_outlier_bound = None
    upper_outlier_bound = None
    outlier_count = 0
    outlier_ratio = 0.0

    if iqr is not None and iqr > 0:
        lower_outlier_bound = (
            first_quartile
            - config.outlier_iqr_multiplier * iqr
        )
        upper_outlier_bound = (
            third_quartile
            + config.outlier_iqr_multiplier * iqr
        )
        outlier_count = int(
            (
                (finite_values < lower_outlier_bound)
                | (finite_values > upper_outlier_bound)
            ).sum()
        )
        outlier_ratio = (
            outlier_count / finite_count
            if finite_count > 0
            else 0.0
        )

    safe_skewness = _safe_number(
        skewness,
        config.decimal_places,
    )

    if safe_skewness is None:
        distribution_shape = "undetermined"
    elif safe_skewness >= config.strong_skewness_threshold:
        distribution_shape = "right_skewed"
    elif safe_skewness <= -config.strong_skewness_threshold:
        distribution_shape = "left_skewed"
    else:
        distribution_shape = "approximately_symmetric"

    transformation_recommendation = None

    if distribution_shape == "right_skewed":
        transformation_recommendation = (
            "log1p_review"
            if minimum is not None and minimum >= 0
            else "power_transform_review"
        )
    elif distribution_shape == "left_skewed":
        transformation_recommendation = (
            "power_transform_review"
        )

    warnings = []
    recommendations = []

    if finite_count == 0:
        warnings.append(
            "The regression target contains no finite numeric "
            "values."
        )
    elif unique_count < 2:
        warnings.append(
            "The regression target has fewer than two unique "
            "finite values and cannot train a regressor."
        )

    if non_numeric_count > 0:
        warnings.append(
            "Some observed target values could not be converted "
            "to numbers."
        )

    if infinite_count > 0:
        warnings.append(
            "The regression target contains infinite values that "
            "must be removed or corrected before training."
        )

    if (
        outlier_ratio
        >= config.high_outlier_ratio_threshold
    ):
        warnings.append(
            "The regression target has a high IQR-outlier ratio."
        )
        recommendations.append(
            "Compare robust losses or tree-based models and review "
            "target outliers using domain knowledge."
        )

    if transformation_recommendation is not None:
        recommendations.append(
            "Evaluate the suggested target transformation with "
            "cross-validation and report metrics on the original "
            "target scale."
        )

    recommendations.append(
        "Report MAE and RMSE together; add R-squared when its "
        "interpretation is appropriate."
    )

    statistics = {
        "finite_count": finite_count,
        "non_numeric_values": non_numeric_count,
        "infinite_values": infinite_count,
        "unique_finite_values": unique_count,
        "mean": _safe_number(mean, config.decimal_places),
        "median": _safe_number(median, config.decimal_places),
        "standard_deviation": _safe_number(
            standard_deviation,
            config.decimal_places,
        ),
        "variance": _safe_number(
            variance,
            config.decimal_places,
        ),
        "minimum": _safe_number(
            minimum,
            config.decimal_places,
        ),
        "first_percentile": _safe_number(
            finite_values.quantile(0.01)
            if finite_count > 0
            else None,
            config.decimal_places,
        ),
        "fifth_percentile": _safe_number(
            finite_values.quantile(0.05)
            if finite_count > 0
            else None,
            config.decimal_places,
        ),
        "first_quartile": _safe_number(
            first_quartile,
            config.decimal_places,
        ),
        "third_quartile": _safe_number(
            third_quartile,
            config.decimal_places,
        ),
        "ninety_fifth_percentile": _safe_number(
            finite_values.quantile(0.95)
            if finite_count > 0
            else None,
            config.decimal_places,
        ),
        "ninety_ninth_percentile": _safe_number(
            finite_values.quantile(0.99)
            if finite_count > 0
            else None,
            config.decimal_places,
        ),
        "maximum": _safe_number(
            maximum,
            config.decimal_places,
        ),
        "range": _safe_number(
            maximum - minimum
            if finite_count > 0
            else None,
            config.decimal_places,
        ),
        "iqr": _safe_number(iqr, config.decimal_places),
        "skewness": safe_skewness,
        "kurtosis": _safe_number(
            kurtosis,
            config.decimal_places,
        ),
        "zero_count": int((finite_values == 0).sum()),
        "negative_count": int((finite_values < 0).sum()),
        "lower_outlier_bound": _safe_number(
            lower_outlier_bound,
            config.decimal_places,
        ),
        "upper_outlier_bound": _safe_number(
            upper_outlier_bound,
            config.decimal_places,
        ),
        "outlier_count": outlier_count,
        "outlier_percentage": round(
            outlier_ratio * 100,
            2,
        ),
        "distribution_shape": distribution_shape,
        "target_transformation_recommendation": (
            transformation_recommendation
        ),
        "target_scaling_guidance": (
            "review_for_linear_distance_based_or_neural_models"
        ),
        "recommended_metrics": [
            "mae",
            "rmse",
            "r2",
        ],
        "automatic_target_transformation_performed": False,
        "automatic_outlier_removal_performed": False,
    }

    return statistics, warnings, recommendations


def generate_target_statistics(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
    problem_type: str | None = None,
) -> dict:
    """
    Generate target-specific EDA for classification or regression.

    Missing target rows are reported but excluded from distribution
    calculations. No class balancing, target transformation, or row
    deletion is performed by EDA.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()
    target_series = df[target_column]
    observed_target = target_series.dropna()

    total_rows = len(df)
    observed_count = len(observed_target)
    missing_count = int(target_series.isna().sum())
    unique_count = int(observed_target.nunique())

    if observed_count > 0:
        target_analysis = analyze_target(
            df,
            target_column,
        )
        detected_problem_type = target_analysis.get(
            "problem_type"
        )
    else:
        target_analysis = {
            "target_column": target_column,
            "data_type": str(target_series.dtype),
            "unique_values": 0,
            "missing_values": missing_count,
            "missing_percentage": 100.0,
            "problem_type": "undetermined",
        }
        detected_problem_type = None

    if observed_count == 0 and problem_type is None:
        resolved_problem_type = "undetermined"
        problem_type_source = "insufficient_target_data"
    else:
        (
            resolved_problem_type,
            problem_type_source,
        ) = _resolve_problem_type(
            detected_problem_type,
            problem_type,
        )

    warnings = []
    recommendations = []
    classification_statistics = None
    regression_statistics = None

    if missing_count > 0:
        warnings.append(
            "Rows with missing target values cannot be used for "
            "supervised model training."
        )
        recommendations.append(
            "Exclude missing-target rows from training while "
            "preserving their count in the audit report."
        )

    if observed_count == 0:
        warnings.append(
            "The selected target contains no observed values."
        )
    elif resolved_problem_type == "classification":
        (
            classification_statistics,
            classification_warnings,
            classification_recommendations,
        ) = _classification_target_statistics(
            observed_target,
            config,
        )
        warnings.extend(classification_warnings)
        recommendations.extend(
            classification_recommendations
        )
    elif resolved_problem_type == "regression":
        (
            regression_statistics,
            regression_warnings,
            regression_recommendations,
        ) = _regression_target_statistics(
            observed_target,
            config,
        )
        warnings.extend(regression_warnings)
        recommendations.extend(
            regression_recommendations
        )

    if resolved_problem_type == "classification":
        training_usable = bool(
            classification_statistics
            and classification_statistics["class_count"] >= 2
        )
    elif resolved_problem_type == "regression":
        training_usable = bool(
            regression_statistics
            and regression_statistics[
                "unique_finite_values"
            ]
            >= 2
            and regression_statistics["finite_count"] > 0
            and regression_statistics[
                "non_numeric_values"
            ]
            == 0
            and regression_statistics["infinite_values"] == 0
        )
    else:
        training_usable = False

    return {
        "target_column": target_column,
        "data_type": str(target_series.dtype),
        "problem_type": resolved_problem_type,
        "problem_type_source": problem_type_source,
        "target_analysis": target_analysis,
        "total_rows": total_rows,
        "observed_target_rows": observed_count,
        "missing_target_rows": missing_count,
        "missing_target_percentage": round(
            missing_count / total_rows * 100,
            2,
        ),
        "unique_observed_values": unique_count,
        "training_usable": training_usable,
        "classification_statistics": (
            classification_statistics
        ),
        "regression_statistics": regression_statistics,
        "warnings": warnings,
        "recommendations": recommendations,
        "automatic_target_modification_performed": False,
    }