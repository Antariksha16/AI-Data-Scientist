import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.eda_config import EDAConfig
from ml_pipeline.eda_statistics import (
    generate_datetime_statistics,
)
from ml_pipeline.eda_target import (
    generate_target_statistics,
)


def _safe_number(
    value: Any,
    decimal_places: int,
) -> float | None:
    """Convert a value into a finite JSON-safe number."""

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
    """Convert a scalar into a JSON-safe Python value."""

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


def _relationship_strength(
    score: float | None,
    config: EDAConfig,
) -> str:
    """Map a bounded absolute association score to a label."""

    if score is None:
        return "undetermined"

    absolute_score = abs(score)

    if absolute_score < config.weak_relationship_threshold:
        return "negligible"

    if absolute_score < config.moderate_relationship_threshold:
        return "weak"

    if absolute_score < config.strong_relationship_threshold:
        return "moderate"

    return "strong"


def _base_result(
    feature: str,
    feature_type: str,
    problem_type: str,
    analysis_method: str,
    sample_size: int,
    total_rows: int,
) -> dict:
    """Create fields shared by every relationship result."""

    return {
        "feature": feature,
        "feature_type": feature_type,
        "target_problem_type": problem_type,
        "analysis_method": analysis_method,
        "status": "analyzed",
        "sample_size": sample_size,
        "excluded_pairwise_rows": total_rows - sample_size,
        "primary_score": None,
        "relationship_strength": "undetermined",
        "potential_leakage": False,
        "warnings": [],
    }


def _skip_result(
    feature: str,
    feature_type: str,
    problem_type: str,
    analysis_method: str,
    sample_size: int,
    total_rows: int,
    reason: str,
) -> dict:
    """Create a transparent result for a skipped relationship."""

    result = _base_result(
        feature,
        feature_type,
        problem_type,
        analysis_method,
        sample_size,
        total_rows,
    )
    result["status"] = "skipped"
    result["warnings"] = [reason]
    return result


def _numeric_pair(
    feature: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Return finite pairwise-complete numerical observations."""

    pair = pd.DataFrame(
        {
            "feature": pd.to_numeric(
                feature,
                errors="coerce",
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    )

    return pair.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()


def _eta_squared(
    numerical_values: pd.Series,
    categories: pd.Series,
) -> float | None:
    """Calculate ANOVA eta-squared without requiring SciPy."""

    if len(numerical_values) == 0:
        return None

    frame = pd.DataFrame(
        {
            "value": numerical_values.to_numpy(),
            "category": categories.to_numpy(),
        }
    )

    if frame["category"].nunique() < 2:
        return None

    grand_mean = frame["value"].mean()
    total_sum_of_squares = float(
        ((frame["value"] - grand_mean) ** 2).sum()
    )

    if total_sum_of_squares <= 0:
        return None

    grouped = frame.groupby(
        "category",
        dropna=False,
        observed=True,
    )["value"]

    between_group_sum_of_squares = float(
        sum(
            len(group)
            * (group.mean() - grand_mean) ** 2
            for _, group in grouped
        )
    )

    eta_squared = (
        between_group_sum_of_squares
        / total_sum_of_squares
    )

    return min(max(eta_squared, 0.0), 1.0)


def _cramers_v(
    contingency_table: pd.DataFrame,
) -> float | None:
    """Calculate bias-corrected Cramer's V without SciPy."""

    observed = contingency_table.to_numpy(dtype=float)
    sample_size = observed.sum()

    if sample_size <= 1:
        return None

    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / sample_size

    valid_expected = expected > 0

    if not valid_expected.any():
        return None

    chi_squared = float(
        np.sum(
            (
                (observed[valid_expected]
                - expected[valid_expected])
                ** 2
            )
            / expected[valid_expected]
        )
    )

    phi_squared = chi_squared / sample_size
    row_count, column_count = observed.shape

    corrected_phi_squared = max(
        0.0,
        phi_squared
        - (
            (column_count - 1)
            * (row_count - 1)
            / (sample_size - 1)
        ),
    )

    corrected_row_count = (
        row_count
        - ((row_count - 1) ** 2)
        / (sample_size - 1)
    )
    corrected_column_count = (
        column_count
        - ((column_count - 1) ** 2)
        / (sample_size - 1)
    )

    denominator = min(
        corrected_row_count - 1,
        corrected_column_count - 1,
    )

    if denominator <= 0:
        return None

    value = math.sqrt(
        corrected_phi_squared / denominator
    )

    return min(max(value, 0.0), 1.0)


def _numerical_regression_relationship(
    feature_name: str,
    feature: pd.Series,
    target: pd.Series,
    feature_type: str,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Analyze numerical or ordinal-datetime versus regression."""

    pair = _numeric_pair(feature, target)
    method = "pearson_and_spearman_correlation"

    if feature_type == "datetime":
        method = "temporal_pearson_and_spearman_correlation"

    if len(pair) < config.minimum_sample_size:
        return _skip_result(
            feature_name,
            feature_type,
            "regression",
            method,
            len(pair),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    if pair["feature"].nunique() < 2:
        return _skip_result(
            feature_name,
            feature_type,
            "regression",
            method,
            len(pair),
            total_rows,
            "The feature is constant in pairwise-complete rows.",
        )

    if pair["target"].nunique() < 2:
        return _skip_result(
            feature_name,
            feature_type,
            "regression",
            method,
            len(pair),
            total_rows,
            "The target is constant in pairwise-complete rows.",
        )

    pearson = pair["feature"].corr(
        pair["target"],
        method="pearson",
    )
    spearman = pair["feature"].corr(
        pair["target"],
        method="spearman",
    )

    safe_pearson = _safe_number(
        pearson,
        config.decimal_places,
    )
    safe_spearman = _safe_number(
        spearman,
        config.decimal_places,
    )

    available_scores = [
        float(score)
        for score in [pearson, spearman]
        if score is not None
        and not pd.isna(score)
        and math.isfinite(float(score))
    ]

    primary_score = (
        max(abs(score) for score in available_scores)
        if available_scores
        else None
    )

    result = _base_result(
        feature_name,
        feature_type,
        "regression",
        method,
        len(pair),
        total_rows,
    )

    result.update(
        {
            "pearson_correlation": safe_pearson,
            "spearman_correlation": safe_spearman,
            "primary_score": _safe_number(
                primary_score,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(
                    primary_score,
                    config,
                )
            ),
            "direction": (
                "positive"
                if safe_pearson is not None
                and safe_pearson > 0
                else "negative"
                if safe_pearson is not None
                and safe_pearson < 0
                else "none_or_undetermined"
            ),
            "potential_leakage": bool(
                primary_score is not None
                and primary_score
                >= config.leakage_correlation_threshold
            ),
        }
    )

    if (
        safe_pearson is not None
        and safe_spearman is not None
        and abs(
            abs(safe_spearman)
            - abs(safe_pearson)
        )
        >= config.weak_relationship_threshold
    ):
        result["warnings"].append(
            "Pearson and Spearman differ meaningfully; review "
            "possible monotonic nonlinearity or influential values."
        )

    if feature_type == "datetime":
        result["temporal_encoding"] = (
            "UTC days since 1970-01-01, used only for "
            "relationship calculation"
        )

    return result


def _group_summaries(
    frame: pd.DataFrame,
    config: EDAConfig,
) -> list[dict]:
    """Return capped, JSON-safe numerical summaries by category."""

    summaries = []

    grouped = frame.groupby(
        "category",
        dropna=False,
        observed=True,
        sort=False,
    )["value"]

    ordered_groups = sorted(
        grouped,
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for category, values in ordered_groups[
        : config.max_categories_displayed
    ]:
        summaries.append(
            {
                "category": _safe_value(category),
                "count": len(values),
                "mean": _safe_number(
                    values.mean(),
                    config.decimal_places,
                ),
                "median": _safe_number(
                    values.median(),
                    config.decimal_places,
                ),
                "standard_deviation": _safe_number(
                    values.std()
                    if len(values) > 1
                    else None,
                    config.decimal_places,
                ),
            }
        )

    return summaries


def _numerical_classification_relationship(
    feature_name: str,
    feature: pd.Series,
    target: pd.Series,
    feature_type: str,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Analyze numerical or ordinal-datetime versus classes."""

    numeric_feature = pd.to_numeric(
        feature,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    valid_mask = numeric_feature.notna() & target.notna()
    frame = pd.DataFrame(
        {
            "value": numeric_feature.loc[valid_mask],
            "category": target.loc[valid_mask],
        }
    )

    method = "eta_squared_and_group_statistics"
    if feature_type == "datetime":
        method = "temporal_eta_squared_and_group_statistics"

    if len(frame) < config.minimum_sample_size:
        return _skip_result(
            feature_name,
            feature_type,
            "classification",
            method,
            len(frame),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    if frame["value"].nunique() < 2:
        return _skip_result(
            feature_name,
            feature_type,
            "classification",
            method,
            len(frame),
            total_rows,
            "The feature is constant in pairwise-complete rows.",
        )

    class_count = int(frame["category"].nunique())
    if class_count < 2:
        return _skip_result(
            feature_name,
            feature_type,
            "classification",
            method,
            len(frame),
            total_rows,
            "The target has fewer than two observed classes.",
        )

    eta_squared = _eta_squared(
        frame["value"],
        frame["category"],
    )
    eta = (
        math.sqrt(eta_squared)
        if eta_squared is not None
        else None
    )

    result = _base_result(
        feature_name,
        feature_type,
        "classification",
        method,
        len(frame),
        total_rows,
    )

    result.update(
        {
            "class_count": class_count,
            "eta_squared": _safe_number(
                eta_squared,
                config.decimal_places,
            ),
            "eta": _safe_number(
                eta,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                eta,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(eta, config)
            ),
            "group_statistics": _group_summaries(
                frame,
                config,
            ),
            "groups_truncated": (
                class_count
                > config.max_categories_displayed
            ),
            "potential_leakage": bool(
                eta is not None
                and eta
                >= config.leakage_correlation_threshold
            ),
        }
    )

    group_counts = frame["category"].value_counts()
    if int(group_counts.min()) < 2:
        result["warnings"].append(
            "At least one target class has fewer than two usable "
            "feature observations."
        )

    if class_count == 2:
        class_values = list(group_counts.index)
        first_group = frame.loc[
            frame["category"] == class_values[0],
            "value",
        ]
        second_group = frame.loc[
            frame["category"] == class_values[1],
            "value",
        ]

        pooled_denominator = (
            len(first_group) + len(second_group) - 2
        )

        if (
            len(first_group) > 1
            and len(second_group) > 1
            and pooled_denominator > 0
        ):
            pooled_variance = (
                (len(first_group) - 1)
                * first_group.var()
                + (len(second_group) - 1)
                * second_group.var()
            ) / pooled_denominator

            pooled_standard_deviation = math.sqrt(
                max(float(pooled_variance), 0.0)
            )

            standardized_mean_difference = (
                (
                    first_group.mean()
                    - second_group.mean()
                )
                / pooled_standard_deviation
                if pooled_standard_deviation > 0
                else None
            )
        else:
            standardized_mean_difference = None

        result["standardized_mean_difference"] = (
            _safe_number(
                standardized_mean_difference,
                config.decimal_places,
            )
        )
        result["standardized_mean_difference_classes"] = [
            _safe_value(class_values[0]),
            _safe_value(class_values[1]),
        ]

    if feature_type == "datetime":
        result["temporal_encoding"] = (
            "UTC days since 1970-01-01, used only for "
            "relationship calculation"
        )

    return result


def _categorical_regression_relationship(
    feature_name: str,
    feature: pd.Series,
    target: pd.Series,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Analyze categorical versus numerical regression target."""

    numeric_target = pd.to_numeric(
        target,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    valid_mask = feature.notna() & numeric_target.notna()
    frame = pd.DataFrame(
        {
            "category": feature.loc[valid_mask].map(str),
            "value": numeric_target.loc[valid_mask],
        }
    )
    method = "correlation_ratio_eta"

    if len(frame) < config.minimum_sample_size:
        return _skip_result(
            feature_name,
            "categorical",
            "regression",
            method,
            len(frame),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    category_count = int(frame["category"].nunique())

    if category_count < 2:
        return _skip_result(
            feature_name,
            "categorical",
            "regression",
            method,
            len(frame),
            total_rows,
            "The feature has fewer than two observed categories.",
        )

    if category_count > config.max_relationship_categories:
        return _skip_result(
            feature_name,
            "categorical",
            "regression",
            method,
            len(frame),
            total_rows,
            "Categorical cardinality exceeds the configured "
            "relationship-analysis safety limit.",
        )

    if frame["value"].nunique() < 2:
        return _skip_result(
            feature_name,
            "categorical",
            "regression",
            method,
            len(frame),
            total_rows,
            "The target is constant in pairwise-complete rows.",
        )

    eta_squared = _eta_squared(
        frame["value"],
        frame["category"],
    )
    eta = (
        math.sqrt(eta_squared)
        if eta_squared is not None
        else None
    )

    result = _base_result(
        feature_name,
        "categorical",
        "regression",
        method,
        len(frame),
        total_rows,
    )

    result.update(
        {
            "category_count": category_count,
            "eta_squared": _safe_number(
                eta_squared,
                config.decimal_places,
            ),
            "eta": _safe_number(
                eta,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                eta,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(eta, config)
            ),
            "group_statistics": _group_summaries(
                frame,
                config,
            ),
            "groups_truncated": (
                category_count
                > config.max_categories_displayed
            ),
            "potential_leakage": bool(
                eta is not None
                and eta
                >= config.leakage_correlation_threshold
            ),
        }
    )

    category_ratio = category_count / len(frame)
    if (
        category_count
        > config.high_cardinality_count_threshold
        and category_ratio
        > config.high_cardinality_ratio_threshold
    ):
        result["warnings"].append(
            "High feature cardinality can inflate apparent "
            "category-target association."
        )

    return result


def _categorical_classification_relationship(
    feature_name: str,
    feature: pd.Series,
    target: pd.Series,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Analyze categorical feature versus categorical target."""

    valid_mask = feature.notna() & target.notna()
    frame = pd.DataFrame(
        {
            "feature": feature.loc[valid_mask].map(str),
            "target": target.loc[valid_mask].map(str),
        }
    )
    method = "bias_corrected_cramers_v"

    if len(frame) < config.minimum_sample_size:
        return _skip_result(
            feature_name,
            "categorical",
            "classification",
            method,
            len(frame),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    feature_category_count = int(
        frame["feature"].nunique()
    )
    target_class_count = int(frame["target"].nunique())

    if feature_category_count < 2:
        return _skip_result(
            feature_name,
            "categorical",
            "classification",
            method,
            len(frame),
            total_rows,
            "The feature has fewer than two observed categories.",
        )

    if target_class_count < 2:
        return _skip_result(
            feature_name,
            "categorical",
            "classification",
            method,
            len(frame),
            total_rows,
            "The target has fewer than two observed classes.",
        )

    if (
        feature_category_count
        > config.max_relationship_categories
        or target_class_count
        > config.max_relationship_categories
    ):
        return _skip_result(
            feature_name,
            "categorical",
            "classification",
            method,
            len(frame),
            total_rows,
            "Categorical cardinality exceeds the configured "
            "relationship-analysis safety limit.",
        )

    contingency_table = pd.crosstab(
        frame["feature"],
        frame["target"],
        dropna=False,
    )

    cramers_v = _cramers_v(contingency_table)

    sparse_cell_ratio = float(
        (contingency_table.to_numpy() < 5).mean()
    )

    conditional_summaries = []
    feature_counts = frame["feature"].value_counts()

    for category in feature_counts.index[
        : config.max_categories_displayed
    ]:
        category_targets = frame.loc[
            frame["feature"] == category,
            "target",
        ].value_counts()
        top_target = category_targets.index[0]
        top_count = int(category_targets.iloc[0])
        category_count = int(feature_counts.loc[category])

        conditional_summaries.append(
            {
                "feature_category": category,
                "count": category_count,
                "most_common_target": top_target,
                "most_common_target_count": top_count,
                "most_common_target_percentage": round(
                    top_count / category_count * 100,
                    2,
                ),
            }
        )

    result = _base_result(
        feature_name,
        "categorical",
        "classification",
        method,
        len(frame),
        total_rows,
    )

    result.update(
        {
            "feature_category_count": (
                feature_category_count
            ),
            "target_class_count": target_class_count,
            "cramers_v": _safe_number(
                cramers_v,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                cramers_v,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(
                    cramers_v,
                    config,
                )
            ),
            "sparse_contingency_cell_percentage": round(
                sparse_cell_ratio * 100,
                2,
            ),
            "conditional_target_summaries": (
                conditional_summaries
            ),
            "categories_truncated": (
                feature_category_count
                > config.max_categories_displayed
            ),
            "potential_leakage": bool(
                cramers_v is not None
                and cramers_v
                >= config.leakage_correlation_threshold
            ),
        }
    )

    if sparse_cell_ratio > 0.50:
        result["warnings"].append(
            "More than half of contingency cells contain fewer "
            "than five observations; association may be unstable."
        )

    category_ratio = feature_category_count / len(frame)
    if (
        feature_category_count
        > config.high_cardinality_count_threshold
        and category_ratio
        > config.high_cardinality_ratio_threshold
    ):
        result["warnings"].append(
            "High feature cardinality can inflate Cramer's V."
        )

    return result


def _datetime_as_numeric_days(
    series: pd.Series,
) -> pd.Series:
    """Parse datetime values and encode them as UTC epoch days."""

    try:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
            utc=True,
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )

    numeric_days = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )
    valid_mask = parsed.notna()

    numeric_days.loc[valid_mask] = (
        parsed.loc[valid_mask].astype("int64")
        / 86_400_000_000_000
    )

    return numeric_days


def generate_feature_target_relationships(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
    problem_type: str | None = None,
) -> dict:
    """
    Analyze every candidate feature against a supervised target.

    The function chooses an association measure based on feature
    type and problem type. It reports possible leakage but never
    removes features or modifies the input DataFrame.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()

    target_report = generate_target_statistics(
        df,
        target_column,
        config=config,
        problem_type=problem_type,
    )
    resolved_problem_type = target_report["problem_type"]

    if resolved_problem_type not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            "Feature-target relationships require a resolved "
            "classification or regression problem type."
        )

    datetime_report = generate_datetime_statistics(
        df,
        target_column,
        config,
    )
    datetime_columns = set(
        datetime_report["datetime_columns"]
    )

    target = df[target_column]
    results = {}

    for feature_name in df.columns:
        if feature_name == target_column:
            continue

        feature = df[feature_name]

        try:
            if feature_name in datetime_columns:
                numeric_datetime = _datetime_as_numeric_days(
                    feature
                )

                if resolved_problem_type == "regression":
                    result = _numerical_regression_relationship(
                        feature_name,
                        numeric_datetime,
                        target,
                        "datetime",
                        config,
                        len(df),
                    )
                else:
                    result = (
                        _numerical_classification_relationship(
                            feature_name,
                            numeric_datetime,
                            target,
                            "datetime",
                            config,
                            len(df),
                        )
                    )

            elif (
                pd.api.types.is_numeric_dtype(feature.dtype)
                and not pd.api.types.is_bool_dtype(
                    feature.dtype
                )
            ):
                if resolved_problem_type == "regression":
                    result = _numerical_regression_relationship(
                        feature_name,
                        feature,
                        target,
                        "numerical",
                        config,
                        len(df),
                    )
                else:
                    result = (
                        _numerical_classification_relationship(
                            feature_name,
                            feature,
                            target,
                            "numerical",
                            config,
                            len(df),
                        )
                    )

            elif resolved_problem_type == "regression":
                result = _categorical_regression_relationship(
                    feature_name,
                    feature,
                    target,
                    config,
                    len(df),
                )
            else:
                result = (
                    _categorical_classification_relationship(
                        feature_name,
                        feature,
                        target,
                        config,
                        len(df),
                    )
                )

        except (TypeError, ValueError, OverflowError) as error:
            result = _skip_result(
                feature_name,
                "unsupported_or_malformed",
                resolved_problem_type,
                "undetermined",
                0,
                len(df),
                "Relationship calculation could not be completed: "
                f"{error}",
            )

        results[feature_name] = result

    analyzed_results = [
        result
        for result in results.values()
        if result["status"] == "analyzed"
        and result["primary_score"] is not None
    ]

    ranked_results = sorted(
        analyzed_results,
        key=lambda item: item["primary_score"],
        reverse=True,
    )

    ranked_features = [
        {
            "feature": result["feature"],
            "feature_type": result["feature_type"],
            "analysis_method": result["analysis_method"],
            "primary_score": result["primary_score"],
            "relationship_strength": result[
                "relationship_strength"
            ],
            "potential_leakage": result[
                "potential_leakage"
            ],
        }
        for result in ranked_results
    ]

    skipped_features = [
        feature_name
        for feature_name, result in results.items()
        if result["status"] == "skipped"
    ]

    strong_relationships = [
        result["feature"]
        for result in analyzed_results
        if result["relationship_strength"] == "strong"
    ]

    potential_leakage_features = [
        result["feature"]
        for result in analyzed_results
        if result["potential_leakage"]
    ]

    return {
        "target_column": target_column,
        "problem_type": resolved_problem_type,
        "candidate_feature_count": len(df.columns) - 1,
        "analyzed_feature_count": len(analyzed_results),
        "skipped_feature_count": len(skipped_features),
        "skipped_features": skipped_features,
        "relationship_thresholds": {
            "weak": config.weak_relationship_threshold,
            "moderate": (
                config.moderate_relationship_threshold
            ),
            "strong": config.strong_relationship_threshold,
            "potential_leakage": (
                config.leakage_correlation_threshold
            ),
        },
        "feature_relationships": results,
        "ranked_features": ranked_features,
        "strong_relationships": strong_relationships,
        "potential_leakage_feature_count": len(
            potential_leakage_features
        ),
        "potential_leakage_features": (
            potential_leakage_features
        ),
        "automatic_feature_removal_performed": False,
        "limitations": [
            (
                "Association does not prove causation or business "
                "usefulness."
            ),
            (
                "Pearson and Spearman do not detect every nonlinear "
                "numerical relationship."
            ),
            (
                "Cramer's V and eta can be unstable for sparse or "
                "high-cardinality groups."
            ),
            (
                "A near-perfect relationship is only a leakage "
                "review flag; semantic and time-based review is "
                "still required."
            ),
        ],
    }


def _feature_pair_base(
    feature_1: str,
    feature_2: str,
    feature_1_type: str,
    feature_2_type: str,
    analysis_method: str,
    sample_size: int,
    total_rows: int,
) -> dict:
    """Create fields shared by feature-feature results."""

    return {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_1_type": feature_1_type,
        "feature_2_type": feature_2_type,
        "analysis_method": analysis_method,
        "status": "analyzed",
        "sample_size": sample_size,
        "excluded_pairwise_rows": total_rows - sample_size,
        "primary_score": None,
        "relationship_strength": "undetermined",
        "high_association": False,
        "warnings": [],
    }


def _skip_feature_pair(
    feature_1: str,
    feature_2: str,
    feature_1_type: str,
    feature_2_type: str,
    analysis_method: str,
    sample_size: int,
    total_rows: int,
    reason: str,
) -> dict:
    """Return an auditable skipped feature-pair result."""

    result = _feature_pair_base(
        feature_1,
        feature_2,
        feature_1_type,
        feature_2_type,
        analysis_method,
        sample_size,
        total_rows,
    )
    result["status"] = "skipped"
    result["warnings"] = [reason]
    return result


def _numerical_feature_pair(
    feature_1_name: str,
    feature_2_name: str,
    feature_1: pd.Series,
    feature_2: pd.Series,
    feature_1_type: str,
    feature_2_type: str,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Calculate Pearson and Spearman for two ordinal features."""

    pair = _numeric_pair(feature_1, feature_2)
    method = "pearson_and_spearman_correlation"

    if "datetime" in {feature_1_type, feature_2_type}:
        method = "temporal_pearson_and_spearman_correlation"

    if len(pair) < config.minimum_sample_size:
        return _skip_feature_pair(
            feature_1_name,
            feature_2_name,
            feature_1_type,
            feature_2_type,
            method,
            len(pair),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    if (
        pair["feature"].nunique() < 2
        or pair["target"].nunique() < 2
    ):
        return _skip_feature_pair(
            feature_1_name,
            feature_2_name,
            feature_1_type,
            feature_2_type,
            method,
            len(pair),
            total_rows,
            "At least one feature is constant in pairwise-complete "
            "rows.",
        )

    pearson = pair["feature"].corr(
        pair["target"],
        method="pearson",
    )
    spearman = pair["feature"].corr(
        pair["target"],
        method="spearman",
    )

    finite_scores = [
        float(score)
        for score in [pearson, spearman]
        if score is not None
        and not pd.isna(score)
        and math.isfinite(float(score))
    ]
    primary_score = (
        max(abs(score) for score in finite_scores)
        if finite_scores
        else None
    )

    result = _feature_pair_base(
        feature_1_name,
        feature_2_name,
        feature_1_type,
        feature_2_type,
        method,
        len(pair),
        total_rows,
    )
    result.update(
        {
            "pearson_correlation": _safe_number(
                pearson,
                config.decimal_places,
            ),
            "spearman_correlation": _safe_number(
                spearman,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                primary_score,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(
                    primary_score,
                    config,
                )
            ),
            "high_association": bool(
                primary_score is not None
                and primary_score
                >= config.high_correlation_threshold
            ),
        }
    )

    return result


def _categorical_feature_pair(
    feature_1_name: str,
    feature_2_name: str,
    feature_1: pd.Series,
    feature_2: pd.Series,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Calculate bias-corrected Cramer's V for two categories."""

    valid_mask = feature_1.notna() & feature_2.notna()
    frame = pd.DataFrame(
        {
            "feature_1": feature_1.loc[valid_mask].map(str),
            "feature_2": feature_2.loc[valid_mask].map(str),
        }
    )
    method = "bias_corrected_cramers_v"

    if len(frame) < config.minimum_sample_size:
        return _skip_feature_pair(
            feature_1_name,
            feature_2_name,
            "categorical",
            "categorical",
            method,
            len(frame),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    if (
        frame["feature_1"].nunique() < 2
        or frame["feature_2"].nunique() < 2
    ):
        return _skip_feature_pair(
            feature_1_name,
            feature_2_name,
            "categorical",
            "categorical",
            method,
            len(frame),
            total_rows,
            "At least one feature has fewer than two observed "
            "categories.",
        )

    contingency_table = pd.crosstab(
        frame["feature_1"],
        frame["feature_2"],
        dropna=False,
    )
    cramers_v = _cramers_v(contingency_table)
    sparse_cell_ratio = float(
        (contingency_table.to_numpy() < 5).mean()
    )

    result = _feature_pair_base(
        feature_1_name,
        feature_2_name,
        "categorical",
        "categorical",
        method,
        len(frame),
        total_rows,
    )
    result.update(
        {
            "cramers_v": _safe_number(
                cramers_v,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                cramers_v,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(
                    cramers_v,
                    config,
                )
            ),
            "high_association": bool(
                cramers_v is not None
                and cramers_v
                >= config.high_correlation_threshold
            ),
            "sparse_contingency_cell_percentage": round(
                sparse_cell_ratio * 100,
                2,
            ),
        }
    )

    if sparse_cell_ratio > 0.50:
        result["warnings"].append(
            "More than half of contingency cells contain fewer "
            "than five observations; association may be unstable."
        )

    return result


def _mixed_feature_pair(
    numerical_name: str,
    categorical_name: str,
    numerical_feature: pd.Series,
    categorical_feature: pd.Series,
    numerical_type: str,
    config: EDAConfig,
    total_rows: int,
) -> dict:
    """Calculate eta for a numerical-categorical feature pair."""

    numeric_values = pd.to_numeric(
        numerical_feature,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    valid_mask = (
        numeric_values.notna()
        & categorical_feature.notna()
    )
    frame = pd.DataFrame(
        {
            "value": numeric_values.loc[valid_mask],
            "category": categorical_feature.loc[
                valid_mask
            ].map(str),
        }
    )
    method = "correlation_ratio_eta"

    if len(frame) < config.minimum_sample_size:
        return _skip_feature_pair(
            numerical_name,
            categorical_name,
            numerical_type,
            "categorical",
            method,
            len(frame),
            total_rows,
            "Too few pairwise-complete rows for reliable analysis.",
        )

    if (
        frame["value"].nunique() < 2
        or frame["category"].nunique() < 2
    ):
        return _skip_feature_pair(
            numerical_name,
            categorical_name,
            numerical_type,
            "categorical",
            method,
            len(frame),
            total_rows,
            "At least one feature is constant in pairwise-complete "
            "rows.",
        )

    eta_squared = _eta_squared(
        frame["value"],
        frame["category"],
    )
    eta = (
        math.sqrt(eta_squared)
        if eta_squared is not None
        else None
    )

    result = _feature_pair_base(
        numerical_name,
        categorical_name,
        numerical_type,
        "categorical",
        method,
        len(frame),
        total_rows,
    )
    result.update(
        {
            "eta_squared": _safe_number(
                eta_squared,
                config.decimal_places,
            ),
            "eta": _safe_number(
                eta,
                config.decimal_places,
            ),
            "primary_score": _safe_number(
                eta,
                config.decimal_places,
            ),
            "relationship_strength": (
                _relationship_strength(eta, config)
            ),
            "high_association": bool(
                eta is not None
                and eta
                >= config.high_correlation_threshold
            ),
        }
    )

    return result


def _build_association_matrix(
    features: list[str],
) -> dict[str, dict[str, float | None]]:
    """Create a JSON-safe square association matrix."""

    return {
        row_feature: {
            column_feature: (
                1.0
                if row_feature == column_feature
                else None
            )
            for column_feature in features
        }
        for row_feature in features
    }


def _redundancy_groups(
    selected_features: list[str],
    high_association_pairs: list[dict],
) -> list[list[str]]:
    """Build connected redundancy groups from high-association pairs."""

    parent = {
        feature: feature
        for feature in selected_features
    }

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for pair in high_association_pairs:
        union(pair["feature_1"], pair["feature_2"])

    grouped_features: dict[str, list[str]] = {}
    for feature in selected_features:
        root = find(feature)
        grouped_features.setdefault(root, []).append(feature)

    return [
        features
        for features in grouped_features.values()
        if len(features) > 1
    ]


def _calculate_vif_report(
    numerical_features: list[str],
    numerical_series: dict[str, pd.Series],
    config: EDAConfig,
) -> dict:
    """Calculate VIF using NumPy least-squares regression."""

    excluded_features = []
    usable_data = {}

    for feature in numerical_features:
        values = pd.to_numeric(
            numerical_series[feature],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        observed_values = values.dropna()
        if (
            observed_values.empty
            or observed_values.nunique() < 2
        ):
            excluded_features.append(feature)
            continue

        usable_data[feature] = values

    usable_features = list(usable_data)

    if len(usable_features) < 2:
        return {
            "status": "skipped",
            "reason": (
                "At least two nonconstant numerical features are "
                "required for VIF."
            ),
            "features_analyzed": [],
            "features_excluded": excluded_features,
            "vif_statistics": [],
            "moderate_vif_features": [],
            "high_vif_features": [],
            "automatic_feature_removal_performed": False,
        }

    frame = pd.DataFrame(usable_data)
    median_imputed_value_count = int(
        frame.isna().sum().sum()
    )
    frame = frame.fillna(frame.median())

    if (
        len(frame) < config.minimum_sample_size
        or len(frame) <= len(usable_features) + 1
    ):
        return {
            "status": "skipped",
            "reason": (
                "Too few rows relative to the number of numerical "
                "features for stable VIF calculation."
            ),
            "features_analyzed": [],
            "features_excluded": excluded_features,
            "vif_statistics": [],
            "moderate_vif_features": [],
            "high_vif_features": [],
            "automatic_feature_removal_performed": False,
        }

    matrix = frame.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    standardized = (
        matrix - means
    ) / standard_deviations

    condition_number = float(
        np.linalg.cond(standardized)
    )

    vif_statistics = []

    for index, feature in enumerate(usable_features):
        dependent = standardized[:, index]
        predictors = np.delete(
            standardized,
            index,
            axis=1,
        )
        predictors = np.column_stack(
            [
                np.ones(len(predictors)),
                predictors,
            ]
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            predictors,
            dependent,
            rcond=None,
        )
        predicted = predictors @ coefficients
        residual_sum_of_squares = float(
            ((dependent - predicted) ** 2).sum()
        )
        total_sum_of_squares = float(
            ((dependent - dependent.mean()) ** 2).sum()
        )

        r_squared = (
            1 - residual_sum_of_squares
            / total_sum_of_squares
            if total_sum_of_squares > 0
            else 1.0
        )
        r_squared = min(max(r_squared, 0.0), 1.0)

        is_infinite = r_squared >= 1 - 1e-12
        vif = (
            None
            if is_infinite
            else max(1.0, 1 / (1 - r_squared))
        )

        if is_infinite or (
            vif is not None
            and vif >= config.high_vif_threshold
        ):
            severity = "high"
        elif (
            vif is not None
            and vif >= config.moderate_vif_threshold
        ):
            severity = "moderate"
        else:
            severity = "low"

        vif_statistics.append(
            {
                "feature": feature,
                "vif": _safe_number(
                    vif,
                    config.decimal_places,
                ),
                "is_infinite": is_infinite,
                "r_squared_against_other_features": (
                    _safe_number(
                        r_squared,
                        config.decimal_places,
                    )
                ),
                "severity": severity,
            }
        )

    vif_statistics.sort(
        key=lambda item: (
            item["is_infinite"],
            item["vif"] or 0.0,
        ),
        reverse=True,
    )

    moderate_vif_features = [
        item["feature"]
        for item in vif_statistics
        if item["severity"] == "moderate"
    ]
    high_vif_features = [
        item["feature"]
        for item in vif_statistics
        if item["severity"] == "high"
    ]

    return {
        "status": "analyzed",
        "row_count": len(frame),
        "feature_count": len(usable_features),
        "features_analyzed": usable_features,
        "features_excluded": excluded_features,
        "median_imputation_used": (
            median_imputed_value_count > 0
        ),
        "median_imputed_value_count": (
            median_imputed_value_count
        ),
        "condition_number": _safe_number(
            condition_number,
            config.decimal_places,
        ),
        "moderate_vif_threshold": (
            config.moderate_vif_threshold
        ),
        "high_vif_threshold": config.high_vif_threshold,
        "vif_statistics": vif_statistics,
        "moderate_vif_features": moderate_vif_features,
        "high_vif_features": high_vif_features,
        "automatic_feature_removal_performed": False,
        "interpretation": (
            "VIF measures how well each numerical feature can be "
            "predicted from the other numerical features."
        ),
    }


def generate_feature_feature_relationships(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
) -> dict:
    """
    Analyze numerical, categorical, mixed, and datetime feature pairs.

    The target is excluded. High associations and VIF values are
    review signals only; no feature is removed automatically.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()
    datetime_report = generate_datetime_statistics(
        df,
        target_column,
        config,
    )
    datetime_columns = set(
        datetime_report["datetime_columns"]
    )

    feature_metadata = {}
    excluded_high_cardinality_features = []

    for original_position, feature in enumerate(df.columns):
        if feature == target_column:
            continue

        series = df[feature]
        observed_count = int(series.notna().sum())

        if feature in datetime_columns:
            feature_type = "datetime"
            analysis_series = _datetime_as_numeric_days(series)
        elif (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            feature_type = "numerical"
            analysis_series = series
        else:
            feature_type = "categorical"
            analysis_series = series
            category_count = int(
                series.dropna().map(str).nunique()
            )

            if (
                category_count
                > config.max_relationship_categories
            ):
                excluded_high_cardinality_features.append(
                    feature
                )
                continue

        feature_metadata[feature] = {
            "feature_type": feature_type,
            "analysis_series": analysis_series,
            "observed_count": observed_count,
            "original_position": original_position,
        }

    eligible_features = sorted(
        feature_metadata,
        key=lambda feature: (
            -feature_metadata[feature]["observed_count"],
            feature_metadata[feature]["original_position"],
        ),
    )

    selected_features = eligible_features[
        : config.max_pairwise_features
    ]
    omitted_due_feature_limit = eligible_features[
        config.max_pairwise_features :
    ]

    numerical_features = [
        feature
        for feature in selected_features
        if feature_metadata[feature]["feature_type"]
        in {"numerical", "datetime"}
    ]
    categorical_features = [
        feature
        for feature in selected_features
        if feature_metadata[feature]["feature_type"]
        == "categorical"
    ]
    datetime_features = [
        feature
        for feature in selected_features
        if feature_metadata[feature]["feature_type"]
        == "datetime"
    ]

    pearson_matrix = _build_association_matrix(
        numerical_features
    )
    spearman_matrix = _build_association_matrix(
        numerical_features
    )
    cramers_v_matrix = _build_association_matrix(
        categorical_features
    )
    mixed_eta_matrix = {
        numerical_feature: {
            categorical_feature: None
            for categorical_feature in categorical_features
        }
        for numerical_feature in numerical_features
    }

    pair_results = []

    for feature_1, feature_2 in combinations(
        numerical_features,
        2,
    ):
        metadata_1 = feature_metadata[feature_1]
        metadata_2 = feature_metadata[feature_2]
        result = _numerical_feature_pair(
            feature_1,
            feature_2,
            metadata_1["analysis_series"],
            metadata_2["analysis_series"],
            metadata_1["feature_type"],
            metadata_2["feature_type"],
            config,
            len(df),
        )
        pair_results.append(result)

        if result["status"] == "analyzed":
            pearson_matrix[feature_1][feature_2] = result[
                "pearson_correlation"
            ]
            pearson_matrix[feature_2][feature_1] = result[
                "pearson_correlation"
            ]
            spearman_matrix[feature_1][feature_2] = result[
                "spearman_correlation"
            ]
            spearman_matrix[feature_2][feature_1] = result[
                "spearman_correlation"
            ]

    for feature_1, feature_2 in combinations(
        categorical_features,
        2,
    ):
        result = _categorical_feature_pair(
            feature_1,
            feature_2,
            feature_metadata[feature_1]["analysis_series"],
            feature_metadata[feature_2]["analysis_series"],
            config,
            len(df),
        )
        pair_results.append(result)

        if result["status"] == "analyzed":
            cramers_v_matrix[feature_1][feature_2] = result[
                "cramers_v"
            ]
            cramers_v_matrix[feature_2][feature_1] = result[
                "cramers_v"
            ]

    for numerical_feature in numerical_features:
        for categorical_feature in categorical_features:
            numerical_metadata = feature_metadata[
                numerical_feature
            ]
            result = _mixed_feature_pair(
                numerical_feature,
                categorical_feature,
                numerical_metadata["analysis_series"],
                feature_metadata[categorical_feature][
                    "analysis_series"
                ],
                numerical_metadata["feature_type"],
                config,
                len(df),
            )
            pair_results.append(result)

            if result["status"] == "analyzed":
                mixed_eta_matrix[numerical_feature][
                    categorical_feature
                ] = result["eta"]

    analyzed_pairs = [
        pair
        for pair in pair_results
        if pair["status"] == "analyzed"
        and pair["primary_score"] is not None
    ]
    skipped_pairs = [
        pair
        for pair in pair_results
        if pair["status"] == "skipped"
    ]

    ranked_pairs = sorted(
        analyzed_pairs,
        key=lambda pair: pair["primary_score"],
        reverse=True,
    )
    high_association_pairs = [
        pair
        for pair in ranked_pairs
        if pair["high_association"]
    ]

    high_association_summaries = [
        {
            "feature_1": pair["feature_1"],
            "feature_2": pair["feature_2"],
            "analysis_method": pair["analysis_method"],
            "primary_score": pair["primary_score"],
            "relationship_strength": pair[
                "relationship_strength"
            ],
        }
        for pair in high_association_pairs
    ]

    numerical_series = {
        feature: feature_metadata[feature][
            "analysis_series"
        ]
        for feature in numerical_features
    }
    vif_report = _calculate_vif_report(
        numerical_features,
        numerical_series,
        config,
    )

    return {
        "target_column": target_column,
        "candidate_feature_count": len(df.columns) - 1,
        "eligible_feature_count": len(eligible_features),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "datetime_features": datetime_features,
        "excluded_high_cardinality_features": (
            excluded_high_cardinality_features
        ),
        "omitted_due_feature_limit": (
            omitted_due_feature_limit
        ),
        "max_pairwise_features": (
            config.max_pairwise_features
        ),
        "pair_count": len(pair_results),
        "analyzed_pair_count": len(analyzed_pairs),
        "skipped_pair_count": len(skipped_pairs),
        "pair_relationships": pair_results,
        "ranked_pairs": ranked_pairs,
        "pearson_matrix": pearson_matrix,
        "spearman_matrix": spearman_matrix,
        "cramers_v_matrix": cramers_v_matrix,
        "mixed_eta_matrix": mixed_eta_matrix,
        "high_association_threshold": (
            config.high_correlation_threshold
        ),
        "high_association_pair_count": len(
            high_association_pairs
        ),
        "high_association_pairs": (
            high_association_summaries
        ),
        "redundancy_groups": _redundancy_groups(
            selected_features,
            high_association_pairs,
        ),
        "multicollinearity_report": vif_report,
        "automatic_feature_removal_performed": False,
        "limitations": [
            (
                "High association does not prove that either "
                "feature should be removed."
            ),
            (
                "VIF is calculated only for original numerical and "
                "datetime features, before one-hot encoding."
            ),
            (
                "Median values are used only in the temporary VIF "
                "matrix; the source DataFrame is not modified."
            ),
            (
                "Pairwise association cannot identify every "
                "multivariate or nonlinear dependency."
            ),
        ],
    }