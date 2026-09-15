import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


INSIGHT_REPORT_VERSION = "1.0"

_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


def _json_safe(value: Any) -> Any:
    """Recursively convert values into strict JSON-compatible data."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def _as_dict(value: Any) -> dict:
    """Return a dictionary or an empty dictionary."""

    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    """Return a list or an empty list."""

    return value if isinstance(value, list) else []


def _column_text(columns: list[Any]) -> str:
    """Create readable column text without assuming string labels."""

    return ", ".join(str(column) for column in columns)


def _unique_text(items: list[Any]) -> list[str]:
    """Deduplicate non-empty messages while preserving their order."""

    output = []
    seen = set()

    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)

    return output


def generate_eda_insights(
    eda_report: dict,
    max_insights: int = 100,
) -> dict:
    """
    Convert a technical EDA report into prioritized human-readable insights.

    The function is deterministic, does not call an external language model,
    and never modifies the source report or underlying dataset. Each insight
    contains evidence, an explanation, a recommendation, and an explicit
    indication that no automatic action was performed.
    """

    if not isinstance(eda_report, dict):
        raise TypeError("eda_report must be a dictionary.")

    if not isinstance(max_insights, int) or isinstance(max_insights, bool):
        raise TypeError("max_insights must be an integer.")

    if max_insights <= 0:
        raise ValueError("max_insights must be greater than zero.")

    target_column = eda_report.get("target_column")
    if target_column is None:
        raise ValueError(
            "The EDA report must contain a target_column."
        )

    insights = []
    insight_ids = set()
    sequence = 0

    def add_insight(
        insight_id: str,
        severity: str,
        category: str,
        title: str,
        explanation: str,
        evidence: dict,
        recommendation: str,
        affected_columns: list[Any] | None = None,
        requires_human_review: bool = True,
    ) -> None:
        nonlocal sequence

        if severity not in _SEVERITY_ORDER:
            raise ValueError(
                f"Unsupported insight severity: {severity}"
            )

        if insight_id in insight_ids:
            return

        insight_ids.add(insight_id)
        insights.append(
            {
                "insight_id": insight_id,
                "severity": severity,
                "category": category,
                "title": title,
                "explanation": explanation,
                "evidence": evidence,
                "affected_columns": list(
                    affected_columns or []
                ),
                "recommendation": recommendation,
                "requires_human_review": (
                    requires_human_review
                ),
                "automatic_action_performed": False,
                "_sequence": sequence,
            }
        )
        sequence += 1

    report_status = str(
        eda_report.get("report_status", "unknown")
    )
    failed_sections = _as_list(
        eda_report.get("failed_sections")
    )
    section_errors = _as_list(
        eda_report.get("section_errors")
    )

    if report_status in {"partial", "failed"} or failed_sections:
        add_insight(
            "execution_incomplete_report",
            "high" if report_status == "failed" else "medium",
            "execution",
            "Some EDA sections did not complete",
            (
                "Insights from successful sections remain usable, but the "
                "missing sections create blind spots in the analysis."
            ),
            {
                "report_status": report_status,
                "failed_sections": failed_sections,
                "section_errors": section_errors,
            },
            (
                "Review the recorded section errors and rerun the affected "
                "analysis before making final modeling decisions."
            ),
            requires_human_review=True,
        )

    overview = _as_dict(eda_report.get("dataset_overview"))
    schema = _as_dict(overview.get("schema"))
    missingness = _as_dict(overview.get("missingness"))
    duplicates = _as_dict(overview.get("duplicates"))
    quality_flags = _as_dict(overview.get("quality_flags"))
    readiness = _as_dict(
        overview.get("basic_modeling_readiness")
    )

    duplicate_names = _as_list(
        schema.get("duplicate_column_names")
    )
    if schema and not schema.get("column_names_are_unique", True):
        add_insight(
            "schema_duplicate_column_names",
            "critical",
            "schema",
            "Duplicate column names make the schema ambiguous",
            (
                "Several pipeline operations select columns by name. A "
                "duplicate label can therefore select the wrong data or more "
                "than one column."
            ),
            {"duplicate_column_names": duplicate_names},
            (
                "Rename duplicate columns to unique, meaningful names before "
                "preprocessing or model training."
            ),
            duplicate_names,
        )

    blockers = _as_list(readiness.get("blockers"))
    if readiness and not readiness.get("ready", False):
        add_insight(
            "schema_modeling_readiness_blocked",
            "critical",
            "schema",
            "Basic modeling readiness checks failed",
            (
                "The dataset failed one or more basic schema or target "
                "usability checks and should not proceed directly to training."
            ),
            {"blockers": blockers},
            (
                "Resolve every readiness blocker, regenerate the EDA report, "
                "and confirm that the readiness flag becomes true."
            ),
        )

    high_missing_columns = _as_list(
        quality_flags.get(
            "high_missing_columns",
            missingness.get("high_missing_columns", []),
        )
    )
    if high_missing_columns:
        add_insight(
            "quality_high_missing_columns",
            "high",
            "data_quality",
            "Some features contain very high missingness",
            (
                "Features dominated by missing values may contribute little "
                "reliable signal and can make imputation unstable."
            ),
            {
                "columns": high_missing_columns,
                "configured_threshold": _as_dict(
                    eda_report.get("configuration")
                ).get("high_missing_threshold"),
            },
            (
                "Review the business meaning and missingness mechanism before "
                "dropping these columns or applying a specialized strategy."
            ),
            high_missing_columns,
        )

    high_missing_rows = missingness.get("high_missing_rows", 0) or 0
    if high_missing_rows > 0:
        add_insight(
            "quality_high_missing_rows",
            "medium",
            "data_quality",
            "Some records contain extensive missing data",
            (
                "Rows with many unavailable fields may be unreliable and can "
                "receive poor predictions after imputation."
            ),
            {
                "high_missing_rows": high_missing_rows,
                "threshold": missingness.get(
                    "row_missing_warning_threshold"
                ),
            },
            (
                "Inspect these records and define a documented row-retention "
                "rule using domain knowledge."
            ),
        )

    duplicate_rows = duplicates.get("duplicate_rows", 0) or 0
    if duplicate_rows > 0:
        add_insight(
            "quality_duplicate_rows",
            "medium",
            "data_quality",
            "Duplicate records were detected",
            (
                "Repeated rows can overweight observations and can leak nearly "
                "identical examples across train and validation sets."
            ),
            {
                "duplicate_rows": duplicate_rows,
                "duplicate_row_percentage": duplicates.get(
                    "duplicate_row_percentage"
                ),
            },
            (
                "Determine whether duplicates are legitimate repeated events "
                "or data-entry copies before deduplication and splitting."
            ),
        )

    identical_pairs = _as_list(
        duplicates.get("identical_column_pairs")
    )
    if identical_pairs:
        affected = []
        for pair in identical_pairs:
            if isinstance(pair, dict):
                affected.extend(
                    [pair.get("column_1"), pair.get("column_2")]
                )
            elif isinstance(pair, (list, tuple)):
                affected.extend(pair)
        affected = [item for item in affected if item is not None]
        add_insight(
            "quality_identical_columns",
            "high",
            "data_quality",
            "Exactly identical feature columns were detected",
            (
                "Duplicate features add no new information and can create "
                "multicollinearity or misleading importance results."
            ),
            {"identical_column_pairs": identical_pairs},
            (
                "Confirm the columns are semantically redundant and retain one "
                "authoritative version before training."
            ),
            affected,
        )

    for flag_key, insight_id, severity, title, explanation, recommendation in [
        (
            "constant_columns",
            "quality_constant_columns",
            "medium",
            "Constant or empty features provide no predictive variation",
            "These columns have only one observed value or no usable values.",
            "Remove confirmed constant features from the modeling feature set.",
        ),
        (
            "near_constant_columns",
            "quality_near_constant_columns",
            "low",
            "Near-constant features may provide little stable information",
            "Almost every observed value in these columns is the same.",
            "Review their predictive value and stability before retaining them.",
        ),
        (
            "identifier_candidates",
            "quality_identifier_candidates",
            "medium",
            "Identifier-like features require exclusion review",
            (
                "Identifiers often describe record identity rather than a "
                "generalizable relationship with the target."
            ),
            (
                "Exclude confirmed identifiers from model inputs while "
                "retaining them separately for traceability."
            ),
        ),
        (
            "infinite_value_columns",
            "quality_infinite_values",
            "high",
            "Infinite numerical values require treatment",
            (
                "Most estimators cannot train directly on positive or negative "
                "infinity."
            ),
            (
                "Trace the calculation that produced infinity and replace it "
                "using a documented domain-aware rule."
            ),
        ),
        (
            "unsupported_columns",
            "quality_unsupported_columns",
            "high",
            "Unsupported feature types were detected",
            (
                "These columns cannot be processed safely by the current "
                "tabular pipeline."
            ),
            (
                "Convert each column to a supported numerical, categorical, "
                "boolean, text, or datetime representation."
            ),
        ),
    ]:
        columns = _as_list(quality_flags.get(flag_key))
        if columns:
            add_insight(
                insight_id,
                severity,
                "data_quality",
                title,
                explanation,
                {"columns": columns},
                recommendation,
                columns,
            )

    target_statistics = _as_dict(
        eda_report.get("target_statistics")
    )
    if target_statistics:
        if not target_statistics.get("training_usable", False):
            add_insight(
                "target_not_training_usable",
                "critical",
                "target",
                "The selected target is not usable for training",
                (
                    "The target failed its minimum variability or value-validity "
                    "requirements."
                ),
                {
                    "problem_type": target_statistics.get(
                        "problem_type"
                    ),
                    "unique_observed_values": target_statistics.get(
                        "unique_observed_values"
                    ),
                    "missing_target_rows": target_statistics.get(
                        "missing_target_rows"
                    ),
                },
                (
                    "Select or repair a target with at least two usable values "
                    "before starting model training."
                ),
                [target_column],
            )

        missing_target_rows = (
            target_statistics.get("missing_target_rows", 0) or 0
        )
        if missing_target_rows > 0:
            add_insight(
                "target_missing_values",
                "high",
                "target",
                "Rows with missing target values cannot train a supervised model",
                (
                    "Supervised learning requires a known label or outcome for "
                    "every training row."
                ),
                {
                    "missing_target_rows": missing_target_rows,
                    "missing_target_percentage": target_statistics.get(
                        "missing_target_percentage"
                    ),
                },
                (
                    "Exclude these rows from model fitting while preserving them "
                    "separately if future labels may become available."
                ),
                [target_column],
            )

        classification = _as_dict(
            target_statistics.get("classification_statistics")
        )
        if classification:
            imbalance_level = str(
                classification.get("imbalance_level", "unknown")
            )
            if imbalance_level in {"mild", "moderate", "severe"}:
                imbalance_severity = {
                    "mild": "low",
                    "moderate": "medium",
                    "severe": "high",
                }[imbalance_level]
                add_insight(
                    "target_class_imbalance",
                    imbalance_severity,
                    "target",
                    f"The target has {imbalance_level} class imbalance",
                    (
                        "A model can appear accurate by favoring the majority "
                        "class while performing poorly on the minority class."
                    ),
                    {
                        "imbalance_level": imbalance_level,
                        "minority_class": classification.get(
                            "minority_class"
                        ),
                        "minority_class_percentage": classification.get(
                            "minority_class_percentage"
                        ),
                        "minority_to_majority_ratio": classification.get(
                            "minority_to_majority_ratio"
                        ),
                        "recommended_metrics": classification.get(
                            "recommended_metrics", []
                        ),
                    },
                    (
                        "Use stratified splitting, class-sensitive metrics, and "
                        "compare class weighting; apply resampling only inside "
                        "training folds if it is later justified."
                    ),
                    [target_column],
                )

            rare_classes = _as_list(
                classification.get("rare_classes")
            )
            if rare_classes:
                add_insight(
                    "target_rare_classes",
                    "high",
                    "target",
                    "Rare target classes may be difficult to learn reliably",
                    (
                        "Very small classes can produce unstable validation "
                        "scores and may prevent stratified cross-validation."
                    ),
                    {
                        "rare_classes": rare_classes,
                        "maximum_stratified_cv_folds": classification.get(
                            "maximum_stratified_cv_folds"
                        ),
                    },
                    (
                        "Confirm label quality, use feasible stratified folds, "
                        "and report per-class metrics with uncertainty."
                    ),
                    [target_column],
                )

        regression = _as_dict(
            target_statistics.get("regression_statistics")
        )
        if regression:
            distribution_shape = regression.get(
                "distribution_shape"
            )
            transformation = regression.get(
                "target_transformation_recommendation",
                regression.get(
                    "transformation_recommendation"
                ),
            )
            if distribution_shape in {
                "right_skewed",
                "left_skewed",
                "heavy_tailed",
            }:
                add_insight(
                    "target_regression_distribution",
                    "medium",
                    "target",
                    "The regression target has a non-symmetric distribution",
                    (
                        "Strong skewness or heavy tails can concentrate errors "
                        "in a small range of unusually large observations."
                    ),
                    {
                        "distribution_shape": distribution_shape,
                        "skewness": regression.get("skewness"),
                        "kurtosis": regression.get("kurtosis"),
                        "transformation_recommendation": transformation,
                    },
                    (
                        "Compare untransformed training with a justified target "
                        "transformation and evaluate errors on the original scale."
                    ),
                    [target_column],
                )

            target_outlier_percentage = (
                regression.get("outlier_percentage", 0) or 0
            )
            if target_outlier_percentage > 0:
                add_insight(
                    "target_regression_outliers",
                    "medium",
                    "target",
                    "The regression target contains IQR outliers",
                    (
                        "Extreme target values can dominate squared-error losses, "
                        "but they may also represent important valid cases."
                    ),
                    {
                        "outlier_count": regression.get("outlier_count"),
                        "outlier_percentage": target_outlier_percentage,
                    },
                    (
                        "Validate extreme outcomes, compare robust losses or "
                        "metrics, and do not remove them automatically."
                    ),
                    [target_column],
                )

    column_profiles = _as_list(
        overview.get("column_profiles")
    )
    semantic_types = {
        profile.get("column"): profile.get("semantic_type")
        for profile in column_profiles
        if isinstance(profile, dict)
        and profile.get("column") is not None
    }

    preprocessing = _as_dict(
        eda_report.get("preprocessing_context")
    )
    outlier_report = _as_dict(
        preprocessing.get("outlier_report")
    )
    authoritative_outlier_summary = _as_dict(
        outlier_report.get("outlier_summary")
    )
    outlier_guidance_available = bool(outlier_report)

    numerical_report = _as_dict(
        eda_report.get("numerical_statistics")
    )
    numerical_statistics = _as_dict(
        numerical_report.get("numerical_statistics")
    )
    configuration = _as_dict(
        eda_report.get("configuration")
    )
    high_outlier_percentage_threshold = (
        configuration.get(
            "high_outlier_ratio_threshold",
            0.05,
        )
        * 100
    )
    numerical_outliers = []
    skewed_numerical = []
    numerical_missing = []

    for column, statistics in numerical_statistics.items():
        statistics = _as_dict(statistics)
        semantic_type = semantic_types.get(column)
        continuous_numerical = semantic_type in {
            None,
            "continuous_numerical",
        }

        if (statistics.get("missing_values", 0) or 0) > 0:
            numerical_missing.append(column)

        if outlier_guidance_available:
            eligible_for_outlier_insight = (
                column in authoritative_outlier_summary
            )
        else:
            eligible_for_outlier_insight = continuous_numerical

        outlier_percentage = (
            statistics.get("outlier_percentage", 0) or 0
        )

        if (
            eligible_for_outlier_insight
            and outlier_percentage
            >= high_outlier_percentage_threshold
        ):
            numerical_outliers.append(
                {
                    "column": column,
                    "outlier_percentage": statistics.get(
                        "outlier_percentage"
                    ),
                    "recommended_scaler": statistics.get(
                        "recommended_scaler"
                    ),
                }
            )
        if (
            continuous_numerical
            and statistics.get("distribution_shape")
            in {
                "right_skewed",
                "left_skewed",
                "heavy_tailed",
            }
        ):
            skewed_numerical.append(
                {
                    "column": column,
                    "distribution_shape": statistics.get(
                        "distribution_shape"
                    ),
                    "skewness": statistics.get("skewness"),
                    "transformation_recommendation": statistics.get(
                        "transformation_recommendation"
                    ),
                }
            )

    if numerical_missing:
        add_insight(
            "numerical_missing_values",
            "medium",
            "numerical_distribution",
            "Numerical features require missing-value handling",
            (
                "Many estimators cannot directly accept missing numerical "
                "values, and the imputation strategy affects their distribution."
            ),
            {"columns": numerical_missing},
            (
                "Fit the configured numerical imputer using training data only "
                "and preserve missingness indicators when useful."
            ),
            numerical_missing,
        )

    if numerical_outliers:
        columns = [item["column"] for item in numerical_outliers]
        add_insight(
            "numerical_feature_outliers",
            "medium",
            "numerical_distribution",
            "Numerical feature outliers require robust preprocessing review",
            (
                "IQR outliers can distort means and standard scaling but are "
                "not automatically invalid observations."
            ),
            {"features": numerical_outliers},
            (
                "Validate extreme values and use robust scaling where "
                "recommended instead of deleting rows automatically."
            ),
            columns,
        )

    if skewed_numerical:
        columns = [item["column"] for item in skewed_numerical]
        add_insight(
            "numerical_skewed_features",
            "low",
            "numerical_distribution",
            "Some numerical features have strongly asymmetric distributions",
            (
                "Skewed features may benefit selected linear or distance-based "
                "models from reviewed transformations."
            ),
            {"features": skewed_numerical},
            (
                "Compare the original representation with the recommended "
                "transformation inside cross-validation."
            ),
            columns,
        )

    categorical_report = _as_dict(
        eda_report.get("categorical_statistics")
    )
    categorical_statistics = _as_dict(
        categorical_report.get("categorical_statistics")
    )
    high_cardinality = []
    rare_categories = []
    categorical_missing = []
    dominant_categories = []

    for column, statistics in categorical_statistics.items():
        statistics = _as_dict(statistics)
        if statistics.get("high_cardinality", False):
            high_cardinality.append(
                {
                    "column": column,
                    "unique_values": statistics.get("unique_values"),
                    "recommended_encoding": statistics.get(
                        "recommended_encoding"
                    ),
                }
            )
        if (statistics.get("rare_category_count", 0) or 0) > 0:
            rare_categories.append(
                {
                    "column": column,
                    "rare_category_count": statistics.get(
                        "rare_category_count"
                    ),
                    "rare_observation_percentage": statistics.get(
                        "rare_observation_percentage"
                    ),
                }
            )
        if (statistics.get("missing_values", 0) or 0) > 0:
            categorical_missing.append(column)
        if statistics.get("dominant_category", False):
            dominant_categories.append(
                {
                    "column": column,
                    "most_frequent_percentage": statistics.get(
                        "most_frequent_percentage"
                    ),
                }
            )

    if high_cardinality:
        columns = [item["column"] for item in high_cardinality]
        add_insight(
            "categorical_high_cardinality",
            "medium",
            "categorical_distribution",
            "High-cardinality categorical features need controlled encoding",
            (
                "Naive one-hot encoding can create a very wide sparse matrix "
                "and overfit rare labels."
            ),
            {"features": high_cardinality},
            (
                "Review frequency, hashing, grouped-category, or leakage-safe "
                "target encoding based on model and domain requirements."
            ),
            columns,
        )

    if categorical_missing:
        add_insight(
            "categorical_missing_values",
            "medium",
            "categorical_distribution",
            "Categorical features require missing-value handling",
            (
                "Missing categories may represent unavailable data or a "
                "meaningful state and should be handled consistently."
            ),
            {"columns": categorical_missing},
            (
                "Fit categorical imputation on training data only and decide "
                "whether an explicit missing category has domain meaning."
            ),
            categorical_missing,
        )

    if rare_categories:
        columns = [item["column"] for item in rare_categories]
        add_insight(
            "categorical_rare_categories",
            "low",
            "categorical_distribution",
            "Rare categorical levels may create unstable estimates",
            (
                "Categories with very few observations can overfit and may be "
                "absent from individual validation folds."
            ),
            {"features": rare_categories},
            (
                "Review grouping infrequent levels into an explicit rare class "
                "using thresholds learned from training data only."
            ),
            columns,
        )

    if dominant_categories:
        columns = [item["column"] for item in dominant_categories]
        add_insight(
            "categorical_dominant_categories",
            "low",
            "categorical_distribution",
            "Some categorical features are dominated by one value",
            (
                "A highly dominant category leaves little variation for the "
                "model to learn from."
            ),
            {"features": dominant_categories},
            "Verify that these features provide stable predictive value.",
            columns,
        )

    datetime_report = _as_dict(
        eda_report.get("datetime_statistics")
    )
    datetime_statistics = _as_dict(
        datetime_report.get("datetime_statistics")
    )
    invalid_datetime = []

    for column, statistics in datetime_statistics.items():
        statistics = _as_dict(statistics)
        if (statistics.get("invalid_datetime_values", 0) or 0) > 0:
            invalid_datetime.append(
                {
                    "column": column,
                    "invalid_datetime_values": statistics.get(
                        "invalid_datetime_values"
                    ),
                    "invalid_datetime_percentage": statistics.get(
                        "invalid_datetime_percentage"
                    ),
                }
            )

    datetime_columns = _as_list(
        datetime_report.get("datetime_columns")
    )
    if invalid_datetime:
        columns = [item["column"] for item in invalid_datetime]
        add_insight(
            "datetime_invalid_values",
            "medium",
            "temporal",
            "Some datetime values could not be parsed",
            (
                "Invalid timestamps can silently become missing values and "
                "distort temporal features."
            ),
            {"features": invalid_datetime},
            (
                "Standardize date formats, validate timezone assumptions, and "
                "document the handling of invalid timestamps."
            ),
            columns,
        )

    if datetime_columns:
        add_insight(
            "datetime_split_review",
            "medium",
            "temporal",
            "Datetime features require temporal leakage review",
            (
                "Random splitting can let future observations influence past "
                "predictions when rows represent a time sequence."
            ),
            {"datetime_columns": datetime_columns},
            (
                "Confirm the prediction-time boundary and use chronological or "
                "group-aware validation when the use case is temporal."
            ),
            datetime_columns,
        )

    feature_target = _as_dict(
        eda_report.get("feature_target_relationships")
    )
    leakage_features = _as_list(
        feature_target.get("potential_leakage_features")
    )
    if leakage_features:
        add_insight(
            "relationship_potential_target_leakage",
            "critical",
            "leakage",
            "Feature-target relationships indicate potential leakage",
            (
                "Near-deterministic association can mean a feature contains the "
                "target or information unavailable at prediction time."
            ),
            {"potential_leakage_features": leakage_features},
            (
                "Investigate feature provenance and availability at inference "
                "time before allowing these columns into training."
            ),
            leakage_features,
        )

    ranked_features = _as_list(
        feature_target.get("ranked_features")
    )
    strong_features = [
        item
        for item in ranked_features
        if isinstance(item, dict)
        and item.get("relationship_strength") == "strong"
        and not item.get("potential_leakage", False)
    ]
    if strong_features:
        columns = [item.get("feature") for item in strong_features]
        columns = [column for column in columns if column is not None]
        add_insight(
            "relationship_strong_target_associations",
            "info",
            "relationship",
            "Some features have strong univariate target associations",
            (
                "These variables may be useful predictors, although a strong "
                "univariate relationship does not prove causal or multivariate "
                "importance."
            ),
            {"features": strong_features},
            (
                "Retain them as model candidates, verify leakage and stability, "
                "and confirm importance using held-out evaluation."
            ),
            columns,
        )

    skipped_relationship_features = _as_list(
        feature_target.get("skipped_features")
    )
    if skipped_relationship_features:
        add_insight(
            "relationship_skipped_features",
            "low",
            "relationship",
            "Some feature-target relationships were not analyzed",
            (
                "Safety limits or insufficient usable samples prevented a "
                "reliable univariate association calculation."
            ),
            {"skipped_features": skipped_relationship_features},
            (
                "Review the per-feature skip reasons before interpreting the "
                "relationship ranking as complete."
            ),
            skipped_relationship_features,
        )

    feature_feature = _as_dict(
        eda_report.get("feature_feature_relationships")
    )
    high_association_pairs = _as_list(
        feature_feature.get("high_association_pairs")
    )
    if high_association_pairs:
        affected = []
        for pair in high_association_pairs:
            if isinstance(pair, dict):
                affected.extend(
                    [pair.get("feature_1"), pair.get("feature_2")]
                )
        affected = [item for item in affected if item is not None]
        add_insight(
            "relationship_redundant_features",
            "medium",
            "multicollinearity",
            "Highly associated feature pairs may be redundant",
            (
                "Strong feature-feature association can destabilize linear "
                "coefficients and divide importance across equivalent inputs."
            ),
            {
                "high_association_pairs": high_association_pairs,
                "redundancy_groups": feature_feature.get(
                    "redundancy_groups", []
                ),
            },
            (
                "Review semantic redundancy and compare retaining, combining, "
                "or regularizing correlated features inside validation."
            ),
            affected,
        )

    vif_report = _as_dict(
        feature_feature.get("multicollinearity_report")
    )
    high_vif_features = _as_list(
        vif_report.get("high_vif_features")
    )
    if high_vif_features:
        add_insight(
            "relationship_high_vif",
            "high",
            "multicollinearity",
            "High variance inflation was detected",
            (
                "A high VIF means a numerical feature can be predicted closely "
                "from other numerical features, which can destabilize linear "
                "model coefficients."
            ),
            {
                "high_vif_features": high_vif_features,
                "vif_statistics": vif_report.get("vif_statistics", []),
            },
            (
                "Inspect redundant variables and compare regularization or "
                "feature reduction without removing columns automatically."
            ),
            high_vif_features,
        )

    cleaning_plan = _as_dict(
        preprocessing.get("cleaning_plan")
    )
    leakage_report = _as_dict(
        preprocessing.get("leakage_report")
    )
    statistical_leakage = _as_list(
        leakage_report.get("potential_leakage_columns")
    )
    uncovered_leakage = [
        column
        for column in statistical_leakage
        if column not in leakage_features
    ]
    if statistical_leakage:
        add_insight(
            "preprocessing_statistical_leakage",
            "critical",
            "leakage",
            "Preprocessing checks detected potential target leakage",
            (
                "These relationships may expose the target directly or encode "
                "information that would not exist when predictions are made."
            ),
            {
                "potential_leakage_columns": statistical_leakage,
                "additional_columns_not_flagged_by_relationship_eda": (
                    uncovered_leakage
                ),
                "leakage_evidence": leakage_report.get(
                    "leakage_evidence", {}
                ),
            },
            (
                "Block model preparation until feature provenance has been "
                "reviewed or an explicit override has been justified."
            ),
            statistical_leakage,
        )

    drop_columns = _as_list(cleaning_plan.get("drop_columns"))
    if drop_columns:
        add_insight(
            "preprocessing_planned_feature_drops",
            "info",
            "preprocessing",
            "The cleaning plan proposes excluding some features",
            (
                "The proposal is based on documented quality rules and has not "
                "modified the source dataset."
            ),
            {
                "drop_columns": drop_columns,
                "drop_reasons": cleaning_plan.get("drop_reasons", {}),
            },
            (
                "Confirm every reason before using the plan in a fitted "
                "preprocessing pipeline."
            ),
            drop_columns,
        )

    imputation_plan = _as_dict(
        cleaning_plan.get("imputation_plan")
    )
    if imputation_plan:
        add_insight(
            "preprocessing_imputation_plan",
            "info",
            "preprocessing",
            "Missing-value imputation is required before modeling",
            (
                "The plan assigns deterministic imputation strategies based on "
                "feature type."
            ),
            {"imputation_plan": imputation_plan},
            (
                "Fit every imputer on training folds only and reuse the fitted "
                "pipeline for validation and inference."
            ),
            list(imputation_plan.keys()),
        )

    review_columns = _as_dict(cleaning_plan.get("review_columns"))
    if review_columns:
        add_insight(
            "preprocessing_manual_review_columns",
            "medium",
            "preprocessing",
            "Some features require manual preprocessing review",
            (
                "The current automatic pipeline cannot choose a safe encoding "
                "or transformation without additional context."
            ),
            {"review_columns": review_columns},
            (
                "Document a feature-specific handling decision before these "
                "columns are included in model training."
            ),
            list(review_columns.keys()),
        )

    robust_features = _as_list(
        outlier_report.get("robust_scaler_features")
    )
    if robust_features:
        add_insight(
            "preprocessing_robust_scaling",
            "info",
            "preprocessing",
            "Robust scaling is recommended for outlier-sensitive features",
            (
                "Median and interquartile-range scaling reduces the influence of "
                "extreme values without deleting observations."
            ),
            {"robust_scaler_features": robust_features},
            (
                "Apply robust scaling inside the fitted preprocessing pipeline "
                "and compare model performance through cross-validation."
            ),
            robust_features,
        )

    if not insights:
        add_insight(
            "overview_no_automatic_concerns",
            "info",
            "overview",
            "No configured EDA warning condition was triggered",
            (
                "The available automated checks did not detect a predefined "
                "quality, target, leakage, or relationship concern."
            ),
            {"report_status": report_status},
            (
                "Continue with validation while still performing semantic, "
                "privacy, fairness, and domain review."
            ),
            requires_human_review=True,
        )

    insights.sort(
        key=lambda item: (
            _SEVERITY_ORDER[item["severity"]],
            item["_sequence"],
        )
    )

    total_generated = len(insights)
    prioritized_insights = insights[:max_insights]

    for priority_rank, insight in enumerate(
        prioritized_insights,
        start=1,
    ):
        insight["priority_rank"] = priority_rank
        insight.pop("_sequence", None)

    returned_severity_counts = Counter(
        item["severity"] for item in prioritized_insights
    )
    returned_category_counts = Counter(
        item["category"] for item in prioritized_insights
    )

    severity_counts = {
        severity: int(returned_severity_counts.get(severity, 0))
        for severity in _SEVERITY_ORDER
    }
    category_counts = {
        category: int(count)
        for category, count in sorted(
            returned_category_counts.items()
        )
    }

    critical_count = severity_counts["critical"]
    high_count = severity_counts["high"]
    medium_count = severity_counts["medium"]

    if critical_count > 0:
        executive_summary = (
            f"EDA generated {len(prioritized_insights)} prioritized "
            f"insights, including {critical_count} critical finding(s). "
            "Resolve critical findings before model training."
        )
    elif high_count > 0:
        executive_summary = (
            f"EDA generated {len(prioritized_insights)} prioritized "
            f"insights, including {high_count} high-severity finding(s). "
            "Review them before finalizing the modeling pipeline."
        )
    elif medium_count > 0:
        executive_summary = (
            f"EDA generated {len(prioritized_insights)} prioritized "
            "insights. No critical or high-severity finding was detected, "
            "but preprocessing decisions still require review."
        )
    else:
        executive_summary = (
            f"EDA generated {len(prioritized_insights)} prioritized "
            "insights with no configured critical, high, or medium finding."
        )

    next_actions = _unique_text(
        [
            insight["recommendation"]
            for insight in prioritized_insights
            if insight["severity"]
            in {"critical", "high", "medium"}
        ]
    )

    result = {
        "insight_report_type": "automated_eda_insights",
        "insight_report_version": INSIGHT_REPORT_VERSION,
        "source_eda_report_version": eda_report.get(
            "report_version"
        ),
        "target_column": target_column,
        "source_report_status": report_status,
        "generated_insight_count": total_generated,
        "insight_count": len(prioritized_insights),
        "insights_truncated": total_generated > max_insights,
        "max_insights": max_insights,
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "executive_summary": executive_summary,
        "next_actions": next_actions,
        "prioritized_insights": prioritized_insights,
        "limitations": [
            (
                "Insights are deterministic interpretations of configured "
                "statistical rules and do not replace domain expertise."
            ),
            (
                "Severity represents modeling risk or review urgency, not "
                "business impact."
            ),
            (
                "Automated association checks do not establish causation, "
                "fairness, privacy compliance, or deployment suitability."
            ),
        ],
        "automatic_data_modification_performed": False,
    }

    return _json_safe(result)