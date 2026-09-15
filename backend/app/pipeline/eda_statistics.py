import math
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.eda_config import EDAConfig


def _safe_number(
    value: Any,
    decimal_places: int,
) -> float | None:
    """
    Convert Pandas/NumPy values into finite JSON-safe floats.
    """

    if value is None or pd.isna(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(number, decimal_places)


def generate_numerical_statistics(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
) -> dict:
    """
    Generate generalized univariate statistics for numerical
    candidate features without modifying the DataFrame.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()

    numerical_columns = [
        column
        for column in df.columns
        if (
            column != target_column
            and pd.api.types.is_numeric_dtype(
                df[column].dtype
            )
            and not pd.api.types.is_bool_dtype(
                df[column].dtype
            )
        )
    ]

    numerical_statistics = {}

    for column in numerical_columns:
        series = df[column]

        missing_count = int(
            series.isnull().sum()
        )

        infinite_count = int(
            series.isin(
                [np.inf, -np.inf]
            ).sum()
        )

        finite_values = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        finite_count = len(finite_values)

        unique_count = int(
            finite_values.nunique()
        )

        unique_ratio = (
            unique_count / finite_count
            if finite_count > 0
            else 0
        )

        mean = (
            finite_values.mean()
            if finite_count > 0
            else None
        )

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

        minimum = (
            finite_values.min()
            if finite_count > 0
            else None
        )

        maximum = (
            finite_values.max()
            if finite_count > 0
            else None
        )

        value_range = (
            maximum - minimum
            if finite_count > 0
            else None
        )

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

        zero_count = int(
            (finite_values == 0).sum()
        )

        negative_count = int(
            (finite_values < 0).sum()
        )

        zero_percentage = (
            zero_count / finite_count
            if finite_count > 0
            else 0
        )

        negative_percentage = (
            negative_count / finite_count
            if finite_count > 0
            else 0
        )

        value_counts = (
            finite_values.value_counts()
        )

        top_frequency = (
            int(value_counts.iloc[0])
            if not value_counts.empty
            else 0
        )

        mode_value = (
            value_counts.index[0]
            if top_frequency > 1
            else None
        )

        top_value_ratio = (
            top_frequency / finite_count
            if finite_count > 0
            else 0
        )

        near_constant = (
            top_value_ratio
            >= config.near_constant_threshold
        )

        outlier_count = 0
        outlier_percentage = 0.0
        lower_outlier_bound = None
        upper_outlier_bound = None

        if (
            finite_count > 0
            and iqr is not None
            and iqr > 0
        ):
            lower_outlier_bound = (
                first_quartile
                - config.outlier_iqr_multiplier
                * iqr
            )

            upper_outlier_bound = (
                third_quartile
                + config.outlier_iqr_multiplier
                * iqr
            )

            outlier_mask = (
                finite_values
                < lower_outlier_bound
            ) | (
                finite_values
                > upper_outlier_bound
            )

            outlier_count = int(
                outlier_mask.sum()
            )

            outlier_percentage = (
                outlier_count / finite_count
            )

        recommended_scaler = (
            "robust"
            if outlier_percentage
            >= config.high_outlier_ratio_threshold
            else "standard"
        )

        safe_skewness = _safe_number(
            skewness,
            config.decimal_places,
        )

        if safe_skewness is None:
            distribution_shape = "undetermined"

        elif (
            safe_skewness
            >= config.strong_skewness_threshold
        ):
            distribution_shape = "right_skewed"

        elif (
            safe_skewness
            <= -config.strong_skewness_threshold
        ):
            distribution_shape = "left_skewed"

        else:
            distribution_shape = (
                "approximately_symmetric"
            )

        transformation_recommendation = None

        if distribution_shape == "right_skewed":
            if (
                minimum is not None
                and minimum >= 0
            ):
                transformation_recommendation = (
                    "log1p_review"
                )
            else:
                transformation_recommendation = (
                    "power_transform_review"
                )

        elif distribution_shape == "left_skewed":
            transformation_recommendation = (
                "power_transform_review"
            )

        numerical_statistics[column] = {
            "data_type": str(series.dtype),
            "total_rows": len(df),
            "finite_count": finite_count,
            "missing_values": missing_count,
            "missing_percentage": round(
                missing_count / len(df) * 100,
                2,
            ),
            "infinite_values": infinite_count,
            "unique_values": unique_count,
            "unique_ratio": round(
                unique_ratio,
                config.decimal_places,
            ),
            "mean": _safe_number(
                mean,
                config.decimal_places,
            ),
            "median": _safe_number(
                median,
                config.decimal_places,
            ),
            "mode": _safe_number(
                mode_value,
                config.decimal_places,
            ),
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
            "ninety_fifth_percentile": (
                _safe_number(
                    finite_values.quantile(0.95)
                    if finite_count > 0
                    else None,
                    config.decimal_places,
                )
            ),
            "ninety_ninth_percentile": (
                _safe_number(
                    finite_values.quantile(0.99)
                    if finite_count > 0
                    else None,
                    config.decimal_places,
                )
            ),
            "maximum": _safe_number(
                maximum,
                config.decimal_places,
            ),
            "range": _safe_number(
                value_range,
                config.decimal_places,
            ),
            "iqr": _safe_number(
                iqr,
                config.decimal_places,
            ),
            "skewness": safe_skewness,
            "kurtosis": _safe_number(
                kurtosis,
                config.decimal_places,
            ),
            "zero_count": zero_count,
            "zero_percentage": round(
                zero_percentage * 100,
                2,
            ),
            "negative_count": negative_count,
            "negative_percentage": round(
                negative_percentage * 100,
                2,
            ),
            "near_constant": near_constant,
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
                outlier_percentage * 100,
                2,
            ),
            "distribution_shape": (
                distribution_shape
            ),
            "recommended_scaler": (
                recommended_scaler
            ),
            "transformation_recommendation": (
                transformation_recommendation
            ),
        }

    return {
        "target_column": target_column,
        "numerical_columns": numerical_columns,
        "numerical_feature_count": len(
            numerical_columns
        ),
        "numerical_statistics": (
            numerical_statistics
        ),
    }


def _safe_value(value: Any) -> Any:
    """
    Convert Pandas/NumPy categorical values into JSON-safe
    Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def _parse_datetime_values(
    values: pd.Series,
) -> pd.Series:
    """
    Parse values as datetimes and normalize timezone-aware
    values to UTC for consistent comparison and JSON output.

    ``format="mixed"`` supports columns containing multiple
    common datetime formats. The fallback keeps compatibility
    with older supported Pandas versions.
    """

    try:
        return pd.to_datetime(
            values,
            errors="coerce",
            format="mixed",
            utc=True,
        )
    except (TypeError, ValueError):
        return pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )


def _datetime_text_hint_mask(
    values: pd.Series,
) -> pd.Series:
    """
    Identify text that structurally resembles a date or time.

    This conservative check prevents ordinary labels, IDs, and
    free-text columns from being classified as datetimes merely
    because a parser can interpret a few values.
    """

    text_values = values.astype("string").str.strip()

    separated_or_named_date = text_values.str.contains(
        (
            r"(?:\d{1,4}[-/.]\d{1,2})"
            r"|(?:\d{1,2}[-/.]\d{1,4})"
            r"|(?:[Tt]\d{1,2}:\d{2})"
            r"|(?:\d{1,2}:\d{2})"
            r"|(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|"
            r"sep|sept|oct|nov|dec)[a-z]*\b)"
        ),
        case=False,
        regex=True,
        na=False,
    )

    compact_calendar_value = text_values.str.fullmatch(
        r"(?:19|20)\d{2}(?:\d{2}){0,2}",
        na=False,
    )

    return separated_or_named_date | compact_calendar_value


def _detect_datetime_columns(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig,
) -> dict[str, dict[str, Any]]:
    """
    Detect native and conservatively inferred datetime features.

    Only a bounded, deterministic sample is used for inference.
    Once a column passes inference, the complete column is parsed
    so its final report reflects all rows.
    """

    detected_columns: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(
            series.dtype
        ):
            parsed_series = _parse_datetime_values(series)
            observed_count = int(series.notna().sum())
            parsed_count = int(parsed_series.notna().sum())

            detected_columns[column] = {
                "parsed_series": parsed_series,
                "detection_method": "native_datetime_dtype",
                "inference_sample_size": observed_count,
                "sample_parse_success_ratio": (
                    parsed_count / observed_count
                    if observed_count > 0
                    else 0.0
                ),
                "sample_datetime_hint_ratio": 1.0,
            }
            continue

        supported_text_dtype = (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
        )

        if not supported_text_dtype:
            continue

        observed_values = series.dropna()

        if observed_values.empty:
            continue

        sample_size = min(
            len(observed_values),
            config.datetime_inference_sample_size,
        )

        if len(observed_values) > sample_size:
            sample_values = observed_values.sample(
                n=sample_size,
                random_state=config.random_state,
            )
        else:
            sample_values = observed_values

        hint_mask = _datetime_text_hint_mask(sample_values)
        hint_ratio = float(hint_mask.mean())

        if hint_ratio < config.datetime_parse_threshold:
            continue

        parsed_sample = _parse_datetime_values(sample_values)
        sample_parse_ratio = float(
            parsed_sample.notna().mean()
        )

        if (
            sample_parse_ratio
            < config.datetime_parse_threshold
        ):
            continue

        parsed_series = _parse_datetime_values(series)

        detected_columns[column] = {
            "parsed_series": parsed_series,
            "detection_method": "inferred_from_text",
            "inference_sample_size": sample_size,
            "sample_parse_success_ratio": (
                sample_parse_ratio
            ),
            "sample_datetime_hint_ratio": hint_ratio,
        }

    return detected_columns


def generate_categorical_statistics(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
) -> dict:
    """
    Generate generalized univariate statistics for categorical
    and Boolean candidate features.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()

    datetime_columns = set(
        _detect_datetime_columns(
            df,
            target_column,
            config,
        )
    )

    categorical_columns = [
        column
        for column in df.columns
        if (
            column != target_column
            and column not in datetime_columns
            and (
                pd.api.types.is_bool_dtype(
                    df[column].dtype
                )
                or pd.api.types.is_string_dtype(
                    df[column].dtype
                )
                or isinstance(
                    df[column].dtype,
                    pd.CategoricalDtype,
                )
            )
        )
    ]

    categorical_statistics = {}

    for column in categorical_columns:
        series = df[column]

        missing_count = int(
            series.isnull().sum()
        )

        observed_values = series.dropna()
        observed_count = len(observed_values)

        value_counts = observed_values.value_counts(
            dropna=False
        )

        unique_count = len(value_counts)

        unique_ratio = (
            unique_count / observed_count
            if observed_count > 0
            else 0
        )

        high_cardinality = (
            unique_count
            > config.high_cardinality_count_threshold
            and unique_ratio
            > config.high_cardinality_ratio_threshold
        )

        if unique_count == 0:
            cardinality_level = "empty"

        elif unique_count <= 2:
            cardinality_level = "binary"

        elif unique_count <= 10:
            cardinality_level = "low"

        elif high_cardinality:
            cardinality_level = "high"

        else:
            cardinality_level = "moderate"

        top_value = (
            value_counts.index[0]
            if not value_counts.empty
            else None
        )

        top_frequency = (
            int(value_counts.iloc[0])
            if not value_counts.empty
            else 0
        )

        top_percentage_ratio = (
            top_frequency / observed_count
            if observed_count > 0
            else 0
        )

        dominant_category = (
            top_percentage_ratio
            >= config.dominant_category_threshold
        )

        near_constant = (
            top_percentage_ratio
            >= config.near_constant_threshold
        )

        category_ratios = (
            value_counts / observed_count
            if observed_count > 0
            else value_counts
        )

        rare_category_mask = (
            category_ratios
            < config.rare_category_threshold
        )

        rare_category_count = int(
            rare_category_mask.sum()
        )

        rare_observation_count = int(
            value_counts[
                rare_category_mask
            ].sum()
        )

        rare_observation_ratio = (
            rare_observation_count
            / observed_count
            if observed_count > 0
            else 0
        )

        displayed_counts = value_counts.head(
            config.max_categories_displayed
        )

        top_categories = []

        for value, count in displayed_counts.items():
            top_categories.append(
                {
                    "value": _safe_value(value),
                    "count": int(count),
                    "percentage": round(
                        int(count)
                        / observed_count
                        * 100,
                        2,
                    )
                    if observed_count > 0
                    else 0,
                }
            )

        displayed_observation_count = int(
            displayed_counts.sum()
        )

        other_observation_count = (
            observed_count
            - displayed_observation_count
        )

        categories_truncated = (
            unique_count
            > config.max_categories_displayed
        )

        if unique_count == 0:
            recommended_encoding = None

        elif high_cardinality:
            recommended_encoding = (
                "frequency_or_hashing_review"
            )

        else:
            recommended_encoding = "one_hot"

        categorical_statistics[column] = {
            "data_type": str(series.dtype),
            "total_rows": len(df),
            "observed_count": observed_count,
            "missing_values": missing_count,
            "missing_percentage": round(
                missing_count / len(df) * 100,
                2,
            ),
            "unique_values": unique_count,
            "unique_ratio": round(
                unique_ratio,
                config.decimal_places,
            ),
            "cardinality_level": (
                cardinality_level
            ),
            "high_cardinality": high_cardinality,
            "most_frequent_value": _safe_value(
                top_value
            ),
            "most_frequent_count": top_frequency,
            "most_frequent_percentage": round(
                top_percentage_ratio * 100,
                2,
            ),
            "dominant_category": (
                dominant_category
            ),
            "near_constant": near_constant,
            "rare_category_count": (
                rare_category_count
            ),
            "rare_observation_count": (
                rare_observation_count
            ),
            "rare_observation_percentage": round(
                rare_observation_ratio * 100,
                2,
            ),
            "top_categories": top_categories,
            "categories_truncated": (
                categories_truncated
            ),
            "other_observation_count": (
                other_observation_count
            ),
            "recommended_encoding": (
                recommended_encoding
            ),
            "missing_category_review": (
                missing_count > 0
            ),
        }

    return {
        "target_column": target_column,
        "categorical_columns": (
            categorical_columns
        ),
        "categorical_feature_count": len(
            categorical_columns
        ),
        "categorical_statistics": (
            categorical_statistics
        ),
    }


def generate_datetime_statistics(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
) -> dict:
    """
    Detect datetime features and generate JSON-safe temporal
    descriptive statistics without modifying the DataFrame.

    Native datetime dtypes are always included. Text columns are
    included only when both their structural datetime-hint ratio
    and parse-success ratio meet the configured threshold.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()

    detected_columns = _detect_datetime_columns(
        df,
        target_column,
        config,
    )

    datetime_statistics = {}
    native_datetime_columns = []
    inferred_datetime_columns = []

    for column, detection in detected_columns.items():
        original_series = df[column]
        parsed_series = detection["parsed_series"]
        valid_values = parsed_series.dropna()

        total_rows = len(df)
        observed_count = int(
            original_series.notna().sum()
        )
        parsed_count = int(valid_values.size)
        missing_count = int(
            original_series.isna().sum()
        )
        invalid_count = max(
            observed_count - parsed_count,
            0,
        )

        parse_success_ratio = (
            parsed_count / observed_count
            if observed_count > 0
            else 0.0
        )

        invalid_ratio = (
            invalid_count / observed_count
            if observed_count > 0
            else 0.0
        )

        unique_datetime_count = int(
            valid_values.nunique()
        )

        duplicate_datetime_count = max(
            parsed_count - unique_datetime_count,
            0,
        )

        duplicate_datetime_ratio = (
            duplicate_datetime_count / parsed_count
            if parsed_count > 0
            else 0.0
        )

        earliest = (
            valid_values.min()
            if parsed_count > 0
            else None
        )

        latest = (
            valid_values.max()
            if parsed_count > 0
            else None
        )

        time_span_days = (
            (latest - earliest).total_seconds()
            / 86_400
            if earliest is not None
            and latest is not None
            else None
        )

        is_monotonic_increasing = (
            bool(valid_values.is_monotonic_increasing)
            if parsed_count > 1
            else None
        )

        is_monotonic_decreasing = (
            bool(valid_values.is_monotonic_decreasing)
            if parsed_count > 1
            else None
        )

        contains_time_component = False
        component_unique_values = {
            "year": 0,
            "quarter": 0,
            "month": 0,
            "day_of_month": 0,
            "day_of_week": 0,
            "hour": 0,
            "minute": 0,
        }

        if parsed_count > 0:
            contains_time_component = bool(
                (
                    (valid_values.dt.hour != 0)
                    | (valid_values.dt.minute != 0)
                    | (valid_values.dt.second != 0)
                    | (valid_values.dt.microsecond != 0)
                ).any()
            )

            component_unique_values = {
                "year": int(
                    valid_values.dt.year.nunique()
                ),
                "quarter": int(
                    valid_values.dt.quarter.nunique()
                ),
                "month": int(
                    valid_values.dt.month.nunique()
                ),
                "day_of_month": int(
                    valid_values.dt.day.nunique()
                ),
                "day_of_week": int(
                    valid_values.dt.dayofweek.nunique()
                ),
                "hour": int(
                    valid_values.dt.hour.nunique()
                ),
                "minute": int(
                    valid_values.dt.minute.nunique()
                ),
            }

        recommended_components = [
            component
            for component, unique_count in (
                component_unique_values.items()
            )
            if unique_count > 1
        ]

        if unique_datetime_count > 1:
            recommended_components.append(
                "elapsed_time_from_earliest"
            )

        sorted_unique_values = pd.Series(
            valid_values.unique()
        ).sort_values()

        gaps_in_days = (
            sorted_unique_values.diff()
            .dropna()
            .dt.total_seconds()
            / 86_400
        )

        minimum_gap_days = (
            gaps_in_days.min()
            if not gaps_in_days.empty
            else None
        )

        median_gap_days = (
            gaps_in_days.median()
            if not gaps_in_days.empty
            else None
        )

        maximum_gap_days = (
            gaps_in_days.max()
            if not gaps_in_days.empty
            else None
        )

        regular_interval = (
            bool(
                gaps_in_days.round(
                    config.decimal_places
                ).nunique()
                == 1
            )
            if len(gaps_in_days) >= 2
            else None
        )

        detection_method = detection[
            "detection_method"
        ]

        if detection_method == "native_datetime_dtype":
            native_datetime_columns.append(column)
        else:
            inferred_datetime_columns.append(column)

        datetime_statistics[column] = {
            "data_type": str(original_series.dtype),
            "detection_method": detection_method,
            "total_rows": total_rows,
            "observed_count": observed_count,
            "missing_values": missing_count,
            "missing_percentage": round(
                missing_count / total_rows * 100,
                2,
            ),
            "parsed_values": parsed_count,
            "parse_success_percentage": round(
                parse_success_ratio * 100,
                2,
            ),
            "invalid_datetime_values": invalid_count,
            "invalid_datetime_percentage": round(
                invalid_ratio * 100,
                2,
            ),
            "inference_sample_size": detection[
                "inference_sample_size"
            ],
            "sample_parse_success_percentage": round(
                detection[
                    "sample_parse_success_ratio"
                ]
                * 100,
                2,
            ),
            "sample_datetime_hint_percentage": round(
                detection[
                    "sample_datetime_hint_ratio"
                ]
                * 100,
                2,
            ),
            "unique_datetimes": unique_datetime_count,
            "duplicate_datetime_count": (
                duplicate_datetime_count
            ),
            "duplicate_datetime_percentage": round(
                duplicate_datetime_ratio * 100,
                2,
            ),
            "earliest_datetime": (
                earliest.isoformat()
                if earliest is not None
                else None
            ),
            "latest_datetime": (
                latest.isoformat()
                if latest is not None
                else None
            ),
            "time_span_days": _safe_number(
                time_span_days,
                config.decimal_places,
            ),
            "is_monotonic_increasing": (
                is_monotonic_increasing
            ),
            "is_monotonic_decreasing": (
                is_monotonic_decreasing
            ),
            "contains_time_component": (
                contains_time_component
            ),
            "component_unique_values": (
                component_unique_values
            ),
            "minimum_gap_days": _safe_number(
                minimum_gap_days,
                config.decimal_places,
            ),
            "median_gap_days": _safe_number(
                median_gap_days,
                config.decimal_places,
            ),
            "maximum_gap_days": _safe_number(
                maximum_gap_days,
                config.decimal_places,
            ),
            "regular_interval": regular_interval,
            "timezone_handling": (
                "normalized_to_utc_for_analysis"
            ),
            "recommended_datetime_features": (
                recommended_components
            ),
        }

    datetime_columns = list(detected_columns)

    return {
        "target_column": target_column,
        "datetime_parse_threshold": (
            config.datetime_parse_threshold
        ),
        "datetime_columns": datetime_columns,
        "native_datetime_columns": (
            native_datetime_columns
        ),
        "inferred_datetime_columns": (
            inferred_datetime_columns
        ),
        "datetime_feature_count": len(
            datetime_columns
        ),
        "datetime_statistics": datetime_statistics,
        "limitations": [
            (
                "Text datetime inference is conservative and may "
                "not recognize numeric spreadsheet serial dates."
            ),
            (
                "Ambiguous text dates are interpreted using Pandas "
                "parsing rules and should be reviewed."
            ),
            (
                "All successfully parsed values are normalized to "
                "UTC for consistent analysis."
            ),
        ],
    }