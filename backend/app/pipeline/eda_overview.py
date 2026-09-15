import re
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda_statistics import (
    generate_datetime_statistics,
)


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


def _safe_nunique(series: pd.Series) -> int:
    """Count unique values, including unhashable object values."""

    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().map(str).nunique())


def _top_frequency(series: pd.Series) -> int:
    """Return the largest observed value frequency safely."""

    observed = series.dropna()

    if observed.empty:
        return 0

    try:
        counts = observed.value_counts()
    except TypeError:
        counts = observed.map(str).value_counts()

    return int(counts.iloc[0]) if not counts.empty else 0


def _identifier_evidence(
    column_name: Any,
    series: pd.Series,
) -> list[str]:
    """Return conservative identifier-candidate evidence."""

    normalized_name = re.sub(
        r"[^a-z0-9]",
        "",
        str(column_name).lower(),
    )

    identifier_names = {
        "id",
        "index",
        "uuid",
        "guid",
        "key",
        "recordid",
        "rowid",
    }

    name_signal = (
        normalized_name in identifier_names
        or normalized_name.endswith("id")
        or normalized_name.endswith("uuid")
        or normalized_name.endswith("guid")
        or normalized_name.endswith("key")
        or normalized_name.endswith("code")
    )

    evidence = []

    if name_signal:
        evidence.append("identifier-like column name")

    observed = series.dropna()

    if len(observed) >= 3:
        numeric_values = pd.to_numeric(
            observed,
            errors="coerce",
        )

        if numeric_values.notna().all():
            values = numeric_values.to_numpy(dtype=float)
            integer_like = bool(
                np.allclose(values, np.round(values))
            )
            unique_ratio = (
                len(np.unique(values)) / len(values)
            )
            differences = np.diff(values)
            sequential = bool(
                integer_like
                and unique_ratio >= 0.95
                and len(differences) > 0
                and (
                    np.allclose(differences, 1)
                    or np.allclose(differences, -1)
                )
            )

            if sequential:
                evidence.append(
                    "sequential integer values"
                )

    return evidence


def _semantic_type(
    series: pd.Series,
    is_datetime: bool,
    unique_count: int,
    unique_ratio: float,
    average_text_length: float | None,
    config: EDAConfig,
) -> str:
    """Infer a practical tabular semantic type."""

    if is_datetime:
        return "datetime"

    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series.dtype):
        observed_numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        integer_like = bool(
            not observed_numeric.empty
            and np.allclose(
                observed_numeric.to_numpy(dtype=float),
                np.round(
                    observed_numeric.to_numpy(dtype=float)
                ),
            )
        )

        if (
            integer_like
            and unique_count
            <= config.max_categories_displayed
        ):
            return "discrete_numerical"

        return "continuous_numerical"

    categorical_dtype = (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )

    if categorical_dtype:
        high_cardinality = (
            unique_count
            > config.high_cardinality_count_threshold
            and unique_ratio
            > config.high_cardinality_ratio_threshold
        )

        if (
            high_cardinality
            and average_text_length is not None
            and average_text_length >= 30
        ):
            return "free_text"

        if high_cardinality:
            return "high_cardinality_categorical"

        return "categorical"

    return "unsupported_or_complex"


def _top_missingness_patterns(
    missing_mask: pd.DataFrame,
    config: EDAConfig,
) -> list[dict]:
    """Return the most frequent row-level missingness patterns."""

    pattern_counts = Counter(
        tuple(bool(value) for value in row)
        for row in missing_mask.to_numpy(dtype=bool)
    ).most_common(config.max_categories_displayed)

    patterns = []

    for pattern, count in pattern_counts:
        missing_positions = [
            position
            for position, is_missing in enumerate(pattern)
            if bool(is_missing)
        ]

        missing_columns = [
            _safe_value(missing_mask.columns[position])
            for position in missing_positions[
                : config.max_categories_displayed
            ]
        ]

        patterns.append(
            {
                "missing_column_count": len(
                    missing_positions
                ),
                "missing_columns": missing_columns,
                "missing_columns_truncated": (
                    len(missing_positions)
                    > config.max_categories_displayed
                ),
                "row_count": int(count),
                "row_percentage": round(
                    int(count) / len(missing_mask) * 100,
                    2,
                ),
            }
        )

    return patterns


def _identical_column_pairs(
    df: pd.DataFrame,
    config: EDAConfig,
) -> tuple[list[dict], list[int]]:
    """Detect exact duplicate columns within a bounded subset."""

    positions = sorted(
        range(len(df.columns)),
        key=lambda position: (
            -int(df.iloc[:, position].notna().sum()),
            position,
        ),
    )

    selected_positions = positions[
        : config.max_pairwise_features
    ]
    omitted_positions = positions[
        config.max_pairwise_features :
    ]

    identical_pairs = []

    for first_position, second_position in combinations(
        selected_positions,
        2,
    ):
        first_series = df.iloc[:, first_position]
        second_series = df.iloc[:, second_position]

        if (
            str(first_series.dtype)
            == str(second_series.dtype)
            and first_series.equals(second_series)
        ):
            identical_pairs.append(
                {
                    "column_1": _safe_value(
                        df.columns[first_position]
                    ),
                    "column_1_position": first_position,
                    "column_2": _safe_value(
                        df.columns[second_position]
                    ),
                    "column_2_position": second_position,
                }
            )

    return identical_pairs, omitted_positions


def generate_dataset_overview(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
) -> dict:
    """
    Generate dataset-level structure and quality diagnostics.

    The report is JSON-safe and never modifies the source DataFrame.
    It reports schema concerns but leaves cleaning decisions to the
    preprocessing pipeline.
    """

    column_names = list(df.columns)

    if target_column not in column_names:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    config = config or EDAConfig()
    total_rows, total_columns = df.shape
    total_cells = total_rows * total_columns

    duplicate_column_names = [
        _safe_value(name)
        for name in pd.Index(column_names)[
            pd.Index(column_names).duplicated()
        ].tolist()
    ]
    column_names_are_unique = len(
        duplicate_column_names
    ) == 0
    target_column_occurrences = column_names.count(
        target_column
    )

    datetime_columns = set()
    datetime_inference_skipped = False

    if column_names_are_unique:
        datetime_report = generate_datetime_statistics(
            df,
            target_column,
            config,
        )
        datetime_columns = set(
            datetime_report["datetime_columns"]
        )
    else:
        datetime_inference_skipped = True

    missing_mask = df.isna()
    total_missing_cells = int(missing_mask.sum().sum())
    rows_with_missing = int(missing_mask.any(axis=1).sum())
    complete_rows = total_rows - rows_with_missing
    fully_empty_rows = int(missing_mask.all(axis=1).sum())
    row_missing_ratios = missing_mask.mean(axis=1)
    high_missing_rows = int(
        (
            row_missing_ratios
            >= config.row_missing_warning_threshold
        ).sum()
    )

    try:
        duplicate_row_count = int(df.duplicated().sum())
        duplicate_row_analysis_available = True
    except TypeError:
        duplicate_row_count = None
        duplicate_row_analysis_available = False

    identical_column_pairs, duplicate_scan_omitted = (
        _identical_column_pairs(df, config)
    )

    column_profiles = []
    dtype_counts: dict[str, int] = {}
    semantic_type_counts: dict[str, int] = {}

    for position in range(total_columns):
        column_name = column_names[position]
        series = df.iloc[:, position]
        data_type = str(series.dtype)
        dtype_counts[data_type] = (
            dtype_counts.get(data_type, 0) + 1
        )

        missing_count = int(series.isna().sum())
        observed = series.dropna()
        observed_count = len(observed)
        unique_count = _safe_nunique(series)
        unique_ratio = (
            unique_count / observed_count
            if observed_count > 0
            else 0.0
        )
        top_frequency = _top_frequency(series)
        top_frequency_ratio = (
            top_frequency / observed_count
            if observed_count > 0
            else 0.0
        )

        average_text_length = None
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
        ) and observed_count > 0:
            average_text_length = float(
                observed.map(str).str.len().mean()
            )

        is_datetime = (
            column_name in datetime_columns
            or pd.api.types.is_datetime64_any_dtype(
                series.dtype
            )
        )

        semantic_type = _semantic_type(
            series,
            is_datetime,
            unique_count,
            unique_ratio,
            average_text_length,
            config,
        )
        semantic_type_counts[semantic_type] = (
            semantic_type_counts.get(semantic_type, 0)
            + 1
        )

        identifier_evidence = _identifier_evidence(
            column_name,
            series,
        )

        high_missing = (
            missing_count / total_rows
            >= config.high_missing_threshold
        )
        high_cardinality = semantic_type in {
            "high_cardinality_categorical",
            "free_text",
        }
        near_constant = (
            observed_count > 0
            and top_frequency_ratio
            >= config.near_constant_threshold
        )
        constant_or_empty = unique_count <= 1

        infinite_count = 0
        if pd.api.types.is_numeric_dtype(series.dtype):
            infinite_count = int(
                pd.to_numeric(
                    series,
                    errors="coerce",
                ).isin([np.inf, -np.inf]).sum()
            )

        warnings = []
        if high_missing:
            warnings.append("high missingness")
        if constant_or_empty:
            warnings.append("constant or empty column")
        elif near_constant:
            warnings.append("near-constant column")
        if high_cardinality:
            warnings.append("high cardinality")
        if identifier_evidence:
            warnings.append("identifier candidate")
        if infinite_count > 0:
            warnings.append("contains infinite values")
        if semantic_type == "unsupported_or_complex":
            warnings.append("unsupported or complex data type")

        column_key = (
            str(column_name)
            if column_names_are_unique
            else f"{column_name}__position_{position}"
        )

        column_profiles.append(
            {
                "column": _safe_value(column_name),
                "column_key": column_key,
                "position": position,
                "role": (
                    "target"
                    if column_name == target_column
                    else "feature"
                ),
                "data_type": data_type,
                "semantic_type": semantic_type,
                "total_rows": total_rows,
                "observed_values": observed_count,
                "missing_values": missing_count,
                "missing_percentage": round(
                    missing_count / total_rows * 100,
                    2,
                ),
                "unique_values": unique_count,
                "unique_ratio": round(
                    unique_ratio,
                    config.decimal_places,
                ),
                "constant_or_empty": constant_or_empty,
                "near_constant": near_constant,
                "high_missing": high_missing,
                "high_cardinality": high_cardinality,
                "identifier_candidate": bool(
                    identifier_evidence
                ),
                "identifier_evidence": identifier_evidence,
                "infinite_values": infinite_count,
                "average_text_length": (
                    round(
                        average_text_length,
                        config.decimal_places,
                    )
                    if average_text_length is not None
                    else None
                ),
                "sample_values": [
                    _safe_value(value)
                    for value in observed.head(3).tolist()
                ],
                "memory_usage_bytes": int(
                    series.memory_usage(
                        index=False,
                        deep=True,
                    )
                ),
                "warnings": warnings,
            }
        )

    high_missing_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["high_missing"]
    ]
    constant_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["constant_or_empty"]
    ]
    near_constant_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["near_constant"]
        and not profile["constant_or_empty"]
    ]
    identifier_candidates = [
        profile["column"]
        for profile in column_profiles
        if profile["identifier_candidate"]
        and profile["role"] == "feature"
    ]
    unsupported_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["semantic_type"]
        == "unsupported_or_complex"
    ]
    high_cardinality_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["high_cardinality"]
    ]
    free_text_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["semantic_type"] == "free_text"
    ]
    detected_datetime_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["semantic_type"] == "datetime"
    ]
    infinite_value_columns = [
        profile["column"]
        for profile in column_profiles
        if profile["infinite_values"] > 0
    ]

    missing_column_summary = sorted(
        [
            {
                "column": profile["column"],
                "missing_values": profile[
                    "missing_values"
                ],
                "missing_percentage": profile[
                    "missing_percentage"
                ],
            }
            for profile in column_profiles
            if profile["missing_values"] > 0
        ],
        key=lambda item: item["missing_values"],
        reverse=True,
    )

    target_position = column_names.index(target_column)
    target_series = df.iloc[:, target_position]
    target_observed_count = int(target_series.notna().sum())
    target_unique_count = _safe_nunique(target_series)

    blockers = []
    if not column_names_are_unique:
        blockers.append("duplicate column names")
    if target_column_occurrences != 1:
        blockers.append("ambiguous target column occurrence")
    if total_columns < 2:
        blockers.append("no candidate feature columns")
    if target_observed_count == 0:
        blockers.append("target contains no observed values")
    elif target_unique_count < 2:
        blockers.append("target has fewer than two unique values")

    warnings = []
    if total_rows < config.minimum_sample_size:
        warnings.append(
            "Dataset row count is below the configured minimum "
            "sample size."
        )
    if duplicate_column_names:
        warnings.append(
            "Duplicate column names make column selection "
            "ambiguous."
        )
    if duplicate_row_count:
        warnings.append("Exact duplicate rows were detected.")
    if not duplicate_row_analysis_available:
        warnings.append(
            "Duplicate-row analysis was unavailable because one "
            "or more columns contain unhashable values."
        )
    if high_missing_columns:
        warnings.append(
            "One or more columns exceed the high-missingness "
            "threshold."
        )
    if high_missing_rows > 0:
        warnings.append(
            "One or more rows exceed the row-missingness warning "
            "threshold."
        )
    if constant_columns:
        warnings.append(
            "Constant or empty columns were detected."
        )
    if identifier_candidates:
        warnings.append(
            "Identifier-like feature columns require exclusion "
            "review before modeling."
        )
    if identical_column_pairs:
        warnings.append(
            "Exactly identical column pairs were detected."
        )
    if datetime_inference_skipped:
        warnings.append(
            "Text datetime inference was skipped because column "
            "names are not unique."
        )

    memory_usage_bytes = int(
        df.memory_usage(index=True, deep=True).sum()
    )

    return {
        "target_column": target_column,
        "shape": {
            "rows": total_rows,
            "columns": total_columns,
            "candidate_features": total_columns - 1,
            "total_cells": total_cells,
        },
        "memory": {
            "memory_usage_bytes": memory_usage_bytes,
            "memory_usage_megabytes": round(
                memory_usage_bytes / (1024 ** 2),
                config.decimal_places,
            ),
        },
        "schema": {
            "column_names_are_unique": (
                column_names_are_unique
            ),
            "duplicate_column_names": (
                duplicate_column_names
            ),
            "target_column_occurrences": (
                target_column_occurrences
            ),
            "dtype_counts": dtype_counts,
            "semantic_type_counts": semantic_type_counts,
            "schema_ready_for_eda": (
                column_names_are_unique
                and target_column_occurrences == 1
            ),
        },
        "missingness": {
            "total_missing_cells": total_missing_cells,
            "total_missing_percentage": round(
                total_missing_cells / total_cells * 100,
                2,
            ),
            "complete_rows": complete_rows,
            "complete_row_percentage": round(
                complete_rows / total_rows * 100,
                2,
            ),
            "rows_with_missing_values": rows_with_missing,
            "rows_with_missing_percentage": round(
                rows_with_missing / total_rows * 100,
                2,
            ),
            "fully_empty_rows": fully_empty_rows,
            "high_missing_rows": high_missing_rows,
            "row_missing_warning_threshold": (
                config.row_missing_warning_threshold
            ),
            "high_missing_columns": high_missing_columns,
            "column_summary": missing_column_summary,
            "top_missingness_patterns": (
                _top_missingness_patterns(
                    missing_mask,
                    config,
                )
            ),
        },
        "duplicates": {
            "duplicate_row_analysis_available": (
                duplicate_row_analysis_available
            ),
            "duplicate_rows": duplicate_row_count,
            "duplicate_row_percentage": (
                round(
                    duplicate_row_count
                    / total_rows
                    * 100,
                    2,
                )
                if duplicate_row_count is not None
                else None
            ),
            "identical_column_pairs": (
                identical_column_pairs
            ),
            "identical_column_scan_limit": (
                config.max_pairwise_features
            ),
            "identical_column_scan_omitted_positions": (
                duplicate_scan_omitted
            ),
        },
        "quality_flags": {
            "high_missing_columns": high_missing_columns,
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "high_cardinality_columns": (
                high_cardinality_columns
            ),
            "free_text_columns": free_text_columns,
            "datetime_columns": detected_datetime_columns,
            "infinite_value_columns": infinite_value_columns,
            "identifier_candidates": identifier_candidates,
            "unsupported_columns": unsupported_columns,
        },
        "column_profiles": column_profiles,
        "basic_modeling_readiness": {
            "ready": len(blockers) == 0,
            "blockers": blockers,
            "scope": (
                "Basic schema and target usability only; leakage, "
                "preprocessing, and model-specific checks are "
                "reported separately."
            ),
        },
        "warnings": warnings,
        "automatic_data_modification_performed": False,
    }