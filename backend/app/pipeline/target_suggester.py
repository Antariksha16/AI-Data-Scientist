import re

import pandas as pd


IDENTIFIER_TOKENS = {
    "id",
    "identifier",
    "uuid",
    "guid",
    "index",
    "key",
    "serial",
    "number",
    "no",
}


def _get_column_name_tokens(column_name: str) -> set[str]:
    """
    Convert names such as EmployeeID and passenger_id
    into lowercase tokens.
    """

    separated_name = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        str(column_name),
    )

    return set(
        re.findall(
            r"[a-z0-9]+",
            separated_name.lower(),
        )
    )


def _is_sequential_integer(series: pd.Series) -> bool:
    """
    Check whether a column contains unique sequential integers,
    such as 1, 2, 3, 4, ...
    """

    values = series.dropna()

    if values.empty:
        return False

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric_values.isna().any():
        return False

    if not ((numeric_values % 1) == 0).all():
        return False

    if numeric_values.nunique() != len(numeric_values):
        return False

    sorted_values = numeric_values.sort_values()
    differences = sorted_values.diff().dropna()

    return (
        not differences.empty
        and differences.eq(1).all()
    )


def _is_identifier_like(
    column_name: str,
    series: pd.Series,
) -> bool:
    """
    Detect identifier-like columns using multiple signals.

    High uniqueness alone is not enough because a valid
    regression target may also contain unique values.
    """

    values = series.dropna()

    if values.empty:
        return False

    unique_ratio = values.nunique() / len(values)

    if unique_ratio < 0.98:
        return False

    name_tokens = _get_column_name_tokens(column_name)

    has_identifier_name = bool(
        name_tokens.intersection(IDENTIFIER_TOKENS)
    )

    has_sequential_values = _is_sequential_integer(series)

    return has_identifier_name or has_sequential_values


def suggest_target_columns(
    df: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    """
    Rank possible target columns.

    This function only provides suggestions.
    The user must make the final target selection.
    """

    suggestions = []
    total_rows = len(df)

    if total_rows == 0 or top_k <= 0:
        return suggestions

    for column in df.columns:
        series = df[column]

        missing_count = int(series.isnull().sum())
        missing_ratio = missing_count / total_rows

        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else 0
        )

        score = 0.0
        reasons = []
        suggested_problem_type = "undetermined"

        # Missing-value quality
        if missing_ratio == 0:
            score += 2
            reasons.append("no missing values")

        elif missing_ratio <= 0.10:
            score += 1
            reasons.append("low missingness")

        elif missing_ratio <= 0.50:
            score -= 1
            reasons.append("moderate missingness")

        else:
            score -= 4
            reasons.append("high missingness")

        identifier_like = _is_identifier_like(
            column,
            series,
        )

        # Unsuitable target: empty or constant
        if unique_count == 0:
            score -= 10
            reasons.append("completely empty column")

        elif unique_count == 1:
            score -= 10
            reasons.append("constant column")

        # Identifier detection
        elif identifier_like:
            score -= 8
            suggested_problem_type = "unlikely_target"
            reasons.append(
                "identifier-like name or sequential values"
            )

        # Boolean target
        elif pd.api.types.is_bool_dtype(series.dtype):
            score += 4
            suggested_problem_type = "classification"
            reasons.extend(
                [
                    "boolean target",
                    "strong classification structure",
                ]
            )

        # Text or categorical target
        elif (
            pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            score += 1
            suggested_problem_type = "classification"
            reasons.append("categorical target")

            if unique_count <= 50 or unique_ratio <= 0.20:
                score += 3
                reasons.append(
                    "usable classification cardinality"
                )
            else:
                score -= 3
                reasons.append(
                    "very high categorical cardinality"
                )

        # Numeric target
        elif pd.api.types.is_numeric_dtype(series.dtype):
            score += 1
            reasons.append("numeric target")

            if unique_count == 2:
                score += 3
                suggested_problem_type = "classification"
                reasons.append(
                    "binary classification structure"
                )

            elif unique_count <= 20 or unique_ratio <= 0.05:
                score += 2
                suggested_problem_type = (
                    "classification_or_regression"
                )
                reasons.append(
                    "discrete numeric target; objective is ambiguous"
                )

            else:
                score += 3
                suggested_problem_type = "regression"
                reasons.append(
                    "continuous numeric regression structure"
                )

        else:
            score -= 2
            reasons.append("unsupported or unclear data type")

        suggestions.append(
            {
                "column": column,
                "score": round(score, 2),
                "suggested_problem_type": suggested_problem_type,
                "identifier_like": identifier_like,
                "unique_values": unique_count,
                "unique_percentage": round(
                    unique_ratio * 100,
                    2,
                ),
                "missing_values": missing_count,
                "missing_percentage": round(
                    missing_ratio * 100,
                    2,
                ),
                "reasons": reasons,
            }
        )

    suggestions.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return suggestions[:top_k]