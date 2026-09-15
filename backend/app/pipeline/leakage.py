import pandas as pd


def detect_potential_leakage(
    df: pd.DataFrame,
    target_column: str,
    correlation_threshold: float = 0.995,
    minimum_pair_coverage: float = 0.80,
) -> dict:
    """
    Detect statistical evidence of possible target leakage.

    This is a warning system, not an automatic column-removal
    function.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if not 0 < correlation_threshold <= 1:
        raise ValueError(
            "correlation_threshold must be between 0 and 1."
        )

    if not 0 < minimum_pair_coverage <= 1:
        raise ValueError(
            "minimum_pair_coverage must be between 0 and 1."
        )

    target = df[target_column]

    candidate_features = [
        column
        for column in df.columns
        if column != target_column
    ]

    leakage_evidence = {}

    def add_evidence(
        column: str,
        reason: str,
    ) -> None:
        leakage_evidence.setdefault(
            column,
            [],
        ).append(reason)

    for column in candidate_features:
        feature = df[column]

        # Check 1: Exact copy of target
        if feature.equals(target):
            add_evidence(
                column,
                "feature is an exact copy of the target",
            )

        paired_data = df[
            [column, target_column]
        ].dropna()

        if len(paired_data) < 3:
            continue

        pair_coverage = len(paired_data) / len(df)

        if pair_coverage < minimum_pair_coverage:
            continue

        # Check 2: Near-perfect numerical correlation
        if (
            pd.api.types.is_numeric_dtype(feature.dtype)
            and pd.api.types.is_numeric_dtype(target.dtype)
            and paired_data[column].nunique() > 1
            and paired_data[target_column].nunique() > 1
        ):
            correlation = paired_data[column].corr(
                paired_data[target_column]
            )

            if (
                pd.notna(correlation)
                and abs(correlation) >= correlation_threshold
            ):
                add_evidence(
                    column,
                    (
                        "near-perfect target correlation: "
                        f"{correlation:.4f}"
                    ),
                )

        # Check 3: Feature values deterministically map to target
        feature_unique_count = int(
            paired_data[column].nunique()
        )

        feature_unique_ratio = (
            feature_unique_count / len(paired_data)
        )

        if (
            2 <= feature_unique_count <= 50
            and feature_unique_ratio <= 0.20
        ):
            try:
                target_counts_per_value = (
                    paired_data.groupby(
                        column,
                        dropna=False,
                    )[target_column].nunique()
                )

                if (
                    not target_counts_per_value.empty
                    and target_counts_per_value.max() == 1
                ):
                    add_evidence(
                        column,
                        (
                            "feature values deterministically "
                            "map to the target"
                        ),
                    )

            except TypeError:
                # Some uncommon object values may not be hashable.
                pass

    return {
        "target_column": target_column,
        "candidate_feature_count": len(candidate_features),
        "potential_leakage_count": len(leakage_evidence),
        "potential_leakage_columns": list(
            leakage_evidence.keys()
        ),
        "leakage_evidence": leakage_evidence,
        "automatic_removal_performed": False,
        "limitations": [
            (
                "Statistical checks cannot detect every form "
                "of semantic or time-based leakage."
            ),
            (
                "Flagged relationships require review before "
                "feature removal."
            ),
        ],
    }