import math
from typing import Any

import pandas as pd

from ml_pipeline.model_registry import ModelSpec


SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}
SUPPORTED_MODEL_FAMILIES = {
    "linear",
    "bagging",
    "boosting",
    "neighbor",
    "kernel",
    "probabilistic",
    "discriminant",
    "robust_linear",
    "tree",
    "baseline",
    "unspecified",
}


def _safe_float(value: Any, decimal_places: int = 6) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(converted):
        return None

    return round(converted, decimal_places)


def _validate_inputs(
    model_specs: list[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    estimated_transformed_features: int,
    max_models: int,
    small_dataset_rows: int,
    large_dataset_rows: int,
    high_dimensional_features: int,
    extreme_imbalance_ratio: float,
) -> None:
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a Pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a Pandas Series.")

    if X_train.empty or y_train.empty:
        raise ValueError("Training data must not be empty.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal rows.")

    if not model_specs:
        raise ValueError("model_specs must not be empty.")

    model_names = [spec.name for spec in model_specs]

    if len(set(model_names)) != len(model_names):
        raise ValueError("model_specs contains duplicate names.")

    if any(
        spec.family not in SUPPORTED_MODEL_FAMILIES
        for spec in model_specs
    ):
        raise ValueError("model_specs contains an unsupported family.")

    if not isinstance(estimated_transformed_features, int):
        raise TypeError(
            "estimated_transformed_features must be an integer."
        )

    if estimated_transformed_features < 1:
        raise ValueError(
            "estimated_transformed_features must be positive."
        )

    if not isinstance(max_models, int) or not 1 <= max_models <= 8:
        raise ValueError("max_models must be between 1 and 8.")

    if small_dataset_rows < 20:
        raise ValueError("small_dataset_rows must be at least 20.")

    if large_dataset_rows <= small_dataset_rows:
        raise ValueError(
            "large_dataset_rows must exceed small_dataset_rows."
        )

    if high_dimensional_features < 10:
        raise ValueError(
            "high_dimensional_features must be at least 10."
        )

    if not 0 < extreme_imbalance_ratio <= 0.50:
        raise ValueError(
            "extreme_imbalance_ratio must be greater than 0 "
            "and at most 0.50."
        )


def _profile_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    preprocessing_metadata: dict[str, Any],
    estimated_transformed_features: int,
    *,
    small_dataset_rows: int,
    large_dataset_rows: int,
    high_dimensional_features: int,
    extreme_imbalance_ratio: float,
) -> dict[str, Any]:
    rows = len(X_train)

    if rows < small_dataset_rows:
        dataset_size = "small"
    elif rows < large_dataset_rows:
        dataset_size = "medium"
    else:
        dataset_size = "large"

    numerical_features = list(
        preprocessing_metadata.get("numerical_features", [])
    )
    categorical_features = list(
        preprocessing_metadata.get("categorical_features", [])
    )
    robust_features = list(
        preprocessing_metadata.get("robust_scaler_features", [])
    )
    missing_values = int(X_train.isna().sum().sum())
    total_feature_values = int(X_train.shape[0] * X_train.shape[1])
    missing_percentage = (
        missing_values / total_feature_values * 100
        if total_feature_values
        else 0.0
    )
    profile: dict[str, Any] = {
        "profile_scope": "training_split_only",
        "holdout_data_used": False,
        "row_count": rows,
        "raw_feature_count": int(X_train.shape[1]),
        "estimated_transformed_features": (
            estimated_transformed_features
        ),
        "dataset_size": dataset_size,
        "high_dimensional": (
            estimated_transformed_features
            >= high_dimensional_features
        ),
        "transformed_feature_to_row_ratio": _safe_float(
            estimated_transformed_features / rows
        ),
        "numerical_feature_count": len(numerical_features),
        "categorical_feature_count": len(categorical_features),
        "mixed_feature_types": bool(
            numerical_features and categorical_features
        ),
        "robust_scaler_feature_count": len(robust_features),
        "missing_feature_values": missing_values,
        "missing_feature_percentage": _safe_float(
            missing_percentage
        ),
        "target_values_stored": False,
    }

    if problem_type == "classification":
        class_counts = y_train.value_counts(dropna=False)
        smallest_class_count = int(class_counts.min())
        minority_class_ratio = smallest_class_count / len(y_train)

        if minority_class_ratio < extreme_imbalance_ratio:
            imbalance_status = "extreme"
        elif minority_class_ratio < 0.25:
            imbalance_status = "moderate"
        else:
            imbalance_status = "not_severe"

        profile.update(
            {
                "class_count": int(len(class_counts)),
                "smallest_class_count": smallest_class_count,
                "minority_class_ratio": _safe_float(
                    minority_class_ratio
                ),
                "class_imbalance_status": imbalance_status,
            }
        )
    else:
        numeric_target = pd.to_numeric(y_train, errors="coerce")
        profile.update(
            {
                "target_skewness": _safe_float(
                    numeric_target.skew()
                ),
                "class_count": None,
                "smallest_class_count": None,
                "minority_class_ratio": None,
                "class_imbalance_status": "not_applicable",
            }
        )

    return profile


def _eligibility(
    spec: ModelSpec,
    profile: dict[str, Any],
) -> tuple[bool, str | None]:
    if spec.is_baseline:
        return False, "dummy_baseline_is_evaluated_separately"

    if (
        spec.max_recommended_rows is not None
        and profile["row_count"] > spec.max_recommended_rows
    ):
        return False, "row_count_exceeds_model_recommendation_limit"

    if (
        spec.max_recommended_transformed_features is not None
        and profile["estimated_transformed_features"]
        > spec.max_recommended_transformed_features
    ):
        return (
            False,
            "transformed_features_exceed_model_recommendation_limit",
        )

    return True, None


def _score_model(
    spec: ModelSpec,
    profile: dict[str, Any],
    problem_type: str,
) -> tuple[float, list[str]]:
    family_base_scores = {
        "linear": 92.0,
        "robust_linear": 92.0,
        "boosting": 90.0,
        "bagging": 88.0,
        "kernel": 86.0,
        "probabilistic": 84.0,
        "neighbor": 82.0,
        "discriminant": 80.0,
        "tree": 82.0,
        "unspecified": 75.0,
    }
    cost_adjustments = {
        "low": 5.0,
        "medium": 0.0,
        "high": -5.0,
    }
    score = family_base_scores.get(spec.family, 75.0)
    score += cost_adjustments.get(spec.computational_cost, -2.0)
    reasons = [
        f"represents_{spec.family}_model_family",
        f"computational_cost_{spec.computational_cost}",
    ]

    if spec.name in {"logistic_regression", "ridge"}:
        score += 4.0
        reasons.append("stable_regularized_reference_model")

    if spec.name == "hist_gradient_boosting":
        score += 4.0
        reasons.append("efficient_histogram_based_tabular_boosting")
    dataset_size = profile["dataset_size"]

    if dataset_size == "small":
        if spec.family in {"linear", "robust_linear"}:
            score += 5.0
            reasons.append("efficient_reference_for_small_dataset")
        elif spec.family == "kernel":
            score += 10.0
            reasons.append("kernel_models_are_viable_on_small_dataset")
        elif spec.family == "neighbor":
            score += 8.0
            reasons.append("local_patterns_are_affordable_on_small_data")
        elif spec.family in {"probabilistic", "discriminant"}:
            score += 6.0
            reasons.append("efficient_statistical_small_data_candidate")
        elif spec.family == "tree":
            score += 5.0
            reasons.append("interpretable_nonlinear_small_data_candidate")
        elif spec.family == "bagging":
            score += 6.0
            reasons.append("captures_small_dataset_nonlinearity")
        elif spec.family == "boosting":
            score += 2.0
            reasons.append("boosting_remains_candidate_on_small_data")
    elif dataset_size == "medium":
        if spec.family == "boosting":
            score += 8.0
            reasons.append("well_suited_to_medium_tabular_dataset")
        elif spec.family == "bagging":
            score += 6.0
            reasons.append("captures_nonlinear_feature_interactions")
        elif spec.family in {"linear", "robust_linear"}:
            score += 5.0
            reasons.append("efficient_linear_reference")
        elif spec.family == "kernel":
            score += 2.0
            reasons.append("kernel_candidate_within_medium_data_budget")
        elif spec.family == "tree":
            score += 4.0
            reasons.append("efficient_nonlinear_tree_reference")
    else:
        if spec.family in {"linear", "robust_linear"}:
            score += 15.0
            reasons.append("scales_well_to_large_dataset")
        elif spec.family == "boosting":
            score += 12.0
            reasons.append("efficient_boosting_for_large_tabular_data")
        elif spec.family == "bagging":
            score -= 10.0
            reasons.append("bagging_cost_penalty_for_large_dataset")
        elif spec.family == "neighbor":
            score -= 20.0
            reasons.append("prediction_cost_penalty_for_large_dataset")
        elif spec.family == "kernel":
            if spec.computational_cost == "low":
                score += 5.0
                reasons.append("linear_margin_model_scales_to_large_data")
            else:
                score -= 15.0
                reasons.append("nonlinear_kernel_cost_penalty_large_data")
        elif spec.family == "probabilistic":
            score += 8.0
            reasons.append("probabilistic_model_scales_to_large_data")
        elif spec.family == "tree":
            score += 2.0
            reasons.append("single_tree_remains_affordable_on_large_data")

    if profile["high_dimensional"]:
        if spec.family in {"linear", "robust_linear"}:
            score += 15.0
            reasons.append("suitable_for_high_dimensional_features")
        elif spec.family == "boosting":
            score -= 10.0
            reasons.append("dense_high_dimensional_cost_penalty")
        elif spec.family == "neighbor":
            score -= 20.0
            reasons.append("distance_quality_high_dimension_penalty")
        elif spec.family == "kernel":
            if spec.computational_cost == "low":
                score += 8.0
                reasons.append("linear_margin_high_dimension_candidate")
            else:
                score -= 15.0
                reasons.append("nonlinear_kernel_high_dimension_penalty")
        elif spec.family == "probabilistic":
            score += 6.0
            reasons.append("independent_feature_high_dimension_candidate")
        elif spec.family in {"discriminant", "tree"}:
            score -= 8.0
            reasons.append("high_dimension_family_penalty")

    if profile["mixed_feature_types"]:
        if spec.family in {"bagging", "boosting", "tree"}:
            score += 5.0
            reasons.append("captures_mixed_feature_interactions")
        elif spec.family in {"linear", "robust_linear"}:
            score += 2.0
            reasons.append("compatible_with_encoded_mixed_features")

    if (
        problem_type == "classification"
        and profile["class_imbalance_status"]
        in {"moderate", "extreme"}
    ):
        if spec.handles_class_imbalance:
            score += 8.0
            reasons.append("configured_for_class_imbalance")
        else:
            score -= 12.0
            reasons.append("class_imbalance_support_penalty")

    if (
        problem_type == "regression"
        and profile["robust_scaler_feature_count"] > 0
    ):
        if spec.family in {"bagging", "boosting"}:
            score += 5.0
            reasons.append("nonlinear_family_for_outlier_context")
        elif spec.family == "robust_linear":
            score += 8.0
            reasons.append("robust_linear_outlier_handling")

    return round(score, 6), reasons


def _select_diverse_models(
    decisions: list[dict[str, Any]],
    max_models: int,
) -> list[str]:
    eligible = [
        decision
        for decision in decisions
        if decision["eligibility"] == "eligible"
    ]
    eligible.sort(
        key=lambda decision: (
            -decision["recommendation_score"],
            decision["model_name"],
        )
    )

    if not eligible:
        raise ValueError("No non-baseline model is eligible.")

    selected = [eligible[0]]
    represented_families = {eligible[0]["family"]}

    for decision in eligible[1:]:
        if len(selected) >= max_models:
            break

        if decision["family"] not in represented_families:
            selected.append(decision)
            represented_families.add(decision["family"])

    for decision in eligible:
        if len(selected) >= max_models:
            break

        if decision not in selected:
            selected.append(decision)

    return [decision["model_name"] for decision in selected]


def recommend_models(
    model_specs: list[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    preprocessing_metadata: dict[str, Any],
    estimated_transformed_features: int,
    *,
    max_models: int = 4,
    small_dataset_rows: int = 5_000,
    large_dataset_rows: int = 50_000,
    high_dimensional_features: int = 1_000,
    extreme_imbalance_ratio: float = 0.10,
) -> dict[str, Any]:
    """Recommend a bounded, diverse shortlist from training data only."""

    _validate_inputs(
        model_specs,
        X_train,
        y_train,
        problem_type,
        estimated_transformed_features,
        max_models,
        small_dataset_rows,
        large_dataset_rows,
        high_dimensional_features,
        extreme_imbalance_ratio,
    )
    profile = _profile_training_data(
        X_train,
        y_train,
        problem_type,
        preprocessing_metadata,
        estimated_transformed_features,
        small_dataset_rows=small_dataset_rows,
        large_dataset_rows=large_dataset_rows,
        high_dimensional_features=high_dimensional_features,
        extreme_imbalance_ratio=extreme_imbalance_ratio,
    )
    baseline = next(
        (spec for spec in model_specs if spec.is_baseline),
        None,
    )

    if baseline is None:
        raise ValueError("model_specs does not contain a baseline.")

    decisions = []

    for spec in model_specs:
        if spec.is_baseline:
            continue

        eligible, exclusion_reason = _eligibility(spec, profile)

        if eligible:
            score, reasons = _score_model(
                spec,
                profile,
                problem_type,
            )
            eligibility = "eligible"
        else:
            score = None
            reasons = [exclusion_reason]
            eligibility = "ineligible"

        decisions.append(
            {
                "model_name": spec.name,
                "display_name": spec.display_name,
                "family": spec.family,
                "computational_cost": spec.computational_cost,
                "eligibility": eligibility,
                "recommendation_score": score,
                "decision": None,
                "reasons": reasons,
            }
        )

    selected_names = _select_diverse_models(
        decisions,
        max_models,
    )
    selected_set = set(selected_names)

    for decision in decisions:
        if decision["model_name"] in selected_set:
            decision["decision"] = "selected"
            decision["reasons"].append(
                "selected_by_bounded_diverse_recommender"
            )
        elif decision["eligibility"] == "eligible":
            decision["decision"] = "skipped"
            decision["reasons"].append(
                "candidate_limit_reached_after_higher_priority_"
                "and_family_diversity_selection"
            )
        else:
            decision["decision"] = "skipped"

    selected_decisions = [
        decision
        for decision in decisions
        if decision["decision"] == "selected"
    ]
    skipped_decisions = [
        decision
        for decision in decisions
        if decision["decision"] == "skipped"
    ]

    return {
        "recommendation_status": "complete",
        "selection_mode": "automatic_recommendation",
        "selection_scope": "training_split_only",
        "holdout_data_used": False,
        "configuration": {
            "max_models": max_models,
            "small_dataset_rows": small_dataset_rows,
            "large_dataset_rows": large_dataset_rows,
            "high_dimensional_features": (
                high_dimensional_features
            ),
            "extreme_imbalance_ratio": extreme_imbalance_ratio,
            "family_diversity_required": True,
        },
        "dataset_profile": profile,
        "baseline_model": baseline.name,
        "registered_non_baseline_model_count": len(decisions),
        "eligible_model_count": sum(
            decision["eligibility"] == "eligible"
            for decision in decisions
        ),
        "selected_model_count": len(selected_names),
        "selected_models": selected_names,
        "selected_families": list(
            dict.fromkeys(
                decision["family"]
                for decision in selected_decisions
            )
        ),
        "skipped_model_count": len(skipped_decisions),
        "skipped_models": [
            decision["model_name"]
            for decision in skipped_decisions
        ],
        "model_decisions": decisions,
        "target_values_stored": False,
    }


def describe_user_model_override(
    all_model_specs: list[ModelSpec],
    selected_model_specs: list[ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    preprocessing_metadata: dict[str, Any],
    estimated_transformed_features: int,
    *,
    small_dataset_rows: int = 5_000,
    large_dataset_rows: int = 50_000,
    high_dimensional_features: int = 1_000,
    extreme_imbalance_ratio: float = 0.10,
) -> dict[str, Any]:
    """Describe an explicit user shortlist without replacing it."""

    _validate_inputs(
        all_model_specs,
        X_train,
        y_train,
        problem_type,
        estimated_transformed_features,
        1,
        small_dataset_rows,
        large_dataset_rows,
        high_dimensional_features,
        extreme_imbalance_ratio,
    )
    profile = _profile_training_data(
        X_train,
        y_train,
        problem_type,
        preprocessing_metadata,
        estimated_transformed_features,
        small_dataset_rows=small_dataset_rows,
        large_dataset_rows=large_dataset_rows,
        high_dimensional_features=high_dimensional_features,
        extreme_imbalance_ratio=extreme_imbalance_ratio,
    )
    selected_names = [
        spec.name
        for spec in selected_model_specs
        if not spec.is_baseline
    ]

    if not selected_names:
        raise ValueError(
            "selected_model_specs must contain a non-baseline model."
        )

    selected_set = set(selected_names)
    decisions = []

    for spec in all_model_specs:
        if spec.is_baseline:
            continue

        selected = spec.name in selected_set
        decisions.append(
            {
                "model_name": spec.name,
                "display_name": spec.display_name,
                "family": spec.family,
                "computational_cost": spec.computational_cost,
                "eligibility": "user_controlled",
                "recommendation_score": None,
                "decision": "selected" if selected else "skipped",
                "reasons": [
                    "explicitly_requested_by_user"
                    if selected
                    else "not_in_explicit_user_request"
                ],
            }
        )

    baseline = next(
        spec for spec in selected_model_specs if spec.is_baseline
    )

    return {
        "recommendation_status": "complete",
        "selection_mode": "user_override",
        "selection_scope": "training_split_only",
        "holdout_data_used": False,
        "configuration": None,
        "dataset_profile": profile,
        "baseline_model": baseline.name,
        "registered_non_baseline_model_count": sum(
            not spec.is_baseline for spec in all_model_specs
        ),
        "eligible_model_count": None,
        "selected_model_count": len(selected_names),
        "selected_models": selected_names,
        "selected_families": list(
            dict.fromkeys(
                spec.family
                for spec in selected_model_specs
                if not spec.is_baseline
            )
        ),
        "skipped_model_count": sum(
            decision["decision"] == "skipped"
            for decision in decisions
        ),
        "skipped_models": [
            decision["model_name"]
            for decision in decisions
            if decision["decision"] == "skipped"
        ],
        "model_decisions": decisions,
        "target_values_stored": False,
    }
