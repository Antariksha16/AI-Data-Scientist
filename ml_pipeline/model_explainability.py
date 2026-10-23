import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from ml_pipeline.explainability_config import (
    ExplainabilityConfig,
)


SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}


def _safe_float(value: Any, decimal_places: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(number, decimal_places)


def _validate_inputs(
    training_result: dict[str, Any],
) -> tuple[Pipeline, dict, pd.DataFrame, pd.Series]:
    if not isinstance(training_result, dict):
        raise TypeError("training_result must be a dictionary.")

    pipeline = training_result.get("pipeline")
    report = training_result.get("report")
    context = training_result.get("evaluation_context")

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "training_result does not contain a fitted Pipeline."
        )

    if not isinstance(report, dict):
        raise TypeError(
            "training_result does not contain a report dictionary."
        )

    if not isinstance(context, dict):
        raise ValueError(
            "training_result does not contain holdout evaluation context."
        )

    X_holdout = context.get("X_holdout")
    y_holdout = context.get("y_holdout")

    if not isinstance(X_holdout, pd.DataFrame):
        raise TypeError("X_holdout must be a Pandas DataFrame.")

    if not isinstance(y_holdout, pd.Series):
        raise TypeError("y_holdout must be a Pandas Series.")

    if X_holdout.empty:
        raise ValueError("Holdout features are empty.")

    if len(X_holdout) != len(y_holdout):
        raise ValueError(
            "Holdout features and target have different lengths."
        )

    if X_holdout.columns.duplicated().any():
        raise ValueError("Holdout features contain duplicate columns.")

    if context.get("evaluation_scope") != "untouched_holdout":
        raise ValueError("Explainability requires untouched holdout rows.")

    if context.get("used_for_model_selection") is not False:
        raise ValueError(
            "Explainability context must not be used for model selection."
        )

    return pipeline, report, X_holdout, y_holdout


def _sample_evaluation_rows(
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    problem_type: str,
    config: ExplainabilityConfig,
) -> tuple[pd.DataFrame, pd.Series, bool]:
    if len(X_holdout) <= config.max_rows:
        return (
            X_holdout.copy(deep=True),
            y_holdout.copy(deep=True),
            False,
        )

    generator = np.random.default_rng(config.random_state)
    all_positions = np.arange(len(X_holdout))

    if problem_type == "classification":
        target_frame = pd.DataFrame(
            {"target": y_holdout.reset_index(drop=True)}
        )
        groups = list(
            target_frame.groupby(
                "target",
                sort=False,
                dropna=False,
            ).indices.values()
        )

        if len(groups) > config.max_rows:
            raise ValueError(
                "max_rows is smaller than the number of classes."
            )

        selected_positions = [
            int(generator.choice(group_positions))
            for group_positions in groups
        ]
        selected_set = set(selected_positions)
        remaining_positions = np.asarray(
            [
                position
                for position in all_positions
                if position not in selected_set
            ],
            dtype=int,
        )
        remaining_count = (
            config.max_rows - len(selected_positions)
        )

        if remaining_count:
            selected_positions.extend(
                generator.choice(
                    remaining_positions,
                    size=remaining_count,
                    replace=False,
                ).astype(int).tolist()
            )

        positions = np.asarray(
            sorted(selected_positions),
            dtype=int,
        )

    else:
        positions = np.sort(
            generator.choice(
                all_positions,
                size=config.max_rows,
                replace=False,
            )
        )

    return (
        X_holdout.iloc[positions].copy(deep=True),
        y_holdout.iloc[positions].copy(deep=True),
        True,
    )


def _selected_feature_names(
    pipeline: Pipeline,
    report: dict,
) -> list[str]:
    reported_names = report.get(
        "feature_selection",
        {},
    ).get("selected_feature_names")

    if isinstance(reported_names, list):
        return [str(name) for name in reported_names]

    preprocessor = pipeline.named_steps.get("preprocessor")
    variance_filter = pipeline.named_steps.get("variance_filter")
    feature_selector = pipeline.named_steps.get("feature_selector")

    if any(
        step is None
        for step in (
            preprocessor,
            variance_filter,
            feature_selector,
        )
    ):
        return []

    transformed_names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=object,
    )
    after_variance = transformed_names[
        variance_filter.get_support()
    ]
    selected_names = after_variance[
        feature_selector.get_support()
    ]
    return [str(name) for name in selected_names.tolist()]


def _native_importance(
    pipeline: Pipeline,
    report: dict,
    top_n: int,
) -> dict[str, Any]:
    model = pipeline.named_steps.get("model")
    selected_names = _selected_feature_names(pipeline, report)

    if model is None:
        return {
            "status": "unavailable",
            "method": None,
            "reason": "Pipeline does not contain a model step.",
            "features": [],
        }

    signed_values = None

    if hasattr(model, "feature_importances_"):
        method = "feature_importances"
        importance_values = np.asarray(
            model.feature_importances_,
            dtype=float,
        ).reshape(-1)

    elif hasattr(model, "coef_"):
        method = "absolute_coefficients"
        coefficients = np.asarray(model.coef_, dtype=float)

        if coefficients.ndim == 1:
            signed_values = coefficients
            importance_values = np.abs(coefficients)
        elif coefficients.shape[0] == 1:
            signed_values = coefficients[0]
            importance_values = np.abs(coefficients[0])
        else:
            importance_values = np.mean(
                np.abs(coefficients),
                axis=0,
            )

    else:
        return {
            "status": "unavailable",
            "method": None,
            "reason": (
                "Selected estimator does not expose coefficients "
                "or feature_importances_."
            ),
            "features": [],
        }

    if len(importance_values) != len(selected_names):
        return {
            "status": "unavailable",
            "method": method,
            "reason": (
                "Native importance length does not match selected "
                "feature names."
            ),
            "features": [],
        }

    total_importance = float(np.sum(np.abs(importance_values)))
    entries = []

    for index, (name, value) in enumerate(
        zip(selected_names, importance_values, strict=True)
    ):
        entry = {
            "feature": name,
            "importance": _safe_float(value),
            "normalized_importance": _safe_float(
                abs(value) / total_importance
                if total_importance > 0
                else 0.0
            ),
        }

        if signed_values is not None:
            signed_value = float(signed_values[index])
            entry["signed_coefficient"] = _safe_float(
                signed_value
            )
            entry["effect_direction"] = (
                "positive"
                if signed_value > 0
                else "negative"
                if signed_value < 0
                else "neutral"
            )

        entries.append(entry)

    entries.sort(
        key=lambda item: item["importance"] or 0.0,
        reverse=True,
    )

    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    return {
        "status": "complete",
        "method": method,
        "reason": None,
        "available_feature_count": len(entries),
        "reported_feature_count": min(len(entries), top_n),
        "features": entries[:top_n],
    }


def generate_model_explanations(
    training_result: dict[str, Any],
    *,
    config: ExplainabilityConfig | None = None,
) -> dict[str, Any]:
    """Explain a fitted model using untouched holdout rows."""

    config = config or ExplainabilityConfig()
    config.validate()
    pipeline, report, X_holdout, y_holdout = _validate_inputs(
        training_result
    )
    problem_type = report.get("problem_type")

    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    source_X = X_holdout.copy(deep=True)
    source_y = y_holdout.copy(deep=True)
    X_evaluation, y_evaluation, sampling_applied = (
        _sample_evaluation_rows(
            X_holdout,
            y_holdout,
            problem_type,
            config,
        )
    )

    if problem_type == "classification":
        scoring = "balanced_accuracy"
        metric_name = "balanced_accuracy"
        metric_direction = "higher"
        baseline_value = balanced_accuracy_score(
            y_evaluation,
            pipeline.predict(X_evaluation),
        )
        importance_unit = "balanced_accuracy_decrease"
    else:
        scoring = "neg_root_mean_squared_error"
        metric_name = "rmse"
        metric_direction = "lower"
        baseline_value = root_mean_squared_error(
            y_evaluation,
            pipeline.predict(X_evaluation),
        )
        importance_unit = "rmse_increase"

    raw_importance = permutation_importance(
        pipeline,
        X_evaluation,
        y_evaluation,
        scoring=scoring,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    entries = []

    for name, mean, standard_deviation in zip(
        X_evaluation.columns,
        raw_importance.importances_mean,
        raw_importance.importances_std,
        strict=True,
    ):
        mean_value = float(mean)
        entries.append(
            {
                "feature": str(name),
                "importance_mean": _safe_float(mean_value),
                "importance_standard_deviation": _safe_float(
                    standard_deviation
                ),
                "importance_direction": (
                    "positive"
                    if mean_value > 0
                    else "negative"
                    if mean_value < 0
                    else "neutral"
                ),
            }
        )

    entries.sort(
        key=lambda item: item["importance_mean"] or 0.0,
        reverse=True,
    )

    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    explanation = {
        "explainability_status": "complete",
        "target_column": report.get("target_column"),
        "problem_type": problem_type,
        "model_name": report.get("best_model", {}).get(
            "model_name"
        ),
        "model_display_name": report.get("best_model", {}).get(
            "display_name"
        ),
        "evaluation_scope": "untouched_holdout",
        "used_for_model_selection": False,
        "configuration": config.to_dict(),
        "evaluation_rows_available": len(X_holdout),
        "evaluation_rows_used": len(X_evaluation),
        "sampling_applied": sampling_applied,
        "baseline_metric": {
            "name": metric_name,
            "value": _safe_float(baseline_value),
            "direction": metric_direction,
        },
        "permutation_importance": {
            "method": "model_agnostic_permutation",
            "scoring": scoring,
            "importance_unit": importance_unit,
            "available_feature_count": len(entries),
            "reported_feature_count": min(
                len(entries),
                config.top_n,
            ),
            "features": entries[: config.top_n],
            "interpretation": (
                "Positive values mean that shuffling the raw feature "
                "worsened holdout performance. Negative values can "
                "indicate noise, redundancy, or sampling variation."
            ),
        },
        "native_importance": _native_importance(
            pipeline,
            report,
            config.top_n,
        ),
        "source_data_modified": not (
            X_holdout.equals(source_X)
            and y_holdout.equals(source_y)
        ),
    }

    return explanation
