import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ml_pipeline.explainability_config import (
    ExplainabilityConfig,
)
from ml_pipeline.model_explainability import (
    generate_model_explanations,
)
from ml_pipeline.training import train_and_compare_models
from tests.test_training import (
    _classification_dataframe,
    _fast_config,
    _regression_dataframe,
)


def _classification_result(rows=120):
    return train_and_compare_models(
        _classification_dataframe(rows=rows),
        "Target",
        config=_fast_config(),
        requested_models=["logistic_regression"],
    )


def _regression_result(rows=120):
    return train_and_compare_models(
        _regression_dataframe(rows=rows),
        "Salary",
        config=_fast_config(),
        requested_models=["ridge"],
    )


def test_training_returns_private_untouched_holdout_context():
    result = _classification_result()
    context = result["evaluation_context"]
    report = result["report"]

    assert isinstance(context["X_holdout"], pd.DataFrame)
    assert isinstance(context["y_holdout"], pd.Series)
    assert len(context["X_holdout"]) == report["split"][
        "test_rows"
    ]
    assert context["evaluation_scope"] == "untouched_holdout"
    assert context["used_for_model_selection"] is False
    assert "evaluation_context" not in report
    json.dumps(report, allow_nan=False)


def test_classification_permutation_and_coefficients_are_json_safe():
    result = _classification_result()
    context = result["evaluation_context"]
    original_X = context["X_holdout"].copy(deep=True)
    original_y = context["y_holdout"].copy(deep=True)

    explanation = generate_model_explanations(
        result,
        config=ExplainabilityConfig(
            n_repeats=3,
            top_n=10,
        ),
    )

    assert explanation["explainability_status"] == "complete"
    assert explanation["problem_type"] == "classification"
    assert explanation["evaluation_scope"] == "untouched_holdout"
    assert explanation["used_for_model_selection"] is False
    assert explanation["baseline_metric"]["name"] == (
        "balanced_accuracy"
    )
    assert explanation["permutation_importance"]["features"]
    assert explanation["native_importance"]["status"] == "complete"
    assert explanation["native_importance"]["method"] == (
        "absolute_coefficients"
    )
    assert explanation["source_data_modified"] is False
    json.dumps(explanation, allow_nan=False)
    assert_frame_equal(context["X_holdout"], original_X)
    assert_series_equal(context["y_holdout"], original_y)


def test_regression_explanation_uses_rmse_increase():
    result = _regression_result()
    explanation = generate_model_explanations(
        result,
        config=ExplainabilityConfig(n_repeats=3),
    )

    assert explanation["problem_type"] == "regression"
    assert explanation["baseline_metric"]["name"] == "rmse"
    assert explanation["baseline_metric"]["direction"] == "lower"
    assert explanation["permutation_importance"][
        "importance_unit"
    ] == "rmse_increase"
    assert explanation["native_importance"]["method"] == (
        "absolute_coefficients"
    )
    top_raw_feature = explanation["permutation_importance"][
        "features"
    ][0]["feature"]
    assert top_raw_feature in {"Experience", "Performance"}
    json.dumps(explanation, allow_nan=False)


def test_tree_model_exposes_selected_feature_importances():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(cv_folds=2),
        requested_models=["random_forest"],
    )
    explanation = generate_model_explanations(
        result,
        config=ExplainabilityConfig(n_repeats=2),
    )
    native = explanation["native_importance"]

    assert native["status"] == "complete"
    assert native["method"] == "feature_importances"
    assert native["available_feature_count"] == result["report"][
        "feature_selection"
    ]["selected_feature_count"]
    assert all(
        feature["normalized_importance"] is not None
        for feature in native["features"]
    )


def test_holdout_sampling_is_bounded_and_deterministic():
    result = _classification_result(rows=300)
    config = ExplainabilityConfig(
        n_repeats=2,
        max_rows=20,
        top_n=3,
        random_state=19,
    )

    first = generate_model_explanations(result, config=config)
    second = generate_model_explanations(result, config=config)

    assert first["sampling_applied"] is True
    assert first["evaluation_rows_used"] == 20
    assert first["permutation_importance"][
        "reported_feature_count"
    ] == 3
    assert first == second


def test_explainability_config_validation():
    ExplainabilityConfig().validate()

    with pytest.raises(ValueError, match="n_repeats"):
        ExplainabilityConfig(n_repeats=0).validate()

    with pytest.raises(ValueError, match="max_rows"):
        ExplainabilityConfig(max_rows=9).validate()

    with pytest.raises(ValueError, match="top_n"):
        ExplainabilityConfig(top_n=0).validate()

    with pytest.raises(ValueError, match="n_jobs"):
        ExplainabilityConfig(n_jobs=0).validate()


def test_missing_or_invalid_evaluation_context_is_rejected():
    result = _classification_result()
    missing_context = dict(result)
    missing_context.pop("evaluation_context")

    with pytest.raises(ValueError, match="evaluation context"):
        generate_model_explanations(missing_context)

    invalid_scope = dict(result)
    invalid_scope["evaluation_context"] = dict(
        result["evaluation_context"]
    )
    invalid_scope["evaluation_context"][
        "used_for_model_selection"
    ] = True

    with pytest.raises(ValueError, match="must not be used"):
        generate_model_explanations(invalid_scope)


def test_explanation_does_not_enter_exported_training_report():
    result = _regression_result()
    explanation = generate_model_explanations(
        result,
        config=ExplainabilityConfig(n_repeats=2),
    )

    assert "evaluation_context" not in result["report"]
    assert "X_holdout" not in result["report"]
    assert explanation["evaluation_rows_available"] == result[
        "report"
    ]["split"]["test_rows"]
    json.dumps(result["report"], allow_nan=False)
