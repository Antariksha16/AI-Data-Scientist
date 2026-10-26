import json

import pytest

import ml_pipeline.training as training_module
from ml_pipeline.hyperparameter_tuning import (
    get_hyperparameter_search_space,
)
from ml_pipeline.training import train_and_compare_models
from ml_pipeline.tuning_config import TuningConfig
from tests.test_training import (
    _classification_dataframe,
    _fast_config,
    _regression_dataframe,
)


def _fast_tuning_config(**overrides) -> TuningConfig:
    values = {
        "n_iter": 3,
        "top_models": 1,
        "random_state": 42,
        "n_jobs": 1,
    }
    values.update(overrides)
    return TuningConfig(**values)


def test_tuning_config_validation():
    TuningConfig().validate()

    with pytest.raises(ValueError, match="n_iter"):
        TuningConfig(n_iter=0).validate()

    with pytest.raises(ValueError, match="top_models"):
        TuningConfig(top_models=0).validate()

    with pytest.raises(ValueError, match="n_jobs"):
        TuningConfig(n_jobs=0).validate()


def test_search_spaces_include_selector_and_model_parameters():
    classification_space = get_hyperparameter_search_space(
        "classification",
        "logistic_regression",
    )
    regression_space = get_hyperparameter_search_space(
        "regression",
        "ridge",
    )

    assert (
        "feature_selector__percentile"
        in classification_space
    )
    assert "model__C" in classification_space
    assert "feature_selector__percentile" in regression_space
    assert "model__alpha" in regression_space

    with pytest.raises(ValueError, match="No hyperparameter"):
        get_hyperparameter_search_space(
            "classification",
            "ridge",
        )


def test_classification_tuning_is_reported_and_json_safe():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        tuning_config=_fast_tuning_config(),
        requested_models=["logistic_regression"],
    )
    report = result["report"]
    tuning = report["hyperparameter_tuning"]
    candidate = tuning["candidates"][0]

    assert tuning["status"] == "complete"
    assert tuning["attempted_model_count"] == 1
    assert tuning["successful_model_count"] == 1
    assert candidate["model_name"] == "logistic_regression"
    assert candidate["status"] == "success"
    assert candidate["evaluated_parameter_sets"] == 3
    assert "model__C" in candidate["best_parameters"]
    assert "feature_selector__percentile" in (
        candidate["best_parameters"]
    )
    assert report["best_model"]["selected_variant"] in {
        "untuned",
        "tuned",
    }
    assert report["holdout_metrics"]["accuracy"] is not None

    json.dumps(report, allow_nan=False)


def test_regression_tuning_uses_lower_rmse_direction():
    result = train_and_compare_models(
        _regression_dataframe(),
        "Salary",
        config=_fast_config(),
        tuning_config=_fast_tuning_config(),
        requested_models=["ridge"],
    )
    report = result["report"]
    candidate = report["hyperparameter_tuning"][
        "candidates"
    ][0]

    assert report["problem_type"] == "regression"
    assert report["primary_metric_direction"] == "lower"
    assert candidate["model_name"] == "ridge"
    assert candidate["tuned_primary_score"] is not None
    assert "model__alpha" in candidate["best_parameters"]
    assert report["holdout_metrics"]["rmse"] is not None


def test_tuning_is_deterministic_with_fixed_random_state():
    arguments = {
        "df": _classification_dataframe(),
        "target_column": "Target",
        "config": _fast_config(),
        "tuning_config": _fast_tuning_config(n_iter=2),
        "requested_models": ["logistic_regression"],
    }

    first = train_and_compare_models(**arguments)["report"]
    second = train_and_compare_models(**arguments)["report"]
    first_candidate = first["hyperparameter_tuning"][
        "candidates"
    ][0]
    second_candidate = second["hyperparameter_tuning"][
        "candidates"
    ][0]

    assert (
        first_candidate["best_parameters"]
        == second_candidate["best_parameters"]
    )
    assert (
        first_candidate["tuned_primary_score"]
        == second_candidate["tuned_primary_score"]
    )
    assert first["holdout_metrics"] == second[
        "holdout_metrics"
    ]


def test_tuning_failure_falls_back_to_untuned_model(
    monkeypatch,
):
    def fail_tuning(**kwargs):
        del kwargs
        raise RuntimeError("synthetic tuning failure")

    monkeypatch.setattr(
        training_module,
        "tune_model_pipeline",
        fail_tuning,
    )

    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        tuning_config=_fast_tuning_config(),
        requested_models=["logistic_regression"],
    )
    report = result["report"]

    assert report["training_status"] == "partial"
    assert (
        report["hyperparameter_tuning"]["status"]
        == "failed_fallback"
    )
    assert report["best_model"]["tuned"] is False
    assert (
        report["best_model"]["selected_variant"]
        == "untuned"
    )
    assert report["holdout_metrics"]["accuracy"] is not None


def test_holdout_evaluation_occurs_after_tuning(monkeypatch):
    events = []
    original_tune = training_module.tune_model_pipeline
    original_evaluate = training_module._evaluate_holdout

    def record_tuning(**kwargs):
        events.append("tuning")
        return original_tune(**kwargs)

    def record_evaluation(*args, **kwargs):
        events.append("holdout")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        training_module,
        "tune_model_pipeline",
        record_tuning,
    )
    monkeypatch.setattr(
        training_module,
        "_evaluate_holdout",
        record_evaluation,
    )

    train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        tuning_config=_fast_tuning_config(n_iter=1),
        requested_models=["logistic_regression"],
    )

    assert events == ["tuning", "holdout"]