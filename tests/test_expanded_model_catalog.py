import json

import numpy as np
import pandas as pd

from ml_pipeline.hyperparameter_tuning import (
    get_hyperparameter_search_space,
)
from ml_pipeline.model_recommender import recommend_models
from ml_pipeline.model_registry import get_model_registry
from ml_pipeline.training import train_and_compare_models
from ml_pipeline.training_config import TrainingConfig
from ml_pipeline.tuning_config import TuningConfig


CLASSIFICATION_MODELS = {
    "logistic_regression",
    "ridge_classifier",
    "sgd_classifier",
    "linear_svc",
    "rbf_svc",
    "knn_classifier",
    "gaussian_nb",
    "linear_discriminant_analysis",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "ada_boost",
    "gradient_boosting",
    "hist_gradient_boosting",
}

REGRESSION_MODELS = {
    "ridge",
    "lasso",
    "elastic_net",
    "sgd_regressor",
    "huber_regressor",
    "linear_svr",
    "rbf_svr",
    "knn_regressor",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "ada_boost",
    "gradient_boosting",
    "hist_gradient_boosting",
}


def _classification_dataframe(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.9, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)

    return pd.DataFrame(
        {
            "Age": rng.integers(18, 75, size=rows),
            "Signal": signal,
            "Balance": rng.normal(80, 20, size=rows),
            "Region": rng.choice(["North", "South", "West"], rows),
            "Target": target,
        }
    )


def _regression_dataframe(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    experience = rng.integers(0, 20, size=rows)
    performance = rng.normal(60, 12, size=rows)
    target = (
        20_000
        + experience * 1_300
        + performance * 180
        + rng.normal(0, 4_000, size=rows)
    )

    return pd.DataFrame(
        {
            "Experience": experience,
            "Performance": performance,
            "Department": rng.choice(["IT", "HR", "Sales"], rows),
            "Salary": target,
        }
    )


def _fast_config(**overrides) -> TrainingConfig:
    values = {
        "cv_folds": 2,
        "feature_selection_percentile": 100,
        "minimum_rows": 20,
        "max_transformed_features": 2_000,
        "max_recommended_models": 4,
        "n_jobs": 1,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _non_baseline_names(problem_type: str) -> set[str]:
    return {
        spec.name
        for spec in get_model_registry(problem_type)
        if not spec.is_baseline
    }


def test_registry_contains_exactly_28_classical_candidates():
    assert _non_baseline_names("classification") == CLASSIFICATION_MODELS
    assert _non_baseline_names("regression") == REGRESSION_MODELS
    assert len(CLASSIFICATION_MODELS) == 14
    assert len(REGRESSION_MODELS) == 14


def test_every_model_has_complete_recommendation_metadata():
    supported_costs = {"low", "medium", "high"}

    for problem_type in ("classification", "regression"):
        specs = get_model_registry(problem_type)
        assert sum(spec.is_baseline for spec in specs) == 1

        for spec in specs:
            assert spec.name
            assert spec.display_name
            assert spec.family
            assert spec.computational_cost in supported_costs
            if spec.is_baseline:
                assert spec.family == "baseline"


def test_every_candidate_has_a_valid_bounded_tuning_space():
    for problem_type in ("classification", "regression"):
        for spec in get_model_registry(problem_type):
            if spec.is_baseline:
                continue

            space = get_hyperparameter_search_space(
                problem_type,
                spec.name,
            )
            estimator_parameters = spec.estimator.get_params()

            assert "feature_selector__percentile" in space
            assert all(values for values in space.values())

            for parameter_name in space:
                if parameter_name.startswith("model__"):
                    estimator_name = parameter_name.removeprefix(
                        "model__"
                    )
                    assert estimator_name in estimator_parameters


def test_every_classification_candidate_trains_successfully():
    df = _classification_dataframe()

    for model_name in sorted(CLASSIFICATION_MODELS):
        result = train_and_compare_models(
            df,
            "Target",
            problem_type="classification",
            config=_fast_config(),
            requested_models=[model_name],
        )
        comparison = next(
            item
            for item in result["report"]["model_comparisons"]
            if item["model_name"] == model_name
        )
        assert comparison["status"] == "success", model_name
        json.dumps(result["report"], allow_nan=False)


def test_every_regression_candidate_trains_successfully():
    df = _regression_dataframe()

    for model_name in sorted(REGRESSION_MODELS):
        result = train_and_compare_models(
            df,
            "Salary",
            problem_type="regression",
            config=_fast_config(),
            requested_models=[model_name],
        )
        comparison = next(
            item
            for item in result["report"]["model_comparisons"]
            if item["model_name"] == model_name
        )
        assert comparison["status"] == "success", model_name
        json.dumps(result["report"], allow_nan=False)


def test_automatic_classification_uses_four_of_fourteen_models():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        requested_models=None,
    )
    report = result["report"]
    recommendation = report["model_recommendation"]

    assert recommendation["registered_non_baseline_model_count"] == 14
    assert recommendation["selected_model_count"] == 4
    assert set(recommendation["selected_families"]) == {
        "linear",
        "kernel",
        "bagging",
        "boosting",
    }
    assert len(report["model_comparisons"]) == 5


def test_automatic_regression_uses_four_of_fourteen_models():
    result = train_and_compare_models(
        _regression_dataframe(),
        "Salary",
        problem_type="regression",
        config=_fast_config(),
        requested_models=None,
    )
    report = result["report"]
    recommendation = report["model_recommendation"]

    assert recommendation["registered_non_baseline_model_count"] == 14
    assert recommendation["selected_model_count"] == 4
    assert set(recommendation["selected_families"]) == {
        "linear",
        "kernel",
        "bagging",
        "boosting",
    }
    assert len(report["model_comparisons"]) == 5


def test_large_dataset_excludes_expensive_candidates():
    rows = 150_001
    X = pd.DataFrame(
        {
            "Amount": np.arange(rows, dtype=float),
            "Segment": np.where(np.arange(rows) % 2, "A", "B"),
        }
    )
    y = pd.Series(np.arange(rows) % 2, name="Target")
    report = recommend_models(
        get_model_registry("classification"),
        X,
        y,
        "classification",
        {
            "numerical_features": ["Amount"],
            "categorical_features": ["Segment"],
            "robust_scaler_features": [],
        },
        estimated_transformed_features=3,
        max_models=4,
    )
    decisions = {
        item["model_name"]: item
        for item in report["model_decisions"]
    }

    assert report["dataset_profile"]["dataset_size"] == "large"
    for model_name in {
        "rbf_svc",
        "knn_classifier",
        "random_forest",
        "extra_trees",
        "ada_boost",
        "gradient_boosting",
    }:
        assert decisions[model_name]["eligibility"] == "ineligible"
        assert decisions[model_name]["decision"] == "skipped"


def test_new_classification_candidate_can_be_tuned():
    result = train_and_compare_models(
        _classification_dataframe(80),
        "Target",
        config=_fast_config(),
        tuning_config=TuningConfig(n_iter=1, top_models=1, n_jobs=1),
        requested_models=["ada_boost"],
    )

    tuning = result["report"]["hyperparameter_tuning"]
    assert tuning["status"] == "complete"
    assert tuning["candidates"][0]["model_name"] == "ada_boost"


def test_new_regression_candidate_can_be_tuned():
    result = train_and_compare_models(
        _regression_dataframe(80),
        "Salary",
        problem_type="regression",
        config=_fast_config(),
        tuning_config=TuningConfig(n_iter=1, top_models=1, n_jobs=1),
        requested_models=["huber_regressor"],
    )

    tuning = result["report"]["hyperparameter_tuning"]
    assert tuning["status"] == "complete"
    assert tuning["candidates"][0]["model_name"] == "huber_regressor"
