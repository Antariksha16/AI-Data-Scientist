import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import backend.app.pipeline.training as training_module
from backend.app.pipeline.model_recommender import (
    describe_user_model_override,
    recommend_models,
)
from backend.app.pipeline.model_registry import get_model_registry
from backend.app.pipeline.training import train_and_compare_models
from backend.app.pipeline.training_config import TrainingConfig


def _classification_dataframe(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)

    return pd.DataFrame(
        {
            "RecordID": range(1, rows + 1),
            "Score": signal,
            "Balance": rng.normal(100, 25, size=rows),
            "Region": rng.choice(["North", "South", "West"], rows),
            "Target": target,
        }
    )


def _fast_config(**overrides) -> TrainingConfig:
    values = {
        "cv_folds": 2,
        "feature_selection_percentile": 100,
        "minimum_rows": 20,
        "max_transformed_features": 1_000,
        "max_recommended_models": 2,
        "n_jobs": 1,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _direct_inputs(rows: int = 100):
    X = pd.DataFrame(
        {
            "Amount": np.linspace(1.0, 100.0, rows),
            "Segment": ["A", "B"] * (rows // 2),
        }
    )
    y = pd.Series([0, 1] * (rows // 2), name="Target")
    metadata = {
        "numerical_features": ["Amount"],
        "categorical_features": ["Segment"],
        "robust_scaler_features": ["Amount"],
    }
    return X, y, metadata


def test_automatic_recommendation_is_bounded_diverse_and_json_safe():
    X, y, metadata = _direct_inputs()
    specs = get_model_registry("classification")

    first = recommend_models(
        specs,
        X,
        y,
        "classification",
        metadata,
        estimated_transformed_features=3,
        max_models=2,
    )
    second = recommend_models(
        specs,
        X,
        y,
        "classification",
        metadata,
        estimated_transformed_features=3,
        max_models=2,
    )

    assert first == second
    assert first["selection_mode"] == "automatic_recommendation"
    assert first["selection_scope"] == "training_split_only"
    assert first["holdout_data_used"] is False
    assert first["selected_model_count"] == 2
    assert len(first["selected_families"]) == 2
    assert first["baseline_model"] == "dummy_classifier"
    assert first["target_values_stored"] is False
    assert first["dataset_profile"]["target_values_stored"] is False
    json.dumps(first, allow_nan=False)


def test_regression_profile_contains_only_aggregate_target_context():
    X, _, metadata = _direct_inputs()
    y = pd.Series(np.linspace(10.0, 500.0, len(X)), name="AmountDue")

    report = recommend_models(
        get_model_registry("regression"),
        X,
        y,
        "regression",
        metadata,
        estimated_transformed_features=3,
        max_models=3,
    )
    profile = report["dataset_profile"]

    assert profile["class_imbalance_status"] == "not_applicable"
    assert profile["target_skewness"] is not None
    assert "target_values" not in profile
    assert report["selected_model_count"] == 3
    json.dumps(report, allow_nan=False)


def test_model_eligibility_limits_are_enforced_and_explained():
    X, y, metadata = _direct_inputs()
    specs = get_model_registry("classification")
    limited_specs = [
        replace(spec, max_recommended_rows=50)
        if spec.name == "random_forest"
        else spec
        for spec in specs
    ]

    report = recommend_models(
        limited_specs,
        X,
        y,
        "classification",
        metadata,
        estimated_transformed_features=3,
        max_models=3,
    )
    decision = next(
        item
        for item in report["model_decisions"]
        if item["model_name"] == "random_forest"
    )

    assert decision["eligibility"] == "ineligible"
    assert decision["decision"] == "skipped"
    assert decision["reasons"] == [
        "row_count_exceeds_model_recommendation_limit"
    ]


def test_extreme_class_imbalance_is_detected_and_influences_scores():
    X, _, metadata = _direct_inputs()
    y = pd.Series([0] * 96 + [1] * 4, name="Target")

    report = recommend_models(
        get_model_registry("classification"),
        X,
        y,
        "classification",
        metadata,
        estimated_transformed_features=3,
        max_models=2,
    )

    assert (
        report["dataset_profile"]["class_imbalance_status"]
        == "extreme"
    )
    assert report["dataset_profile"]["minority_class_ratio"] == 0.04
    for decision in report["model_decisions"]:
        if decision["decision"] == "selected":
            assert "configured_for_class_imbalance" in decision["reasons"]


def test_user_override_is_preserved_and_fully_explained():
    X, y, metadata = _direct_inputs()
    all_specs = get_model_registry("classification")
    selected_specs = get_model_registry(
        "classification",
        requested_models=["random_forest"],
    )

    report = describe_user_model_override(
        all_specs,
        selected_specs,
        X,
        y,
        "classification",
        metadata,
        estimated_transformed_features=3,
    )

    assert report["selection_mode"] == "user_override"
    assert report["selected_models"] == ["random_forest"]
    assert report["selected_families"] == ["bagging"]
    assert report["selected_model_count"] == 1
    selected = next(
        item
        for item in report["model_decisions"]
        if item["model_name"] == "random_forest"
    )
    assert selected["reasons"] == ["explicitly_requested_by_user"]


def test_invalid_recommendation_inputs_are_rejected():
    X, y, metadata = _direct_inputs()
    specs = get_model_registry("classification")

    with pytest.raises(ValueError, match="between 1 and 8"):
        recommend_models(
            specs,
            X,
            y,
            "classification",
            metadata,
            estimated_transformed_features=3,
            max_models=9,
        )

    with pytest.raises(ValueError, match="equal rows"):
        recommend_models(
            specs,
            X,
            y.iloc[:-1],
            "classification",
            metadata,
            estimated_transformed_features=3,
        )


def test_training_automatically_trains_only_recommended_models():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(max_recommended_models=2),
        requested_models=None,
    )
    report = result["report"]
    recommendation = report["model_recommendation"]
    compared_names = {
        item["model_name"] for item in report["model_comparisons"]
    }

    assert recommendation["selection_mode"] == "automatic_recommendation"
    assert recommendation["selected_model_count"] == 2
    assert compared_names == {
        recommendation["baseline_model"],
        *recommendation["selected_models"],
    }
    assert len(compared_names) == 3


def test_training_explicit_models_bypass_automatic_shortlisting():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        requested_models=["random_forest"],
    )
    recommendation = result["report"]["model_recommendation"]

    assert recommendation["selection_mode"] == "user_override"
    assert recommendation["selected_models"] == ["random_forest"]
    assert {
        item["model_name"]
        for item in result["report"]["model_comparisons"]
    } == {"dummy_classifier", "random_forest"}


def test_recommender_receives_training_rows_only(monkeypatch):
    df = _classification_dataframe()
    config = _fast_config(max_recommended_models=1)
    expected_train, _ = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
        stratify=df["Target"],
    )
    captured_indices = set()
    original_recommend = training_module.recommend_models

    def capture_recommend(model_specs, X_train, y_train, *args, **kwargs):
        captured_indices.update(X_train.index.tolist())
        return original_recommend(
            model_specs,
            X_train,
            y_train,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        training_module,
        "recommend_models",
        capture_recommend,
    )

    result = train_and_compare_models(
        df,
        "Target",
        config=config,
        requested_models=None,
    )

    assert captured_indices == set(expected_train.index)
    assert len(captured_indices) == result["report"]["split"][
        "training_rows"
    ]

