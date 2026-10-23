import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import ml_pipeline.training as training_module
from ml_pipeline.model_registry import get_model_registry
from ml_pipeline.training import train_and_compare_models
from ml_pipeline.training_config import TrainingConfig


def _classification_dataframe(
    rows: int = 120,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=rows)
    noise = rng.normal(scale=1.4, size=rows)
    target = (signal + noise > 0).astype(int)

    # Guarantee that both classes remain well represented even if
    # a future NumPy random implementation changes slightly.
    target[:20] = np.array([0, 1] * 10)

    age = 35 + signal * 8
    age[::17] = np.nan

    return pd.DataFrame(
        {
            "RecordID": range(1, rows + 1),
            "Age": age,
            "IncomeScore": rng.normal(50, 12, size=rows),
            "City": rng.choice(
                ["North", "South", "West"],
                size=rows,
            ),
            "Target": target,
        }
    )


def _regression_dataframe(
    rows: int = 120,
) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    experience = rng.integers(0, 16, size=rows)
    performance = rng.normal(60, 10, size=rows)
    salary = (
        30_000
        + experience * 1_500
        + performance * 150
        + rng.normal(0, 5_000, size=rows)
    )

    return pd.DataFrame(
        {
            "EmployeeID": range(1, rows + 1),
            "Experience": experience,
            "Performance": performance,
            "Department": rng.choice(
                ["IT", "HR", "Sales", "Finance"],
                size=rows,
            ),
            "Salary": salary,
        }
    )


def _fast_config(**overrides) -> TrainingConfig:
    values = {
        "test_size": 0.20,
        "cv_folds": 4,
        "random_state": 42,
        "feature_selection_percentile": 75,
        "minimum_rows": 20,
        "max_transformed_features": 1_000,
        "decimal_places": 6,
        "n_jobs": 1,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_training_config_validation():
    TrainingConfig().validate()

    with pytest.raises(ValueError, match="test_size"):
        TrainingConfig(test_size=1.0).validate()

    with pytest.raises(ValueError, match="cv_folds"):
        TrainingConfig(cv_folds=1).validate()

    with pytest.raises(
        ValueError,
        match="feature_selection_percentile",
    ):
        TrainingConfig(
            feature_selection_percentile=0
        ).validate()

    with pytest.raises(ValueError, match="n_jobs"):
        TrainingConfig(n_jobs=0).validate()


def test_model_registry_is_problem_type_specific():
    classification_specs = get_model_registry(
        "classification",
        requested_models=["logistic_regression"],
    )
    regression_specs = get_model_registry(
        "regression",
        requested_models=["ridge"],
    )

    assert [
        spec.name
        for spec in classification_specs
    ] == ["dummy_classifier", "logistic_regression"]

    assert [
        spec.name
        for spec in regression_specs
    ] == ["dummy_regressor", "ridge"]

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_registry(
            "classification",
            requested_models=["ridge"],
        )


def test_classification_training_is_stratified_and_json_safe():
    df = _classification_dataframe()
    original = df.copy(deep=True)

    result = train_and_compare_models(
        df,
        "Target",
        config=_fast_config(),
        requested_models=["logistic_regression"],
    )
    report = result["report"]

    assert report["training_status"] == "complete"
    assert report["problem_type"] == "classification"
    assert report["problem_type_source"] == "inferred"
    assert report["split"]["stratified"] is True
    assert (
        report["split"]["preprocessing_decision_scope"]
        == "training_split_only"
    )
    assert report["cross_validation"]["used_folds"] == 4
    assert report["primary_metric"] == "balanced_accuracy"
    assert (
        report["best_model"]["model_name"]
        == "logistic_regression"
    )
    assert report["feature_selection"][
        "selected_feature_count"
    ] > 0
    assert report["holdout_metrics"]["accuracy"] is not None
    assert report["source_dataframe_modified"] is False
    assert isinstance(result["pipeline"], Pipeline)

    json.dumps(report, allow_nan=False)
    assert_frame_equal(df, original)


def test_regression_training_uses_regression_metrics():
    df = _regression_dataframe()

    result = train_and_compare_models(
        df,
        "Salary",
        config=_fast_config(),
        requested_models=["ridge"],
    )
    report = result["report"]

    assert report["training_status"] == "complete"
    assert report["problem_type"] == "regression"
    assert report["split"]["stratified"] is False
    assert report["primary_metric"] == "rmse"
    assert report["primary_metric_direction"] == "lower"
    assert report["best_model"]["model_name"] == "ridge"
    assert report["holdout_metrics"]["rmse"] is not None
    assert report["holdout_metrics"]["mae"] is not None
    assert report["holdout_metrics"]["r2"] is not None
    assert (
        report["feature_selection"]["method"]
        == "SelectPercentile(f_regression)"
    )

    json.dumps(report, allow_nan=False)


def test_preprocessing_decisions_receive_only_training_rows(
    monkeypatch,
):
    df = _classification_dataframe()
    config = _fast_config()
    expected_train, _ = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
        stratify=df["Target"],
    )
    captured_indices = set()
    original_prepare = (
        training_module.prepare_dataset_for_modeling
    )

    def capture_prepare(input_df, *args, **kwargs):
        captured_indices.update(input_df.index.tolist())
        return original_prepare(input_df, *args, **kwargs)

    monkeypatch.setattr(
        training_module,
        "prepare_dataset_for_modeling",
        capture_prepare,
    )

    result = train_and_compare_models(
        df,
        "Target",
        config=config,
        requested_models=["logistic_regression"],
    )

    assert captured_indices == set(expected_train.index)
    assert len(captured_indices) == result["report"]["split"][
        "training_rows"
    ]
    assert len(captured_indices) < len(df)


def test_missing_target_rows_are_removed_before_split():
    df = _classification_dataframe()
    df.loc[[2, 11, 39], "Target"] = np.nan

    result = train_and_compare_models(
        df,
        "Target",
        config=_fast_config(),
        requested_models=["logistic_regression"],
    )
    split = result["report"]["split"]

    assert split["original_rows"] == 120
    assert split["usable_rows"] == 117
    assert split["missing_target_rows_removed"] == 3
    assert split["training_rows"] + split["test_rows"] == 117


def test_cross_validation_folds_reduce_for_rare_class():
    rows = 60
    df = pd.DataFrame(
        {
            "Measurement": np.linspace(0, 10, rows),
            "Group": ["A", "B", "C"] * 20,
            "Target": [1] * 6 + [0] * 54,
        }
    )

    result = train_and_compare_models(
        df,
        "Target",
        config=_fast_config(cv_folds=10),
        requested_models=["logistic_regression"],
    )

    used_folds = result["report"]["cross_validation"][
        "used_folds"
    ]
    assert 2 <= used_folds < 10


def test_regression_override_rejects_text_target():
    df = _classification_dataframe()
    df["Target"] = df["Target"].map(
        {0: "No", 1: "Yes"}
    )

    with pytest.raises(
        ValueError,
        match="regression target must contain numerical",
    ):
        train_and_compare_models(
            df,
            "Target",
            problem_type="regression",
            config=_fast_config(),
            requested_models=["ridge"],
        )


def test_classification_requires_splittable_classes():
    df = _classification_dataframe(rows=40)
    df["Target"] = 0
    df.loc[0, "Target"] = 1

    with pytest.raises(
        ValueError,
        match="at least two rows",
    ):
        train_and_compare_models(
            df,
            "Target",
            problem_type="classification",
            config=_fast_config(),
            requested_models=["logistic_regression"],
        )


def test_transformed_feature_safety_limit_blocks_training():
    df = _regression_dataframe()

    with pytest.raises(
        ValueError,
        match="transformed feature count",
    ):
        train_and_compare_models(
            df,
            "Salary",
            config=_fast_config(
                max_transformed_features=2
            ),
            requested_models=["ridge"],
        )
