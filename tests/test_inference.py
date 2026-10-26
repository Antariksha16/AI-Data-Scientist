import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.inference import (
    InferenceArtifactError,
    InferenceInputError,
    describe_model_input,
    predict_with_model_artifact,
)
from ml_pipeline.model_artifacts import (
    export_model_artifacts,
)
from ml_pipeline.training import train_and_compare_models
from ml_pipeline.training_config import TrainingConfig


def _fast_config() -> TrainingConfig:
    return TrainingConfig(
        cv_folds=2,
        feature_selection_percentile=100,
    )


def _classification_dataframe() -> pd.DataFrame:
    rows = 60
    return pd.DataFrame(
        {
            "RecordID": range(1, rows + 1),
            "Age": [20 + index % 35 for index in range(rows)],
            "Department": ["IT", "HR", "Sales"] * 20,
            "Outcome": [0, 1] * 30,
        }
    )


def _regression_dataframe() -> pd.DataFrame:
    rows = 60
    return pd.DataFrame(
        {
            "Experience": [index % 15 for index in range(rows)],
            "Department": ["A", "B", "C"] * 20,
            "Salary": [
                25_000.5 + index * 731.25
                for index in range(rows)
            ],
        }
    )


@pytest.fixture(scope="module")
def classification_training_result():
    return train_and_compare_models(
        _classification_dataframe(),
        "Outcome",
        config=_fast_config(),
        requested_models=["logistic_regression"],
    )


@pytest.fixture(scope="module")
def regression_training_result():
    return train_and_compare_models(
        _regression_dataframe(),
        "Salary",
        problem_type="regression",
        config=_fast_config(),
        requested_models=["ridge"],
    )


def _export_bundle(
    tmp_path: Path,
    training_result: dict,
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict:
    return export_model_artifacts(
        training_result,
        tmp_path,
        sample_input=dataframe.drop(
            columns=[target_column]
        ).head(10),
    )


def test_new_bundle_contains_value_free_input_schema(
    tmp_path,
    classification_training_result,
):
    export = _export_bundle(
        tmp_path / "schema_bundle",
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )
    schema = export["manifest"]["input_schema"]

    assert schema["schema_version"] == "1.0"
    assert schema["sample_values_stored"] is False
    assert schema["raw_feature_names"] == [
        "RecordID",
        "Age",
        "Department",
    ]
    assert schema["required_features"] == [
        "Age",
        "Department",
    ]
    assert schema["optional_features"] == ["RecordID"]
    assert "Outcome" not in schema["raw_feature_names"]
    json.dumps(export["manifest"], allow_nan=False)


def test_classification_prediction_matches_fitted_pipeline(
    tmp_path,
    classification_training_result,
):
    dataframe = _classification_dataframe()
    export = _export_bundle(
        tmp_path / "classification_bundle",
        classification_training_result,
        dataframe,
        "Outcome",
    )
    raw_input = dataframe.drop(columns=["Outcome"]).head(4)
    records = raw_input[
        ["Age", "Department"]
    ].to_dict(orient="records")
    records_copy = [dict(record) for record in records]

    result = predict_with_model_artifact(
        export["manifest_path"],
        records,
    )
    expected = classification_training_result[
        "pipeline"
    ].predict(raw_input)

    assert result["prediction_status"] == "complete"
    assert result["problem_type"] == "classification"
    assert result["prediction_count"] == 4
    assert result["probability_status"] == "complete"
    assert result["artifact_verification"] == {
        "verification_status": "complete",
        "pipeline_load_verified": True,
    }
    assert [
        row["prediction"]
        for row in result["predictions"]
    ] == expected.tolist()
    assert all(
        sum(
            item["probability"]
            for item in row["probabilities"]
        )
        == pytest.approx(1.0)
        for row in result["predictions"]
    )
    assert records == records_copy
    assert result["source_records_modified"] is False
    json.dumps(result, allow_nan=False)


def test_classification_probabilities_can_be_disabled(
    tmp_path,
    classification_training_result,
):
    dataframe = _classification_dataframe()
    export = _export_bundle(
        tmp_path / "disabled_probability_bundle",
        classification_training_result,
        dataframe,
        "Outcome",
    )
    records = dataframe.drop(columns=["Outcome"]).head(2).to_dict(
        orient="records"
    )

    result = predict_with_model_artifact(
        export["manifest_path"],
        records,
        include_probabilities=False,
    )

    assert result["probability_status"] == "disabled"
    assert result["class_labels"] == []
    assert all(
        "probabilities" not in row
        for row in result["predictions"]
    )


def test_missing_value_and_unseen_category_are_supported(
    tmp_path,
    classification_training_result,
):
    export = _export_bundle(
        tmp_path / "unseen_category_bundle",
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )

    result = predict_with_model_artifact(
        export["manifest_path"],
        [{"Age": None, "Department": "NewDepartment"}],
    )

    assert result["prediction_status"] == "complete"
    assert result["prediction_count"] == 1
    assert result["probability_status"] == "complete"
    json.dumps(result, allow_nan=False)


def test_regression_prediction_is_json_safe(
    tmp_path,
    regression_training_result,
):
    dataframe = _regression_dataframe()
    export = _export_bundle(
        tmp_path / "regression_bundle",
        regression_training_result,
        dataframe,
        "Salary",
    )
    raw_input = dataframe.drop(columns=["Salary"]).head(5)
    records = raw_input[
        ["Experience", "Department"]
    ].to_dict(orient="records")
    result = predict_with_model_artifact(
        export["manifest_path"],
        records,
    )
    expected = regression_training_result[
        "pipeline"
    ].predict(raw_input)

    assert result["problem_type"] == "regression"
    assert result["probability_status"] == "not_applicable"
    assert result["class_labels"] == []
    assert np.allclose(
        [
            row["prediction"]
            for row in result["predictions"]
        ],
        expected,
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("records", "expected_message"),
    [
        (
            [{"Department": "IT"}],
            "missing required features",
        ),
        (
            [
                {
                    "Age": 30,
                    "Department": "IT",
                    "Unknown": "value",
                }
            ],
            "unknown features",
        ),
        (
            [{"Age": "not-a-number", "Department": "IT"}],
            "must be numeric or null",
        ),
        (
            [{"Age": float("inf"), "Department": "IT"}],
            "finite JSON value",
        ),
        (
            [{"Age": 30, "Department": ["IT"]}],
            "string, number, boolean, or null",
        ),
    ],
)
def test_invalid_prediction_records_are_rejected(
    tmp_path,
    classification_training_result,
    records,
    expected_message,
):
    export = _export_bundle(
        tmp_path / expected_message.replace(" ", "_"),
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )

    with pytest.raises(
        InferenceInputError,
        match=expected_message,
    ):
        predict_with_model_artifact(
            export["manifest_path"],
            records,
        )


def test_prediction_row_limit_is_enforced(
    tmp_path,
    classification_training_result,
):
    export = _export_bundle(
        tmp_path / "bounded_bundle",
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )
    records = [
        {"Age": 30, "Department": "IT"},
        {"Age": 31, "Department": "HR"},
    ]

    with pytest.raises(
        InferenceInputError,
        match="limited to 1 rows",
    ):
        predict_with_model_artifact(
            export["manifest_path"],
            records,
            max_records=1,
        )


def test_legacy_bundle_schema_is_derived_safely(
    tmp_path,
    classification_training_result,
):
    export = _export_bundle(
        tmp_path / "legacy_bundle",
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )
    manifest_path = Path(export["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("input_schema")
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False),
        encoding="utf-8",
    )

    description = describe_model_input(manifest_path)

    assert description["schema_status"] == "complete"
    assert description["input_schema"]["schema_source"] == (
        "legacy_bundle_fallback"
    )
    assert description["input_schema"]["required_features"] == [
        "Age",
        "Department",
    ]


def test_corrupted_pipeline_blocks_prediction(
    tmp_path,
    classification_training_result,
):
    export = _export_bundle(
        tmp_path / "corrupt_bundle",
        classification_training_result,
        _classification_dataframe(),
        "Outcome",
    )
    pipeline_path = Path(export["pipeline_path"])
    payload = bytearray(pipeline_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    pipeline_path.write_bytes(payload)

    with pytest.raises(
        InferenceArtifactError,
        match="verification failed",
    ):
        predict_with_model_artifact(
            export["manifest_path"],
            [{"Age": 30, "Department": "IT"}],
        )
