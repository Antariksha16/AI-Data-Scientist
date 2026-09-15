import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core import dataset_files
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_prediction_storage(tmp_path, monkeypatch):
    upload_directory = tmp_path / "uploads"
    model_output_directory = tmp_path / "outputs" / "models"
    upload_directory.mkdir(parents=True)

    monkeypatch.setattr(
        dataset_files,
        "UPLOAD_DIR",
        upload_directory,
    )
    monkeypatch.setattr(
        dataset_files,
        "MODEL_OUTPUT_DIR",
        model_output_directory,
    )


def _write_classification_dataset(
    filename="classification.csv",
) -> pd.DataFrame:
    rows = 60
    dataframe = pd.DataFrame(
        {
            "Age": [20 + index % 35 for index in range(rows)],
            "Department": ["IT", "HR", "Sales"] * 20,
            "Outcome": [0, 1] * 30,
        }
    )
    dataframe.to_csv(
        dataset_files.UPLOAD_DIR / filename,
        index=False,
    )
    return dataframe


def _write_regression_dataset(
    filename="regression.csv",
) -> pd.DataFrame:
    rows = 60
    dataframe = pd.DataFrame(
        {
            "Experience": [index % 15 for index in range(rows)],
            "Department": ["A", "B", "C"] * 20,
            "Salary": [
                25_000.5 + index * 731.25
                for index in range(rows)
            ],
        }
    )
    dataframe.to_csv(
        dataset_files.UPLOAD_DIR / filename,
        index=False,
    )
    return dataframe


def _train_exported_model(
    filename: str,
    target_column: str,
    model_name: str,
    *,
    problem_type: str | None = None,
) -> str:
    request = {
        "filename": filename,
        "target_column": target_column,
        "requested_models": [model_name],
        "run_tuning": False,
        "include_explanations": False,
        "export_artifacts": True,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }

    if problem_type is not None:
        request["problem_type"] = problem_type

    response = client.post("/analysis/train", json=request)
    assert response.status_code == 200, response.text
    return response.json()["training_id"]


def test_classification_schema_and_batch_prediction_are_safe():
    dataframe = _write_classification_dataset()
    training_id = _train_exported_model(
        "classification.csv",
        "Outcome",
        "logistic_regression",
    )

    schema_response = client.get(
        f"/analysis/predict/schema/{training_id}"
    )
    assert schema_response.status_code == 200
    schema_body = schema_response.json()
    assert schema_body["training_id"] == training_id
    assert schema_body["problem_type"] == "classification"
    assert schema_body["input_schema"]["required_features"] == [
        "Age",
        "Department",
    ]
    assert schema_body["input_schema"][
        "sample_values_stored"
    ] is False

    records = dataframe.drop(columns=["Outcome"]).head(4).to_dict(
        orient="records"
    )
    response = client.post(
        "/analysis/predict",
        json={
            "training_id": training_id,
            "records": records,
            "include_probabilities": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    report = body["report"]
    assert body["training_id"] == training_id
    assert report["prediction_status"] == "complete"
    assert report["prediction_count"] == 4
    assert report["probability_status"] == "complete"
    assert report["artifact_verification"] == {
        "verification_status": "complete",
        "pipeline_load_verified": True,
    }
    assert all(
        len(row["probabilities"]) == 2
        for row in report["predictions"]
    )
    assert report["source_records_modified"] is False
    json.dumps(body, allow_nan=False)


def test_regression_api_returns_predictions_without_probabilities():
    dataframe = _write_regression_dataset()
    training_id = _train_exported_model(
        "regression.csv",
        "Salary",
        "ridge",
        problem_type="regression",
    )
    records = dataframe.drop(columns=["Salary"]).head(3).to_dict(
        orient="records"
    )

    response = client.post(
        "/analysis/predict",
        json={
            "training_id": training_id,
            "records": records,
        },
    )

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["problem_type"] == "regression"
    assert report["prediction_count"] == 3
    assert report["probability_status"] == "not_applicable"
    assert all(
        isinstance(row["prediction"], float)
        for row in report["predictions"]
    )
    assert all(
        "probabilities" not in row
        for row in report["predictions"]
    )
    json.dumps(response.json(), allow_nan=False)


def test_prediction_rejects_invalid_training_identifier():
    response = client.post(
        "/analysis/predict",
        json={
            "training_id": "x" * 32,
            "records": [{"Age": 30, "Department": "IT"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid training identifier."
    )


def test_prediction_rejects_unknown_training_identifier():
    response = client.post(
        "/analysis/predict",
        json={
            "training_id": uuid4().hex,
            "records": [{"Age": 30, "Department": "IT"}],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Model artifact was not found."
    )


def test_prediction_rejects_missing_and_unknown_features():
    _write_classification_dataset()
    training_id = _train_exported_model(
        "classification.csv",
        "Outcome",
        "logistic_regression",
    )

    missing_response = client.post(
        "/analysis/predict",
        json={
            "training_id": training_id,
            "records": [{"Department": "IT"}],
        },
    )
    assert missing_response.status_code == 422
    assert "missing required features" in missing_response.json()[
        "detail"
    ]

    unknown_response = client.post(
        "/analysis/predict",
        json={
            "training_id": training_id,
            "records": [
                {
                    "Age": 30,
                    "Department": "IT",
                    "Outcome": 1,
                }
            ],
        },
    )
    assert unknown_response.status_code == 422
    assert "unknown features" in unknown_response.json()["detail"]


def test_prediction_request_size_is_bounded():
    response = client.post(
        "/analysis/predict",
        json={
            "training_id": uuid4().hex,
            "records": [
                {"Age": 30, "Department": "IT"}
                for _ in range(1_001)
            ],
        },
    )

    assert response.status_code == 422


def test_corrupted_server_model_returns_conflict():
    _write_classification_dataset()
    training_id = _train_exported_model(
        "classification.csv",
        "Outcome",
        "logistic_regression",
    )
    pipeline_path = (
        dataset_files.MODEL_OUTPUT_DIR
        / training_id
        / "model_pipeline.joblib"
    )
    payload = bytearray(Path(pipeline_path).read_bytes())
    payload[len(payload) // 2] ^= 0x01
    Path(pipeline_path).write_bytes(payload)

    response = client.post(
        "/analysis/predict",
        json={
            "training_id": training_id,
            "records": [{"Age": 30, "Department": "IT"}],
        },
    )

    assert response.status_code == 409
    assert "verification failed" in response.json()["detail"]
