import json
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core import dataset_files
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_training_storage(tmp_path, monkeypatch):
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
):
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


def _write_regression_dataset(filename="regression.csv"):
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


def _fast_training_options():
    return {
        "run_tuning": False,
        "export_artifacts": False,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }


def test_classification_training_without_artifact_export():
    _write_classification_dataset()
    request = {
        "filename": "classification.csv",
        "target_column": "Outcome",
        "requested_models": ["logistic_regression"],
        **_fast_training_options(),
    }

    response = client.post("/analysis/train", json=request)

    assert response.status_code == 200
    body = response.json()
    assert body["training_id"] is None
    assert body["artifact_base_url"] is None
    assert body["artifacts"] is None
    assert body["report"]["problem_type"] == "classification"
    assert body["report"]["training_status"] == "complete"
    assert body["report"]["split"]["stratified"] is True
    json.dumps(body, allow_nan=False)


def test_regression_training_exports_downloadable_bundle():
    _write_regression_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "regression.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            "requested_models": ["ridge"],
            "run_tuning": False,
            "export_artifacts": True,
            "training_configuration_overrides": {
                "cv_folds": 2,
                "feature_selection_percentile": 100,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["training_id"]) == 32
    assert body["report"]["problem_type"] == "regression"
    assert body["artifacts"][
        "prediction_equivalence_verified"
    ] is True
    assert body["artifacts"]["verification_rows"] == 20
    assert "output_directory" not in body["artifacts"]

    downloads = body["artifacts"]["downloads"]
    assert len(downloads) == 3

    for download in downloads:
        assert download["artifact_url"].startswith(
            f"/analysis/train/artifacts/{body['training_id']}/"
        )
        artifact_response = client.get(download["artifact_url"])
        assert artifact_response.status_code == 200
        assert len(artifact_response.content) > 0

    json.dumps(body, allow_nan=False)


def test_bounded_tuning_configuration_is_applied():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "requested_models": ["logistic_regression"],
            "run_tuning": True,
            "tuning_iterations": 2,
            "tuning_top_models": 1,
            "export_artifacts": False,
            "training_configuration_overrides": {
                "cv_folds": 2,
                "feature_selection_percentile": 100,
            },
        },
    )

    assert response.status_code == 200
    tuning = response.json()["report"]["hyperparameter_tuning"]
    assert tuning["status"] == "complete"
    assert tuning["configuration"]["n_iter"] == 2
    assert tuning["configuration"]["top_models"] == 1
    assert tuning["configuration"]["n_jobs"] == 1


def test_invalid_training_configuration_is_rejected():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "run_tuning": False,
            "training_configuration_overrides": {
                "test_size": 1.5,
            },
        },
    )

    assert response.status_code == 400
    assert "Invalid training configuration" in response.json()[
        "detail"
    ]


def test_api_rejects_parallel_training_jobs():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "run_tuning": False,
            "training_configuration_overrides": {"n_jobs": -1},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "API training requires n_jobs=1."
    )


def test_unknown_requested_model_is_rejected():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "requested_models": ["unknown_model"],
            **_fast_training_options(),
        },
    )

    assert response.status_code == 400
    assert "Unknown model names" in response.json()["detail"]


def test_training_rejects_dataset_path_traversal():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "../classification.csv",
            "target_column": "Outcome",
            **_fast_training_options(),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filename."


def test_training_rejects_missing_target():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "MissingTarget",
            **_fast_training_options(),
        },
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_training_leakage_requires_explicit_override():
    dataframe = _write_classification_dataset("leakage.csv")
    dataframe["OutcomeCopy"] = (
        dataframe["Outcome"]
        + pd.Series(range(len(dataframe))) * 0.000001
    )
    dataframe.to_csv(
        dataset_files.UPLOAD_DIR / "leakage.csv",
        index=False,
    )
    request = {
        "filename": "leakage.csv",
        "target_column": "Outcome",
        "requested_models": ["logistic_regression"],
        **_fast_training_options(),
    }

    blocked_response = client.post(
        "/analysis/train",
        json=request,
    )
    assert blocked_response.status_code == 400
    assert "Potential target leakage" in blocked_response.json()[
        "detail"
    ]

    request["allow_potential_leakage"] = True
    allowed_response = client.post(
        "/analysis/train",
        json=request,
    )
    assert allowed_response.status_code == 200


def test_invalid_training_artifact_identifier_is_rejected():
    response = client.get(
        "/analysis/train/artifacts/not-a-uuid/model_pipeline.joblib"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid training identifier."
    )


def test_unsupported_training_artifact_is_rejected():
    response = client.get(
        f"/analysis/train/artifacts/{uuid4().hex}/unknown.txt"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported model artifact."
    )


def test_unknown_training_artifact_returns_not_found():
    response = client.get(
        "/analysis/train/artifacts/"
        f"{uuid4().hex}/model_pipeline.joblib"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Model artifact was not found."
    )


def test_api_limits_tuning_search_size():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "tuning_iterations": 51,
        },
    )

    assert response.status_code == 422
