import json
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core import dataset_files
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    upload_directory = tmp_path / "uploads"
    output_directory = tmp_path / "outputs" / "eda"
    upload_directory.mkdir(parents=True)

    monkeypatch.setattr(
        dataset_files,
        "UPLOAD_DIR",
        upload_directory,
    )
    monkeypatch.setattr(
        dataset_files,
        "EDA_OUTPUT_DIR",
        output_directory,
    )


def _write_classification_dataset(filename="classification.csv"):
    rows = 60
    df = pd.DataFrame(
        {
            "Age": [20 + index % 35 for index in range(rows)],
            "Department": ["IT", "HR", "Sales"] * 20,
            "Outcome": [0, 1] * 30,
        }
    )
    path = dataset_files.UPLOAD_DIR / filename
    df.to_csv(path, index=False)
    return path


def _write_regression_dataset(filename="regression.csv"):
    rows = 60
    df = pd.DataFrame(
        {
            "Experience": [index % 15 for index in range(rows)],
            "Department": ["A", "B", "C"] * 20,
            "Salary": [
                25_000.5 + index * 731.25
                for index in range(rows)
            ],
        }
    )
    path = dataset_files.UPLOAD_DIR / filename
    df.to_csv(path, index=False)
    return path


def test_existing_target_endpoint_remains_compatible():
    _write_classification_dataset()

    response = client.post(
        "/analysis/target",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_target"] == "Outcome"
    assert body["target_analysis"]["problem_type"] == "classification"
    assert "Outcome" not in body["candidate_features"]


def test_classification_eda_endpoint_without_visualizations():
    _write_classification_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "include_visualizations": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    report = body["report"]
    assert body["analysis_id"] is None
    assert body["artifact_base_url"] is None
    assert report["report_status"] == "complete"
    assert report["summary"]["problem_type"] == "classification"
    assert report["visualizations"] is None
    assert report["automatic_data_modification_performed"] is False
    json.dumps(body, allow_nan=False)


def test_regression_eda_generates_downloadable_visualizations():
    _write_regression_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": "regression.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            "include_visualizations": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    report = body["report"]
    visualizations = report["visualizations"]
    assert len(body["analysis_id"]) == 32
    assert report["summary"]["problem_type"] == "regression"
    assert visualizations["generated_plot_count"] >= 1
    assert visualizations["output_directory"] is None
    assert report["execution_options"][
        "visualization_output_directory"
    ] is None

    first_plot = visualizations["generated_plots"][0]
    assert "file_path" not in first_plot
    assert first_plot["artifact_url"].startswith(
        f"/analysis/eda/artifacts/{body['analysis_id']}/"
    )

    artifact_response = client.get(first_plot["artifact_url"])
    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"].startswith("image/")
    assert len(artifact_response.content) > 0


def test_eda_accepts_valid_configuration_overrides():
    _write_classification_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "include_visualizations": False,
            "configuration_overrides": {
                "high_correlation_threshold": 0.85,
                "max_plot_rows": 5000,
            },
        },
    )

    assert response.status_code == 200
    configuration = response.json()["report"]["configuration"]
    assert configuration["high_correlation_threshold"] == 0.85
    assert configuration["max_plot_rows"] == 5000


def test_invalid_configuration_is_rejected():
    _write_classification_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": "classification.csv",
            "target_column": "Outcome",
            "include_visualizations": False,
            "configuration_overrides": {
                "high_correlation_threshold": 1.5,
            },
        },
    )

    assert response.status_code == 400
    assert "Invalid EDA configuration" in response.json()["detail"]


@pytest.mark.parametrize(
    "unsafe_filename",
    ["../classification.csv", "..\\classification.csv", "C:dataset.csv"],
)
def test_dataset_path_traversal_is_rejected(unsafe_filename):
    _write_classification_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": unsafe_filename,
            "target_column": "Outcome",
            "include_visualizations": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filename."


def test_missing_dataset_returns_not_found():
    response = client.post(
        "/analysis/eda",
        json={
            "filename": "missing.csv",
            "target_column": "Outcome",
            "include_visualizations": False,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Uploaded dataset was not found."
    )


def test_missing_target_returns_bad_request():
    _write_classification_dataset()

    response = client.post(
        "/analysis/eda",
        json={
            "filename": "classification.csv",
            "target_column": "UnknownTarget",
            "include_visualizations": False,
        },
    )

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_invalid_artifact_identifier_is_rejected():
    response = client.get(
        "/analysis/eda/artifacts/not-a-uuid/plot.png"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid analysis identifier."
    )


def test_unknown_artifact_returns_not_found():
    response = client.get(
        f"/analysis/eda/artifacts/{uuid4().hex}/plot.png"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "EDA artifact was not found."


def test_upload_rejects_unsafe_filename():
    response = client.post(
        "/upload",
        files={
            "file": (
                "../unsafe.csv",
                b"Feature,Target\n1,0\n2,1\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid filename."