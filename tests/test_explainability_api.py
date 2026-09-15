import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core import dataset_files
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_api_storage(tmp_path, monkeypatch):
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
    filename="classification-explanation.csv",
):
    rows = 80
    dataframe = pd.DataFrame(
        {
            "Age": [20 + index % 35 for index in range(rows)],
            "Department": [
                ["IT", "HR", "Sales"][index % 3]
                for index in range(rows)
            ],
            "Outcome": [0, 1] * 40,
        }
    )
    dataframe.to_csv(
        dataset_files.UPLOAD_DIR / filename,
        index=False,
    )
    return dataframe


def _write_regression_dataset(
    filename="regression-explanation.csv",
):
    rows = 80
    dataframe = pd.DataFrame(
        {
            "Experience": [index % 15 for index in range(rows)],
            "Department": ["A", "B", "C", "D"] * 20,
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


def _fast_options():
    return {
        "run_tuning": False,
        "requested_models": ["logistic_regression"],
        "include_explanations": True,
        "explainability_repeats": 2,
        "export_artifacts": False,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }


def test_classification_api_returns_safe_explanations():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "classification-explanation.csv",
            "target_column": "Outcome",
            **_fast_options(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    explanation = body["report"]["explainability"]

    assert explanation["explainability_status"] == "complete"
    assert explanation["problem_type"] == "classification"
    assert explanation["evaluation_scope"] == "untouched_holdout"
    assert explanation["used_for_model_selection"] is False
    assert explanation["permutation_importance"]["scoring"] == (
        "balanced_accuracy"
    )
    assert explanation["native_importance"]["method"] == (
        "absolute_coefficients"
    )
    assert explanation["source_data_modified"] is False

    encoded = json.dumps(body, allow_nan=False)
    assert "evaluation_context" not in encoded
    assert "X_holdout" not in encoded
    assert "y_holdout" not in encoded


def test_regression_api_uses_rmse_explanations():
    _write_regression_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "regression-explanation.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            "requested_models": ["ridge"],
            "run_tuning": False,
            "include_explanations": True,
            "explainability_repeats": 2,
            "export_artifacts": False,
            "training_configuration_overrides": {
                "cv_folds": 2,
                "feature_selection_percentile": 100,
            },
        },
    )

    assert response.status_code == 200
    explanation = response.json()["report"]["explainability"]
    assert explanation["problem_type"] == "regression"
    assert explanation["baseline_metric"]["name"] == "rmse"
    assert explanation["permutation_importance"][
        "importance_unit"
    ] == "rmse_increase"
    assert explanation["native_importance"]["method"] == (
        "absolute_coefficients"
    )
    json.dumps(explanation, allow_nan=False)


def test_api_applies_bounded_explainability_configuration():
    _write_classification_dataset()
    request = {
        "filename": "classification-explanation.csv",
        "target_column": "Outcome",
        **_fast_options(),
        "explainability_repeats": 3,
        "explainability_max_rows": 10,
        "explainability_top_n": 1,
    }
    response = client.post("/analysis/train", json=request)

    assert response.status_code == 200
    explanation = response.json()["report"]["explainability"]
    assert explanation["configuration"]["n_repeats"] == 3
    assert explanation["configuration"]["max_rows"] == 10
    assert explanation["configuration"]["top_n"] == 1
    assert explanation["evaluation_rows_used"] == 10
    assert explanation["sampling_applied"] is True
    assert len(
        explanation["permutation_importance"]["features"]
    ) == 1


def test_api_can_explicitly_skip_explanations():
    _write_classification_dataset()
    request = {
        "filename": "classification-explanation.csv",
        "target_column": "Outcome",
        **_fast_options(),
        "include_explanations": False,
    }
    response = client.post("/analysis/train", json=request)

    assert response.status_code == 200
    assert response.json()["report"]["explainability"] == {
        "explainability_status": "skipped",
        "reason": "disabled_by_request",
    }


def test_api_rejects_oversized_explainability_request():
    _write_classification_dataset()
    request = {
        "filename": "classification-explanation.csv",
        "target_column": "Outcome",
        **_fast_options(),
        "explainability_repeats": 21,
    }
    response = client.post("/analysis/train", json=request)

    assert response.status_code == 422


def test_exported_report_contains_explanation_without_holdout_rows():
    _write_regression_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "regression-explanation.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            "requested_models": ["ridge"],
            "run_tuning": False,
            "include_explanations": True,
            "explainability_repeats": 2,
            "export_artifacts": True,
            "training_configuration_overrides": {
                "cv_folds": 2,
                "feature_selection_percentile": 100,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    report_download = next(
        download
        for download in body["artifacts"]["downloads"]
        if download["artifact_type"] == "training_report"
    )
    report_response = client.get(report_download["artifact_url"])

    assert report_response.status_code == 200
    exported_report = report_response.json()
    assert exported_report["explainability"][
        "explainability_status"
    ] == "complete"

    encoded = json.dumps(exported_report, allow_nan=False)
    assert "evaluation_context" not in encoded
    assert "X_holdout" not in encoded
    assert "y_holdout" not in encoded
