import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core import dataset_files
from backend.app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    upload_directory = tmp_path / "uploads"
    model_output_directory = tmp_path / "outputs" / "models"
    upload_directory.mkdir(parents=True)

    monkeypatch.setattr(dataset_files, "UPLOAD_DIR", upload_directory)
    monkeypatch.setattr(
        dataset_files,
        "MODEL_OUTPUT_DIR",
        model_output_directory,
    )


def _write_classification_dataset() -> None:
    rng = np.random.default_rng(404)
    rows = 90
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)
    pd.DataFrame(
        {
            "Age": rng.integers(18, 70, size=rows),
            "Signal": signal,
            "Department": rng.choice(["IT", "HR", "Sales"], rows),
            "Outcome": target,
        }
    ).to_csv(
        dataset_files.UPLOAD_DIR / "expanded_classification.csv",
        index=False,
    )


def _write_regression_dataset() -> None:
    rng = np.random.default_rng(505)
    rows = 90
    experience = rng.integers(0, 20, size=rows)
    performance = rng.normal(60, 10, size=rows)
    salary = (
        25_000
        + experience * 1_400
        + performance * 150
        + rng.normal(0, 4_000, size=rows)
    )
    pd.DataFrame(
        {
            "Experience": experience,
            "Performance": performance,
            "Department": rng.choice(["IT", "HR", "Sales"], rows),
            "Salary": salary,
        }
    ).to_csv(
        dataset_files.UPLOAD_DIR / "expanded_regression.csv",
        index=False,
    )


def _options() -> dict:
    return {
        "run_tuning": False,
        "include_explanations": False,
        "export_artifacts": False,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }


def test_api_reports_four_selected_from_fourteen_candidates():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "expanded_classification.csv",
            "target_column": "Outcome",
            "max_recommended_models": 4,
            **_options(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    recommendation = body["report"]["model_recommendation"]
    assert recommendation["registered_non_baseline_model_count"] == 14
    assert recommendation["selected_model_count"] == 4
    assert len(body["report"]["model_comparisons"]) == 5
    json.dumps(body, allow_nan=False)


def test_api_accepts_new_classification_model_override():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "expanded_classification.csv",
            "target_column": "Outcome",
            "requested_models": ["linear_svc"],
            **_options(),
        },
    )

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["model_recommendation"]["selection_mode"] == (
        "user_override"
    )
    assert report["model_recommendation"]["selected_models"] == [
        "linear_svc"
    ]
    assert {
        item["model_name"] for item in report["model_comparisons"]
    } == {"dummy_classifier", "linear_svc"}


def test_api_accepts_new_regression_model_override():
    _write_regression_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "expanded_regression.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            "requested_models": ["huber_regressor"],
            **_options(),
        },
    )

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["problem_type"] == "regression"
    assert report["model_recommendation"]["selected_models"] == [
        "huber_regressor"
    ]
    assert report["training_status"] == "complete"

