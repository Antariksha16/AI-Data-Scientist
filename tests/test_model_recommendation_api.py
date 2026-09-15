import json

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


def _write_dataset(filename: str = "recommendation.csv") -> None:
    rows = 80
    dataframe = pd.DataFrame(
        {
            "Age": [20 + index % 45 for index in range(rows)],
            "Score": [float(index % 17) for index in range(rows)],
            "Department": [
                ["IT", "HR", "Sales"][index % 3]
                for index in range(rows)
            ],
            "Outcome": [0, 1] * 40,
        }
    )
    dataframe.to_csv(dataset_files.UPLOAD_DIR / filename, index=False)


def _base_request() -> dict:
    return {
        "filename": "recommendation.csv",
        "target_column": "Outcome",
        "run_tuning": False,
        "include_explanations": False,
        "export_artifacts": False,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }


def test_api_automatically_trains_only_bounded_recommendations():
    _write_dataset()
    request = _base_request()
    request["max_recommended_models"] = 2

    response = client.post("/analysis/train", json=request)

    assert response.status_code == 200
    body = response.json()
    recommendation = body["report"]["model_recommendation"]
    assert recommendation["selection_mode"] == "automatic_recommendation"
    assert recommendation["selected_model_count"] == 2
    assert recommendation["configuration"]["max_models"] == 2
    assert recommendation["selection_scope"] == "training_split_only"
    assert recommendation["holdout_data_used"] is False
    assert len(body["report"]["model_comparisons"]) == 3
    json.dumps(body, allow_nan=False)


def test_api_explicit_model_request_remains_user_controlled():
    _write_dataset()
    request = _base_request()
    request["requested_models"] = ["logistic_regression"]
    request["max_recommended_models"] = 4

    response = client.post("/analysis/train", json=request)

    assert response.status_code == 200
    recommendation = response.json()["report"]["model_recommendation"]
    assert recommendation["selection_mode"] == "user_override"
    assert recommendation["selected_models"] == ["logistic_regression"]
    assert recommendation["configuration"] is None


def test_api_rejects_oversized_recommendation_request():
    _write_dataset()
    request = _base_request()
    request["max_recommended_models"] = 7

    response = client.post("/analysis/train", json=request)

    assert response.status_code == 422


def test_api_rejects_invalid_recommendation_threshold_override():
    _write_dataset()
    request = _base_request()
    request["training_configuration_overrides"].update(
        {
            "recommendation_small_dataset_rows": 100,
            "recommendation_large_dataset_rows": 50,
        }
    )

    response = client.post("/analysis/train", json=request)

    assert response.status_code == 400
    assert "recommendation_large_dataset_rows" in response.json()[
        "detail"
    ]
