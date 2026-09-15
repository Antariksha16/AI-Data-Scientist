import hashlib
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


def _write_dataset(filename="chart-classification.csv"):
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


def _request(**overrides):
    request = {
        "filename": "chart-classification.csv",
        "target_column": "Outcome",
        "requested_models": ["logistic_regression"],
        "run_tuning": False,
        "include_explanations": True,
        "explainability_repeats": 2,
        "include_explanation_visualizations": True,
        "export_artifacts": True,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }
    request.update(overrides)
    return request


def test_api_exports_hashed_downloadable_png_charts():
    _write_dataset()
    response = client.post(
        "/analysis/train",
        json=_request(),
    )

    assert response.status_code == 200
    body = response.json()
    visualizations = body["report"]["explainability"][
        "visualizations"
    ]
    assert visualizations["visualization_status"] == "complete"
    assert visualizations["generated_plot_count"] == 2
    assert visualizations["raw_holdout_rows_used"] is False

    chart_downloads = [
        artifact
        for artifact in body["artifacts"]["downloads"]
        if artifact["artifact_type"].endswith("_chart")
    ]
    assert len(chart_downloads) == 2

    for artifact in chart_downloads:
        download = client.get(artifact["artifact_url"])
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("image/png")
        assert download.content.startswith(b"\x89PNG")
        assert hashlib.sha256(download.content).hexdigest() == (
            artifact["sha256"]
        )

    json.dumps(body, allow_nan=False)


def test_api_exports_svg_chart_with_correct_media_type():
    _write_dataset()
    response = client.post(
        "/analysis/train",
        json=_request(explanation_image_format="svg"),
    )

    assert response.status_code == 200
    chart_download = next(
        artifact
        for artifact in response.json()["artifacts"]["downloads"]
        if artifact["artifact_type"]
        == "permutation_importance_chart"
    )
    download = client.get(chart_download["artifact_url"])
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in download.content


def test_visualizations_are_skipped_without_artifact_export():
    _write_dataset()
    response = client.post(
        "/analysis/train",
        json=_request(export_artifacts=False),
    )

    assert response.status_code == 200
    visualizations = response.json()["report"]["explainability"][
        "visualizations"
    ]
    assert visualizations["visualization_status"] == "skipped"
    assert visualizations["reason"] == "artifact_export_disabled"
    assert visualizations["raw_holdout_rows_used"] is False


def test_api_rejects_invalid_visualization_dpi():
    _write_dataset()
    response = client.post(
        "/analysis/train",
        json=_request(explanation_visualization_dpi=301),
    )

    assert response.status_code == 422


def test_unapproved_model_chart_name_is_rejected():
    response = client.get(
        "/analysis/train/artifacts/"
        "00000000000000000000000000000000/arbitrary.png"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported model artifact."
    )
