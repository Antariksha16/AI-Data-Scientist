import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api import analysis as analysis_api
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
    monkeypatch.setattr(
        analysis_api,
        "generate_eda_report",
        lambda *args, **kwargs: {
            "report_status": "complete",
            "dataset_overview": {
                "shape": {
                    "rows": len(args[0]),
                    "columns": len(args[0].columns),
                },
                "missingness": {
                    "total_missing_cells": 0,
                    "total_missing_percentage": 0.0,
                },
                "duplicates": {"duplicate_rows": 0},
            },
            "insights": {
                "prioritized_insights": [],
                "limitations": [],
            },
            "warnings": [],
            "limitations": [],
            "visualizations": None,
        },
    )


def _write_dataset(filename: str = "report.csv") -> None:
    rng = np.random.default_rng(2_024)
    rows = 70
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.8, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)
    pd.DataFrame(
        {
            "Age": rng.integers(18, 70, size=rows),
            "Signal": signal,
            "Region": rng.choice(["North", "South", "West"], rows),
            "Outcome": target,
        }
    ).to_csv(dataset_files.UPLOAD_DIR / filename, index=False)


def _request(**overrides) -> dict:
    request = {
        "filename": "report.csv",
        "target_column": "Outcome",
        "requested_models": ["logistic_regression"],
        "run_tuning": False,
        "include_explanations": False,
        "include_professional_report": True,
        "include_eda_visualizations_in_report": False,
        "export_artifacts": True,
        "training_configuration_overrides": {
            "cv_folds": 2,
            "feature_selection_percentile": 100,
        },
    }
    request.update(overrides)
    return request


def test_api_exports_downloadable_markdown_and_pdf_reports():
    _write_dataset()
    response = client.post("/analysis/train", json=_request())

    assert response.status_code == 200
    body = response.json()
    report = body["report"]["professional_report"]
    assert report["report_status"] == "complete"
    assert report["formats"] == ["markdown", "pdf"]
    assert report["private_holdout_rows_included"] is False

    by_type = {
        item["artifact_type"]: item
        for item in body["artifacts"]["downloads"]
    }
    markdown = client.get(
        by_type["professional_report_markdown"]["artifact_url"]
    )
    pdf = client.get(by_type["professional_report_pdf"]["artifact_url"])
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "Why this model was selected" in markdown.text
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-")


def test_api_can_export_markdown_only():
    _write_dataset()
    response = client.post(
        "/analysis/train",
        json=_request(professional_report_formats=["markdown"]),
    )

    assert response.status_code == 200
    body = response.json()
    report = body["report"]["professional_report"]
    assert report["formats"] == ["markdown"]
    assert set(report["files"]) == {"professional_report_markdown"}
    artifact_types = {
        item["artifact_type"]
        for item in body["artifacts"]["downloads"]
    }
    assert "professional_report_markdown" in artifact_types
    assert "professional_report_pdf" not in artifact_types


def test_professional_report_requires_artifact_export():
    response = client.post(
        "/analysis/train",
        json=_request(export_artifacts=False),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Professional reports require export_artifacts=True."
    )


def test_api_rejects_empty_report_format_list():
    response = client.post(
        "/analysis/train",
        json=_request(professional_report_formats=[]),
    )
    assert response.status_code == 422


def test_exported_json_records_reports_without_private_rows():
    _write_dataset()
    response = client.post("/analysis/train", json=_request())
    assert response.status_code == 200
    body = response.json()
    training_report_download = next(
        item
        for item in body["artifacts"]["downloads"]
        if item["artifact_type"] == "training_report"
    )
    exported = client.get(training_report_download["artifact_url"])

    assert exported.status_code == 200
    report = exported.json()
    serialized = json.dumps(report, allow_nan=False)
    assert report["professional_report"]["report_status"] == "complete"
    assert report["professional_report"][
        "raw_dataset_rows_included"
    ] is False
    assert "X_holdout" not in serialized
    assert "y_holdout" not in serialized
