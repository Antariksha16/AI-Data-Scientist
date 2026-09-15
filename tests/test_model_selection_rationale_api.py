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
    rng = np.random.default_rng(1_020)
    rows = 90
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.9, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)
    pd.DataFrame(
        {
            "Age": rng.integers(18, 70, size=rows),
            "Signal": signal,
            "Region": rng.choice(["North", "South", "West"], rows),
            "Outcome": target,
        }
    ).to_csv(
        dataset_files.UPLOAD_DIR / "rationale_classification.csv",
        index=False,
    )


def _write_regression_dataset() -> None:
    rng = np.random.default_rng(1_121)
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
        dataset_files.UPLOAD_DIR / "rationale_regression.csv",
        index=False,
    )


def _options() -> dict:
    return {
        "run_tuning": False,
        "include_explanations": False,
        "export_artifacts": False,
        "max_recommended_models": 3,
        "training_configuration_overrides": {
            "cv_folds": 3,
            "feature_selection_percentile": 100,
        },
    }


def test_classification_api_returns_public_selection_rationale():
    _write_classification_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "rationale_classification.csv",
            "target_column": "Outcome",
            **_options(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    rationale = body["report"]["model_selection_rationale"]
    serialized = json.dumps(body, allow_nan=False)

    assert rationale["rationale_status"] == "complete"
    assert rationale["winner"]["model_name"] == body["report"][
        "best_model"
    ]["model_name"]
    assert rationale["holdout_used_for_selection"] is False
    assert rationale["private_holdout_rows_exposed"] is False
    assert "X_holdout" not in serialized
    assert "y_holdout" not in serialized


def test_regression_api_returns_lower_is_better_rationale():
    _write_regression_dataset()
    response = client.post(
        "/analysis/train",
        json={
            "filename": "rationale_regression.csv",
            "target_column": "Salary",
            "problem_type": "regression",
            **_options(),
        },
    )

    assert response.status_code == 200
    rationale = response.json()["report"][
        "model_selection_rationale"
    ]
    assert rationale["primary_metric"] == {
        "name": "rmse",
        "direction": "lower",
    }
    assert rationale["runner_up"]["score_gap_from_winner"] >= 0


def test_user_override_is_explicitly_identified_by_api():
    _write_classification_dataset()
    options = _options()
    options["requested_models"] = ["logistic_regression"]
    response = client.post(
        "/analysis/train",
        json={
            "filename": "rationale_classification.csv",
            "target_column": "Outcome",
            **options,
        },
    )

    assert response.status_code == 200
    rationale = response.json()["report"][
        "model_selection_rationale"
    ]
    assert rationale["selection_mode"] == "user_override"
    assert rationale["runner_up"] is None
    assert rationale["decision_confidence"]["level"] == "limited"
    assert "included_by_explicit_user_request" in rationale["winner"][
        "selected_because"
    ]


def test_exported_training_report_contains_selection_rationale():
    _write_classification_dataset()
    options = _options()
    options.update(
        {
            "requested_models": ["logistic_regression"],
            "export_artifacts": True,
        }
    )
    response = client.post(
        "/analysis/train",
        json={
            "filename": "rationale_classification.csv",
            "target_column": "Outcome",
            **options,
        },
    )

    assert response.status_code == 200
    body = response.json()
    report_download = next(
        item
        for item in body["artifacts"]["downloads"]
        if item["artifact_type"] == "training_report"
    )
    artifact_response = client.get(report_download["artifact_url"])

    assert artifact_response.status_code == 200
    exported_report = artifact_response.json()
    rationale = exported_report["model_selection_rationale"]
    serialized = json.dumps(exported_report, allow_nan=False)
    assert rationale["rationale_status"] == "complete"
    assert rationale["private_holdout_rows_exposed"] is False
    assert "X_holdout" not in serialized
    assert "y_holdout" not in serialized

