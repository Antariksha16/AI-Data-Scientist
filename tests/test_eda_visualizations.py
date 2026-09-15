import copy
import json
from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda_visualizations import (
    _unique_display_labels,
    generate_eda_visualizations,
)


def _classification_dataframe(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20 + index % 40 for index in range(rows)],
            "Fare": [
                None if index % 17 == 0 else 10.5 + index * 1.25
                for index in range(rows)
            ],
            "Department": ["IT", "HR", "Sales"] * (rows // 3),
            "Outcome": [0, 1] * (rows // 2),
        }
    )


def _classification_report() -> dict:
    return {
        "report_version": "1.0",
        "report_status": "complete",
        "target_column": "Outcome",
        "summary": {"problem_type": "classification"},
        "numerical_statistics": {
            "numerical_columns": ["Age", "Fare"],
        },
        "categorical_statistics": {
            "categorical_columns": ["Department"],
        },
        "datetime_statistics": {
            "datetime_columns": [],
        },
        "feature_target_relationships": {
            "ranked_features": [
                {
                    "feature": "Department",
                    "feature_type": "categorical",
                    "primary_score": 0.3,
                },
                {
                    "feature": "Age",
                    "feature_type": "numerical",
                    "primary_score": 0.1,
                },
            ]
        },
    }


def _plot_ids(manifest: dict) -> set[str]:
    return {
        item["plot_id"]
        for item in manifest["generated_plots"]
    }


def _assert_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 100
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_long_plot_labels_remain_distinct():
    labels = _unique_display_labels(
        [
            "a_very_long_category_name_with_shared_prefix_one",
            "a_very_long_category_name_with_shared_prefix_two",
        ]
    )

    assert labels[0] != labels[1]
    assert labels[1].endswith("[2]")


def test_classification_visualizations_and_manifest(tmp_path):
    df = _classification_dataframe()

    manifest = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=_classification_report(),
    )

    assert manifest["visualization_status"] == "complete"
    assert manifest["problem_type"] == "classification"
    assert manifest["backend"].lower() == "agg"
    assert manifest["plot_error_count"] == 0
    assert {
        "target_distribution",
        "missingness",
        "numerical_distributions",
        "categorical_distributions",
        "correlation_heatmap",
        "feature_target_relationships",
    }.issubset(_plot_ids(manifest))

    for plot in manifest["generated_plots"]:
        _assert_png(Path(plot["file_path"]))


def test_regression_target_and_relationship_visualizations(tmp_path):
    rows = 90
    df = pd.DataFrame(
        {
            "Experience": [index % 15 for index in range(rows)],
            "Department": ["A", "B", "C"] * 30,
            "Salary": [25_000.5 + index * 731.25 for index in range(rows)],
        }
    )
    report = {
        "target_column": "Salary",
        "summary": {"problem_type": "regression"},
        "numerical_statistics": {
            "numerical_columns": ["Experience"],
        },
        "categorical_statistics": {
            "categorical_columns": ["Department"],
        },
        "datetime_statistics": {"datetime_columns": []},
        "feature_target_relationships": {
            "ranked_features": [
                {
                    "feature": "Experience",
                    "feature_type": "numerical",
                    "primary_score": 0.2,
                },
                {
                    "feature": "Department",
                    "feature_type": "categorical",
                    "primary_score": 0.1,
                },
            ]
        },
    }

    manifest = generate_eda_visualizations(
        df,
        "Salary",
        tmp_path,
        eda_report=report,
    )

    assert manifest["problem_type"] == "regression"
    assert "target_distribution" in _plot_ids(manifest)
    assert "feature_target_relationships" in _plot_ids(manifest)
    assert manifest["plot_error_count"] == 0


def test_datetime_visualization_is_generated(tmp_path):
    rows = 120
    df = pd.DataFrame(
        {
            "EventDate": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="D",
            ),
            "Value": range(rows),
            "Outcome": [0, 1] * 60,
        }
    )
    report = {
        "target_column": "Outcome",
        "summary": {"problem_type": "classification"},
        "numerical_statistics": {"numerical_columns": ["Value"]},
        "categorical_statistics": {"categorical_columns": []},
        "datetime_statistics": {
            "datetime_columns": ["EventDate"],
        },
        "feature_target_relationships": {
            "ranked_features": [
                {
                    "feature": "EventDate",
                    "feature_type": "datetime",
                    "primary_score": 0.2,
                }
            ]
        },
    }

    manifest = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=report,
    )

    assert "datetime_overview" in _plot_ids(manifest)
    datetime_plot = next(
        item
        for item in manifest["generated_plots"]
        if item["plot_id"] == "datetime_overview"
    )
    assert datetime_plot["features"] == ["EventDate"]


def test_missingness_plot_is_skipped_when_not_applicable(tmp_path):
    df = _classification_dataframe()
    df["Fare"] = df["Fare"].fillna(0)

    manifest = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=_classification_report(),
    )

    assert "missingness" not in _plot_ids(manifest)
    assert any(
        item["plot_id"] == "missingness"
        for item in manifest["skipped_plots"]
    )
    assert manifest["visualization_status"] == "complete"


def test_sampling_limit_and_source_non_mutation(tmp_path):
    df = _classification_dataframe(rows=120)
    original = df.copy(deep=True)
    config = EDAConfig(max_plot_rows=25)

    manifest = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=_classification_report(),
        config=config,
    )

    assert manifest["sampling"] == {
        "original_rows": 120,
        "plotted_rows": 25,
        "sampling_performed": True,
        "max_plot_rows": 25,
        "random_state": 42,
    }
    pd.testing.assert_frame_equal(df, original)
    assert manifest["source_dataframe_modified"] is False


def test_feature_and_category_limits_are_enforced(tmp_path):
    rows = 120
    df = pd.DataFrame(
        {
            **{
                f"Numeric{index}": [
                    value + index for value in range(rows)
                ]
                for index in range(5)
            },
            "Category": [f"value_{index}" for index in range(rows)],
            "Outcome": [0, 1] * 60,
        }
    )
    config = EDAConfig(
        max_features_visualized=2,
        max_categories_displayed=3,
        max_pairwise_features=2,
    )

    manifest = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        config=config,
        problem_type="classification",
    )

    assert len(manifest["detected_features"]["numerical"]) == 2
    assert manifest["limits"]["max_categories_displayed"] == 3
    correlation = next(
        item
        for item in manifest["generated_plots"]
        if item["plot_id"] == "correlation_heatmap"
    )
    assert len(correlation["features"]) == 2


def test_plot_manifest_is_json_safe(tmp_path):
    manifest = generate_eda_visualizations(
        _classification_dataframe(),
        "Outcome",
        tmp_path,
        eda_report=_classification_report(),
    )

    json.dumps(manifest, allow_nan=False)
    assert manifest["automatic_data_modification_performed"] is False
    assert Path(manifest["output_directory"]) == tmp_path.resolve()


def test_existing_plot_files_are_replaced_and_reported(tmp_path):
    df = _classification_dataframe()
    report = _classification_report()

    first = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=report,
    )
    second = generate_eda_visualizations(
        df,
        "Outcome",
        tmp_path,
        eda_report=report,
    )

    assert first["overwritten_files"] == []
    assert len(second["overwritten_files"]) == (
        second["generated_plot_count"]
    )
    assert all(
        item["overwrote_existing_file"]
        for item in second["generated_plots"]
    )


def test_plot_failure_is_isolated_by_default(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated image write failure")

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fail_save,
    )

    manifest = generate_eda_visualizations(
        _classification_dataframe(),
        "Outcome",
        tmp_path,
        eda_report=_classification_report(),
        fail_fast=False,
    )

    assert manifest["visualization_status"] == "failed"
    assert manifest["generated_plot_count"] == 0
    assert manifest["plot_error_count"] > 0
    assert all(
        error["error_type"] == "RuntimeError"
        for error in manifest["plot_errors"]
    )


def test_fail_fast_reraises_plot_failure(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated image write failure")

    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        fail_save,
    )

    with pytest.raises(RuntimeError, match="simulated"):
        generate_eda_visualizations(
            _classification_dataframe(),
            "Outcome",
            tmp_path,
            eda_report=_classification_report(),
            fail_fast=True,
        )


def test_target_report_mismatch_is_rejected(tmp_path):
    report = _classification_report()
    report["target_column"] = "DifferentTarget"

    with pytest.raises(ValueError, match="does not match"):
        generate_eda_visualizations(
            _classification_dataframe(),
            "Outcome",
            tmp_path,
            eda_report=report,
        )


def test_visualization_input_validation(tmp_path):
    df = _classification_dataframe()

    with pytest.raises(ValueError, match="does not exist"):
        generate_eda_visualizations(
            df,
            "UnknownTarget",
            tmp_path,
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_eda_visualizations(
            pd.DataFrame(columns=["Target"]),
            "Target",
            tmp_path,
        )

    with pytest.raises(ValueError, match="image_format"):
        generate_eda_visualizations(
            df,
            "Outcome",
            tmp_path,
            image_format="jpg",
        )

    with pytest.raises(ValueError, match="greater than zero"):
        generate_eda_visualizations(
            df,
            "Outcome",
            tmp_path,
            dpi=0,
        )