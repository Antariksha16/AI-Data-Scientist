import json

import pandas as pd
import pytest

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda import generate_eda_report


def _classification_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EmployeeID": range(1, 61),
            "Age": [20 + index % 35 for index in range(60)],
            "Department": ["IT", "HR", "Sales"] * 20,
            "EventDate": pd.date_range(
                "2025-01-01",
                periods=60,
                freq="D",
            ),
            "Outcome": [0, 1] * 30,
        }
    )


def test_complete_classification_eda_report():
    df = _classification_dataframe()

    report = generate_eda_report(df, "Outcome")

    assert report["report_type"] == "automated_tabular_eda"
    assert report["report_version"] == "1.0"
    assert report["report_status"] == "complete"
    assert report["failed_sections"] == []
    assert report["summary"]["rows"] == 60
    assert report["summary"]["columns"] == 5
    assert report["summary"]["problem_type"] == "classification"
    assert report["dataset_overview"] is not None
    assert report["target_statistics"] is not None
    assert report["numerical_statistics"] is not None
    assert report["categorical_statistics"] is not None
    assert report["datetime_statistics"] is not None
    assert report["feature_target_relationships"] is not None
    assert report["feature_feature_relationships"] is not None
    assert report["insights"] is not None
    assert report["insights"]["insight_report_type"] == (
        "automated_eda_insights"
    )
    assert report["section_status"]["insights"] == "success"
    assert report["visualizations"] is None
    assert report["section_status"]["visualizations"] == (
        "skipped_by_configuration"
    )
    assert report["summary"]["generated_insights"] >= 1
    assert report["summary"]["generated_visualizations"] is None
    assert report["preprocessing_context"][
        "cleaning_plan"
    ] is not None
    assert report["preprocessing_context"][
        "leakage_report"
    ] is not None
    assert report["preprocessing_context"][
        "outlier_report"
    ] is not None


def test_complete_regression_eda_report_with_override():
    df = pd.DataFrame(
        {
            "Feature": range(1, 61),
            "Department": ["A", "B", "C"] * 20,
            "Salary": [
                25_000.5 + index * 731.25
                for index in range(60)
            ],
        }
    )

    report = generate_eda_report(
        df,
        "Salary",
        problem_type="regression",
    )

    assert report["report_status"] == "complete"
    assert report["summary"]["problem_type"] == "regression"
    assert report["target_statistics"][
        "problem_type_source"
    ] == "user_override"
    assert report["execution_options"][
        "problem_type_override"
    ] == "regression"


def test_optional_sections_can_be_disabled():
    df = _classification_dataframe()

    report = generate_eda_report(
        df,
        "Outcome",
        include_feature_relationships=False,
        include_preprocessing_context=False,
        include_insights=False,
    )

    assert report["report_status"] == "complete"
    assert report["feature_target_relationships"] is None
    assert report["feature_feature_relationships"] is None
    assert report["preprocessing_context"][
        "cleaning_plan"
    ] is None
    assert report["section_status"][
        "feature_target_relationships"
    ] == "skipped_by_configuration"
    assert "cleaning_plan" in report["skipped_sections"]
    assert report["insights"] is None
    assert report["section_status"]["insights"] == (
        "skipped_by_configuration"
    )
    assert report["summary"]["generated_insights"] is None
    assert report["visualizations"] is None


def test_partial_report_records_section_errors():
    df = pd.DataFrame(
        [
            [1, 1, 0],
            [2, 2, 1],
            [3, 3, 0],
            [4, 4, 1],
        ]
        * 5,
        columns=["Feature", "Feature", "Target"],
    )

    report = generate_eda_report(
        df,
        "Target",
        fail_fast=False,
    )

    assert report["report_status"] == "partial"
    assert report["dataset_overview"] is not None
    assert len(report["failed_sections"]) > 0
    assert len(report["section_errors"]) > 0
    assert report["dataset_overview"]["schema"][
        "column_names_are_unique"
    ] is False
    insight_ids = [
        insight["insight_id"]
        for insight in report["insights"][
            "prioritized_insights"
        ]
    ]
    assert "execution_incomplete_report" in insight_ids


def test_fail_fast_reraises_section_error():
    df = pd.DataFrame(
        [
            [1, 1, 0],
            [2, 2, 1],
            [3, 3, 0],
            [4, 4, 1],
        ]
        * 5,
        columns=["Feature", "Feature", "Target"],
    )

    with pytest.raises(Exception):
        generate_eda_report(
            df,
            "Target",
            fail_fast=True,
        )


def test_configuration_is_included_in_report():
    df = _classification_dataframe()
    config = EDAConfig(
        high_correlation_threshold=0.85,
        max_pairwise_features=10,
    )

    report = generate_eda_report(
        df,
        "Outcome",
        config=config,
    )

    assert report["configuration"][
        "high_correlation_threshold"
    ] == 0.85
    assert report["configuration"][
        "max_pairwise_features"
    ] == 10


def test_eda_report_is_json_safe_and_non_mutating():
    df = _classification_dataframe()
    original_df = df.copy(deep=True)

    report = generate_eda_report(df, "Outcome")

    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(df, original_df)
    assert report[
        "automatic_data_modification_performed"
    ] is False


def test_eda_report_validates_required_input():
    df = _classification_dataframe()

    with pytest.raises(ValueError, match="does not exist"):
        generate_eda_report(
            df,
            "UnknownTarget",
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_eda_report(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )


def test_insights_work_when_expensive_sections_are_disabled():
    df = _classification_dataframe()

    report = generate_eda_report(
        df,
        "Outcome",
        include_feature_relationships=False,
        include_preprocessing_context=False,
        include_insights=True,
    )

    assert report["report_status"] == "complete"
    assert report["insights"] is not None
    assert report["insights"]["insight_count"] >= 1
    assert report["section_status"]["insights"] == "success"
    assert report["section_status"][
        "feature_target_relationships"
    ] == "skipped_by_configuration"


def test_integrated_insight_summary_matches_insight_report():
    df = _classification_dataframe()

    report = generate_eda_report(df, "Outcome")
    insights = report["insights"]

    assert report["summary"]["generated_insights"] == (
        insights["generated_insight_count"]
    )
    assert report["summary"]["critical_insights"] == (
        insights["severity_counts"]["critical"]
    )
    assert report["summary"]["high_insights"] == (
        insights["severity_counts"]["high"]
    )
    json.dumps(report, allow_nan=False)


def test_visualizations_integrate_with_central_report(tmp_path):
    df = _classification_dataframe()
    output_directory = tmp_path / "plots"

    report = generate_eda_report(
        df,
        "Outcome",
        include_visualizations=True,
        visualization_output_directory=output_directory,
    )

    visualizations = report["visualizations"]
    assert report["report_status"] == "complete"
    assert report["section_status"]["visualizations"] == (
        "success"
    )
    assert visualizations["visualization_status"] == "complete"
    assert visualizations["generated_plot_count"] >= 1
    assert report["summary"]["generated_visualizations"] == (
        visualizations["generated_plot_count"]
    )
    assert report["summary"]["visualization_errors"] == 0

    for plot in visualizations["generated_plots"]:
        assert (output_directory / plot["file_name"]).exists()

    json.dumps(report, allow_nan=False)


def test_visualizations_require_explicit_output_directory():
    df = _classification_dataframe()

    with pytest.raises(
        ValueError,
        match="visualization_output_directory",
    ):
        generate_eda_report(
            df,
            "Outcome",
            include_visualizations=True,
        )


def test_visualization_failure_isolated_in_central_report(tmp_path):
    df = _classification_dataframe()
    invalid_directory = tmp_path / "not_a_directory"
    invalid_directory.write_text("blocking file", encoding="utf-8")

    report = generate_eda_report(
        df,
        "Outcome",
        include_visualizations=True,
        visualization_output_directory=invalid_directory,
        fail_fast=False,
    )

    assert report["report_status"] == "partial"
    assert report["section_status"]["visualizations"] == "error"
    assert report["visualizations"] is None
    assert "visualizations" in report["failed_sections"]
    assert any(
        error["section"] == "visualizations"
        for error in report["section_errors"]
    )