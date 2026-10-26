from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from ml_pipeline.professional_report import (
    MARKDOWN_REPORT_FILENAME,
    PDF_REPORT_FILENAME,
    generate_professional_report,
)


def _training_report() -> dict:
    return {
        "training_status": "complete",
        "target_column": "Outcome",
        "problem_type": "classification",
        "problem_type_source": "inferred",
        "primary_metric": "balanced_accuracy",
        "primary_metric_direction": "higher",
        "split": {
            "original_rows": 100,
            "usable_rows": 98,
            "missing_target_rows_removed": 2,
            "training_rows": 78,
            "test_rows": 20,
            "stratified": True,
        },
        "cross_validation": {
            "used_folds": 3,
            "splitter": "StratifiedKFold",
        },
        "feature_selection": {
            "method": "SelectPercentile(f_classif)",
            "percentile": 75,
            "selected_feature_count": 3,
            "selected_feature_names": ["Age", "Signal", "Region_North"],
        },
        "model_recommendation": {
            "selection_mode": "automatic_recommendation",
            "registered_non_baseline_model_count": 14,
            "selected_model_count": 3,
        },
        "model_comparisons": [],
        "hyperparameter_tuning": {
            "status": "complete",
            "attempted_model_count": 2,
            "successful_model_count": 2,
            "failed_model_count": 0,
            "selected_model_name": "hist_gradient_boosting",
            "selected_variant": "tuned",
        },
        "best_model": {
            "model_name": "hist_gradient_boosting",
            "display_name": "Histogram Gradient Boosting Classifier",
            "selected_variant": "tuned",
            "cross_validation_primary_score": 0.81,
        },
        "holdout_metrics": {
            "accuracy": 0.80,
            "balanced_accuracy": 0.79,
            "f1_weighted": 0.80,
        },
        "preprocessing_report": {
            "preprocessing_metadata": {
                "model_features": ["Age", "Signal", "Region"],
                "numerical_features": ["Age", "Signal"],
                "categorical_features": ["Region"],
                "standard_scaler_features": ["Age"],
                "robust_scaler_features": ["Signal"],
                "excluded_review_features": ["Notes"],
            },
            "cleaning_plan": {
                "drop_columns": ["RecordID"],
                "imputation_plan": {"Age": "median"},
            },
            "leakage_report": {
                "potential_leakage_columns": [],
            },
            "outlier_report": {"rows_removed": 0},
        },
        "model_selection_rationale": {
            "summary": [
                "Histogram Gradient Boosting ranked first using training-only cross-validation.",
                "The untouched holdout did not influence the winner.",
            ],
            "winner": {
                "rank": 1,
                "model_name": "hist_gradient_boosting",
                "display_name": "Histogram Gradient Boosting Classifier",
                "family": "boosting",
                "selected_variant": "tuned",
                "cross_validation_primary_score": 0.81,
                "cross_validation_standard_deviation": 0.02,
                "selected_because": [
                    "highest_validated_primary_score_after_bounded_tuning",
                    "selected_using_training_cross_validation_only",
                    "beats_dummy_baseline",
                ],
            },
            "runner_up": {
                "rank": 2,
                "model_name": "logistic_regression",
                "display_name": "Logistic Regression",
                "family": "linear",
                "selected_variant": "untuned",
                "cross_validation_primary_score": 0.78,
                "cross_validation_standard_deviation": 0.03,
                "score_gap_from_winner": 0.03,
            },
            "model_ranking": [
                {
                    "rank": 1,
                    "display_name": "Histogram Gradient Boosting Classifier",
                    "family": "boosting",
                    "selected_variant": "tuned",
                    "cross_validation_primary_score": 0.81,
                    "cross_validation_standard_deviation": 0.02,
                },
                {
                    "rank": 2,
                    "display_name": "Logistic Regression",
                    "family": "linear",
                    "selected_variant": "untuned",
                    "cross_validation_primary_score": 0.78,
                    "cross_validation_standard_deviation": 0.03,
                },
            ],
            "baseline_assessment": {
                "baseline_primary_score": 0.50,
                "winner_beats_baseline": True,
                "absolute_improvement": 0.31,
                "relative_improvement": 0.62,
            },
            "decision_confidence": {
                "level": "moderate",
                "close_competition": False,
                "warnings": [],
            },
        },
        "explainability": {
            "explainability_status": "complete",
            "evaluation_scope": "untouched_holdout",
            "used_for_model_selection": False,
            "permutation_importance": {
                "features": [
                    {
                        "rank": 1,
                        "feature": "Signal",
                        "importance_mean": 0.18,
                        "importance_standard_deviation": 0.02,
                    },
                    {
                        "rank": 2,
                        "feature": "Age",
                        "importance_mean": 0.07,
                        "importance_standard_deviation": 0.01,
                    },
                ]
            },
        },
    }


def _eda_report() -> dict:
    return {
        "dataset_overview": {
            "shape": {"rows": 100, "columns": 6},
            "missingness": {
                "total_missing_cells": 8,
                "total_missing_percentage": 1.33,
            },
            "duplicates": {"duplicate_rows": 1},
        },
        "insights": {
            "prioritized_insights": [
                {
                    "priority_rank": 1,
                    "severity": "medium",
                    "title": "Signal contains outliers",
                    "recommendation": "Use robust scaling and review extremes.",
                }
            ],
            "limitations": ["Rules do not replace domain expertise."],
        },
        "warnings": ["One feature needs manual review."],
        "limitations": ["Associations do not establish causality."],
    }


def test_markdown_report_contains_decision_evidence(tmp_path):
    result = generate_professional_report(
        _training_report(),
        tmp_path,
        eda_report=_eda_report(),
        dataset_filename="example.csv",
        formats=["markdown"],
    )

    text = (tmp_path / MARKDOWN_REPORT_FILENAME).read_text("utf-8")
    assert result["report_status"] == "complete"
    assert "Why this model was selected" in text
    assert "Histogram Gradient Boosting Classifier" in text
    assert "Logistic Regression" in text
    assert "untouched holdout" in text.lower()
    assert "Private holdout rows included: `False`" in text


def test_pdf_report_has_valid_signature_and_metadata(tmp_path):
    result = generate_professional_report(
        _training_report(),
        tmp_path,
        eda_report=_eda_report(),
        formats=["pdf"],
    )
    pdf_path = tmp_path / PDF_REPORT_FILENAME

    assert pdf_path.read_bytes().startswith(b"%PDF-")
    assert pdf_path.stat().st_size > 5_000
    assert result["files"]["professional_report_pdf"]["sha256"]
    assert result["private_holdout_rows_included"] is False


def test_both_formats_are_created_without_mutating_sources(tmp_path):
    training = _training_report()
    eda = _eda_report()
    original_training = deepcopy(training)
    original_eda = deepcopy(eda)

    result = generate_professional_report(
        training,
        tmp_path,
        eda_report=eda,
    )

    assert training == original_training
    assert eda == original_eda
    assert result["source_training_report_modified"] is False
    assert result["source_eda_report_modified"] is False
    assert (tmp_path / MARKDOWN_REPORT_FILENAME).exists()
    assert (tmp_path / PDF_REPORT_FILENAME).exists()


def test_chart_is_embedded_and_path_is_bounded(tmp_path):
    chart_path = tmp_path / "importance.png"
    Image.new("RGB", (640, 360), color=(232, 244, 250)).save(chart_path)

    result = generate_professional_report(
        _training_report(),
        tmp_path,
        chart_paths={"importance_chart": chart_path},
        formats=["markdown", "pdf"],
    )

    markdown = (tmp_path / MARKDOWN_REPORT_FILENAME).read_text("utf-8")
    assert "![Importance chart](importance.png)" in markdown
    assert result["chart_count"] == 1
    assert result["embedded_pdf_chart_count"] == 1


def test_chart_outside_output_directory_is_rejected(tmp_path):
    output_directory = tmp_path / "bundle"
    outside = tmp_path / "outside.png"
    Image.new("RGB", (20, 20), color="white").save(outside)

    with pytest.raises(ValueError, match="inside output_directory"):
        generate_professional_report(
            _training_report(),
            output_directory,
            chart_paths={"outside": outside},
            formats=["markdown"],
        )


def test_invalid_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported report formats"):
        generate_professional_report(
            _training_report(),
            tmp_path,
            formats=["docx"],
        )


def test_missing_required_training_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing required fields"):
        generate_professional_report(
            {"training_status": "complete"},
            tmp_path,
            formats=["markdown"],
        )


def test_markdown_generation_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_professional_report(
        _training_report(),
        first,
        eda_report=_eda_report(),
        dataset_filename="example.csv",
        formats=["markdown"],
    )
    generate_professional_report(
        _training_report(),
        second,
        eda_report=_eda_report(),
        dataset_filename="example.csv",
        formats=["markdown"],
    )

    assert (first / MARKDOWN_REPORT_FILENAME).read_bytes() == (
        second / MARKDOWN_REPORT_FILENAME
    ).read_bytes()
