import copy
import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.eda_insights import (
    generate_eda_insights,
)


def _base_report() -> dict:
    return {
        "report_type": "automated_tabular_eda",
        "report_version": "1.0",
        "report_status": "complete",
        "target_column": "Outcome",
        "configuration": {
            "high_missing_threshold": 0.60,
            "high_outlier_ratio_threshold": 0.05,
        },
        "failed_sections": [],
        "section_errors": [],
        "dataset_overview": {
            "schema": {
                "column_names_are_unique": True,
                "duplicate_column_names": [],
            },
            "missingness": {
                "high_missing_columns": [],
                "high_missing_rows": 0,
            },
            "duplicates": {
                "duplicate_rows": 0,
                "identical_column_pairs": [],
            },
            "quality_flags": {
                "high_missing_columns": [],
                "constant_columns": [],
                "near_constant_columns": [],
                "identifier_candidates": [],
                "infinite_value_columns": [],
                "unsupported_columns": [],
            },
            "basic_modeling_readiness": {
                "ready": True,
                "blockers": [],
            },
            "column_profiles": [],
        },
        "target_statistics": {
            "problem_type": "classification",
            "training_usable": True,
            "missing_target_rows": 0,
            "classification_statistics": {
                "imbalance_level": "balanced",
                "rare_classes": [],
            },
            "regression_statistics": None,
        },
        "numerical_statistics": {
            "numerical_statistics": {},
        },
        "categorical_statistics": {
            "categorical_statistics": {},
        },
        "datetime_statistics": {
            "datetime_columns": [],
            "datetime_statistics": {},
        },
        "feature_target_relationships": {
            "potential_leakage_features": [],
            "ranked_features": [],
            "skipped_features": [],
        },
        "feature_feature_relationships": {
            "high_association_pairs": [],
            "redundancy_groups": [],
            "multicollinearity_report": {
                "high_vif_features": [],
            },
        },
        "preprocessing_context": {
            "cleaning_plan": {
                "drop_columns": [],
                "imputation_plan": {},
                "review_columns": {},
            },
            "leakage_report": {
                "potential_leakage_columns": [],
            },
            "outlier_report": {
                "robust_scaler_features": [],
                "outlier_summary": {},
            },
        },
    }


def _insight_ids(result: dict) -> list[str]:
    return [
        insight["insight_id"]
        for insight in result["prioritized_insights"]
    ]


def test_healthy_report_produces_safe_overview_insight():
    result = generate_eda_insights(_base_report())

    assert result["insight_report_type"] == (
        "automated_eda_insights"
    )
    assert result["insight_report_version"] == "1.0"
    assert result["source_report_status"] == "complete"
    assert result["insight_count"] == 1
    assert _insight_ids(result) == [
        "overview_no_automatic_concerns"
    ]
    assert result["severity_counts"]["critical"] == 0
    assert result["automatic_data_modification_performed"] is False


def test_schema_and_quality_problems_are_prioritized():
    report = _base_report()
    overview = report["dataset_overview"]
    overview["schema"] = {
        "column_names_are_unique": False,
        "duplicate_column_names": ["Age"],
    }
    overview["basic_modeling_readiness"] = {
        "ready": False,
        "blockers": ["Column names must be unique."],
    }
    overview["missingness"]["high_missing_columns"] = [
        "MostlyMissing"
    ]
    overview["duplicates"]["duplicate_rows"] = 4
    overview["duplicates"]["duplicate_row_percentage"] = 5.0
    overview["quality_flags"]["high_missing_columns"] = [
        "MostlyMissing"
    ]
    overview["quality_flags"]["identifier_candidates"] = [
        "RecordID"
    ]

    result = generate_eda_insights(report)
    ids = _insight_ids(result)

    assert ids[0] == "schema_duplicate_column_names"
    assert result["prioritized_insights"][0]["severity"] == (
        "critical"
    )
    assert "schema_modeling_readiness_blocked" in ids
    assert "quality_high_missing_columns" in ids
    assert "quality_duplicate_rows" in ids
    assert "quality_identifier_candidates" in ids


def test_classification_imbalance_and_rare_classes():
    report = _base_report()
    classification = report["target_statistics"][
        "classification_statistics"
    ]
    classification.update(
        {
            "imbalance_level": "severe",
            "minority_class": 1,
            "minority_class_percentage": 2.0,
            "minority_to_majority_ratio": 0.0204,
            "recommended_metrics": [
                "balanced_accuracy",
                "average_precision",
            ],
            "rare_classes": [
                {
                    "class_value": 1,
                    "count": 2,
                    "percentage": 2.0,
                }
            ],
            "maximum_stratified_cv_folds": 2,
        }
    )

    result = generate_eda_insights(report)
    insights = {
        item["insight_id"]: item
        for item in result["prioritized_insights"]
    }

    assert insights["target_class_imbalance"]["severity"] == "high"
    assert insights["target_class_imbalance"]["evidence"][
        "minority_to_majority_ratio"
    ] == 0.0204
    assert "target_rare_classes" in insights


def test_regression_target_distribution_and_outliers():
    report = _base_report()
    report["target_statistics"] = {
        "problem_type": "regression",
        "training_usable": True,
        "missing_target_rows": 0,
        "classification_statistics": None,
        "regression_statistics": {
            "distribution_shape": "right_skewed",
            "skewness": 4.7,
            "kurtosis": 33.0,
            "target_transformation_recommendation": (
                "log1p_review"
            ),
            "outlier_count": 12,
            "outlier_percentage": 12.0,
        },
    }

    result = generate_eda_insights(report)
    ids = _insight_ids(result)

    assert "target_regression_distribution" in ids
    assert "target_regression_outliers" in ids
    assert result["category_counts"]["target"] == 2
    distribution_insight = next(
        item
        for item in result["prioritized_insights"]
        if item["insight_id"]
        == "target_regression_distribution"
    )
    assert distribution_insight["evidence"][
        "transformation_recommendation"
    ] == "log1p_review"


def test_distribution_and_temporal_insights_are_generated():
    report = _base_report()
    report["numerical_statistics"]["numerical_statistics"] = {
        "Fare": {
            "missing_values": 0,
            "outlier_percentage": 13.02,
            "recommended_scaler": "robust",
            "distribution_shape": "right_skewed",
            "skewness": 4.7873,
            "transformation_recommendation": "log1p_review",
        }
    }
    report["categorical_statistics"][
        "categorical_statistics"
    ] = {
        "Ticket": {
            "high_cardinality": True,
            "unique_values": 681,
            "recommended_encoding": "frequency_or_hashing_review",
            "rare_category_count": 681,
            "rare_observation_percentage": 100.0,
            "missing_values": 0,
            "dominant_category": False,
        }
    }
    report["datetime_statistics"] = {
        "datetime_columns": ["EventDate"],
        "datetime_statistics": {
            "EventDate": {
                "invalid_datetime_values": 2,
                "invalid_datetime_percentage": 2.0,
            }
        },
    }
    report["preprocessing_context"]["outlier_report"] = {
        "outlier_summary": {
            "Fare": {
                "outlier_percentage": 13.02,
            }
        },
        "robust_scaler_features": ["Fare"],
    }

    result = generate_eda_insights(report)
    ids = _insight_ids(result)

    assert "numerical_feature_outliers" in ids
    assert "numerical_skewed_features" in ids
    assert "categorical_high_cardinality" in ids
    assert "categorical_rare_categories" in ids
    assert "datetime_invalid_values" in ids
    assert "datetime_split_review" in ids


def test_discrete_numerical_features_do_not_create_continuous_warnings():
    report = _base_report()
    report["dataset_overview"]["column_profiles"] = [
        {
            "column": "EventCount",
            "semantic_type": "discrete_numerical",
        },
        {
            "column": "ContinuousValue",
            "semantic_type": "continuous_numerical",
        },
    ]
    report["numerical_statistics"]["numerical_statistics"] = {
        "EventCount": {
            "missing_values": 0,
            "outlier_percentage": 20.0,
            "recommended_scaler": "robust",
            "distribution_shape": "right_skewed",
            "skewness": 3.0,
            "transformation_recommendation": "log1p_review",
        },
        "ContinuousValue": {
            "missing_values": 0,
            "outlier_percentage": 10.0,
            "recommended_scaler": "robust",
            "distribution_shape": "right_skewed",
            "skewness": 2.0,
            "transformation_recommendation": "log1p_review",
        },
    }
    report["preprocessing_context"]["outlier_report"] = {
        "outlier_summary": {
            "ContinuousValue": {
                "outlier_percentage": 10.0,
            }
        },
        "skipped_discrete_features": ["EventCount"],
        "robust_scaler_features": ["ContinuousValue"],
    }

    result = generate_eda_insights(report)
    insights = {
        item["insight_id"]: item
        for item in result["prioritized_insights"]
    }

    assert insights["numerical_feature_outliers"][
        "affected_columns"
    ] == ["ContinuousValue"]
    assert insights["numerical_skewed_features"][
        "affected_columns"
    ] == ["ContinuousValue"]


def test_leakage_and_multicollinearity_are_high_priority():
    report = _base_report()
    report["feature_target_relationships"][
        "potential_leakage_features"
    ] = ["OutcomeCopy"]
    report["preprocessing_context"]["leakage_report"] = {
        "potential_leakage_columns": [
            "OutcomeCopy",
            "PostOutcomeStatus",
        ],
        "leakage_evidence": {
            "OutcomeCopy": ["exact copy"],
        },
    }
    report["feature_feature_relationships"].update(
        {
            "high_association_pairs": [
                {
                    "feature_1": "FeatureA",
                    "feature_2": "FeatureB",
                    "primary_score": 1.0,
                }
            ],
            "redundancy_groups": [["FeatureA", "FeatureB"]],
            "multicollinearity_report": {
                "high_vif_features": ["FeatureA", "FeatureB"],
                "vif_statistics": [],
            },
        }
    )

    result = generate_eda_insights(report)
    insights = {
        item["insight_id"]: item
        for item in result["prioritized_insights"]
    }

    assert insights[
        "relationship_potential_target_leakage"
    ]["severity"] == "critical"
    assert insights[
        "preprocessing_statistical_leakage"
    ]["severity"] == "critical"
    assert insights["relationship_high_vif"]["severity"] == "high"
    assert "relationship_redundant_features" in insights


def test_preprocessing_plan_becomes_explainable_insights():
    report = _base_report()
    report["preprocessing_context"].update(
        {
            "cleaning_plan": {
                "drop_columns": ["RecordID", "Cabin"],
                "drop_reasons": {
                    "RecordID": ["identifier-like column"],
                    "Cabin": ["high missingness"],
                },
                "imputation_plan": {
                    "Age": "median",
                    "Region": "most_frequent",
                },
                "review_columns": {
                    "Description": "high-cardinality categorical feature"
                },
            },
            "leakage_report": {
                "potential_leakage_columns": [],
            },
            "outlier_report": {
                "robust_scaler_features": ["Fare"],
            },
        }
    )

    result = generate_eda_insights(report)
    ids = _insight_ids(result)

    assert "preprocessing_planned_feature_drops" in ids
    assert "preprocessing_imputation_plan" in ids
    assert "preprocessing_manual_review_columns" in ids
    assert "preprocessing_robust_scaling" in ids


def test_partial_report_creates_execution_insight():
    report = _base_report()
    report["report_status"] = "partial"
    report["failed_sections"] = ["datetime_statistics"]
    report["section_errors"] = [
        {
            "section": "datetime_statistics",
            "error_type": "ValueError",
            "message": "invalid date",
        }
    ]
    report["datetime_statistics"] = None

    result = generate_eda_insights(report)
    insight = result["prioritized_insights"][0]

    assert insight["insight_id"] == "execution_incomplete_report"
    assert insight["severity"] == "medium"
    assert insight["evidence"]["failed_sections"] == [
        "datetime_statistics"
    ]


def test_output_is_ranked_truncated_json_safe_and_non_mutating():
    report = _base_report()
    report["dataset_overview"]["quality_flags"].update(
        {
            "high_missing_columns": ["Cabin"],
            "identifier_candidates": ["PassengerId"],
        }
    )
    report["dataset_overview"]["missingness"][
        "high_missing_columns"
    ] = ["Cabin"]
    report["numerical_statistics"]["numerical_statistics"] = {
        "Feature": {
            "missing_values": np.int64(2),
            "outlier_percentage": np.float64(5.5),
            "recommended_scaler": "robust",
            "distribution_shape": "right_skewed",
            "skewness": np.float64(2.0),
            "transformation_recommendation": "log1p_review",
        }
    }
    report["generated_at"] = pd.Timestamp("2026-08-25")
    original_report = copy.deepcopy(report)

    result = generate_eda_insights(report, max_insights=2)

    assert result["insight_count"] == 2
    assert result["generated_insight_count"] > 2
    assert result["insights_truncated"] is True
    assert [
        item["priority_rank"]
        for item in result["prioritized_insights"]
    ] == [1, 2]
    assert result["prioritized_insights"][0]["severity"] in {
        "critical",
        "high",
    }
    json.dumps(result, allow_nan=False)

    assert report.keys() == original_report.keys()
    assert report["generated_at"] == original_report["generated_at"]
    assert np.array_equal(
        np.array(
            report["numerical_statistics"]["numerical_statistics"][
                "Feature"
            ]["missing_values"]
        ),
        np.array(
            original_report["numerical_statistics"][
                "numerical_statistics"
            ]["Feature"]["missing_values"]
        ),
    )


def test_input_validation():
    with pytest.raises(TypeError, match="dictionary"):
        generate_eda_insights([])

    with pytest.raises(ValueError, match="target_column"):
        generate_eda_insights({})

    with pytest.raises(TypeError, match="integer"):
        generate_eda_insights(
            {"target_column": "Target"},
            max_insights=2.5,
        )

    with pytest.raises(ValueError, match="greater than zero"):
        generate_eda_insights(
            {"target_column": "Target"},
            max_insights=0,
        )
