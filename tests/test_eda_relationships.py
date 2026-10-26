import json

import pandas as pd
import pytest

from ml_pipeline.eda_config import EDAConfig
from ml_pipeline.eda_relationships import (
    generate_feature_target_relationships,
)


def test_numerical_regression_relationships_are_ranked():
    df = pd.DataFrame(
        {
            "StrongFeature": range(1, 101),
            "WeakFeature": [
                (index * 17) % 31
                for index in range(100)
            ],
            "Target": [
                5.0 * index + (index % 3)
                for index in range(1, 101)
            ],
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Target",
        problem_type="regression",
    )
    strong = report["feature_relationships"][
        "StrongFeature"
    ]

    assert report["problem_type"] == "regression"
    assert strong["status"] == "analyzed"
    assert strong["feature_type"] == "numerical"
    assert strong["pearson_correlation"] > 0.99
    assert strong["spearman_correlation"] > 0.99
    assert strong["relationship_strength"] == "strong"
    assert report["ranked_features"][0]["feature"] == (
        "StrongFeature"
    )


def test_numerical_classification_uses_eta_squared():
    df = pd.DataFrame(
        {
            "Score": list(range(30))
            + list(range(70, 100)),
            "Outcome": ["No"] * 30 + ["Yes"] * 30,
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Outcome",
    )
    relationship = report["feature_relationships"]["Score"]

    assert relationship["status"] == "analyzed"
    assert relationship["analysis_method"] == (
        "eta_squared_and_group_statistics"
    )
    assert relationship["eta"] > 0.80
    assert relationship["relationship_strength"] == "strong"
    assert relationship[
        "standardized_mean_difference"
    ] is not None


def test_categorical_classification_uses_cramers_v():
    df = pd.DataFrame(
        {
            "Segment": ["A", "B", "C"] * 30,
            "Outcome": ["Yes", "No", "Pending"] * 30,
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Outcome",
    )
    relationship = report["feature_relationships"][
        "Segment"
    ]

    assert relationship["analysis_method"] == (
        "bias_corrected_cramers_v"
    )
    assert relationship["cramers_v"] == 1.0
    assert relationship["relationship_strength"] == "strong"
    assert relationship["potential_leakage"] is True
    assert report["automatic_feature_removal_performed"] is False


def test_categorical_regression_uses_correlation_ratio():
    df = pd.DataFrame(
        {
            "Department": ["IT", "HR", "Sales"] * 30,
            "Salary": [
                value
                for _ in range(30)
                for value in (90_000.0, 40_000.0, 65_000.0)
            ],
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Salary",
        problem_type="regression",
    )
    relationship = report["feature_relationships"][
        "Department"
    ]

    assert relationship["analysis_method"] == (
        "correlation_ratio_eta"
    )
    assert relationship["eta"] == 1.0
    assert relationship["relationship_strength"] == "strong"


def test_native_datetime_regression_relationship():
    df = pd.DataFrame(
        {
            "EventDate": pd.date_range(
                "2025-01-01",
                periods=40,
                freq="D",
            ),
            "Demand": [100.0 + index * 4 for index in range(40)],
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Demand",
        problem_type="regression",
    )
    relationship = report["feature_relationships"][
        "EventDate"
    ]

    assert relationship["feature_type"] == "datetime"
    assert relationship["status"] == "analyzed"
    assert relationship["pearson_correlation"] == 1.0
    assert relationship["relationship_strength"] == "strong"
    assert "temporal_encoding" in relationship


def test_text_datetime_regression_relationship():
    df = pd.DataFrame(
        {
            "EventDate": [
                f"2025-03-{day:02d}"
                for day in range(1, 29)
            ],
            "Demand": [50.0 + index * 2 for index in range(28)],
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Demand",
        problem_type="regression",
    )
    relationship = report["feature_relationships"][
        "EventDate"
    ]

    assert relationship["feature_type"] == "datetime"
    assert relationship["status"] == "analyzed"
    assert relationship["spearman_correlation"] == 1.0


def test_high_cardinality_categorical_feature_is_skipped():
    df = pd.DataFrame(
        {
            "RecordCode": [
                f"record_{index}"
                for index in range(120)
            ],
            "Outcome": ["Yes", "No"] * 60,
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Outcome",
        EDAConfig(max_relationship_categories=100),
    )
    relationship = report["feature_relationships"][
        "RecordCode"
    ]

    assert relationship["status"] == "skipped"
    assert "safety limit" in relationship["warnings"][0]
    assert "RecordCode" in report["skipped_features"]


def test_insufficient_pairwise_sample_is_skipped():
    df = pd.DataFrame(
        {
            "Feature": range(10),
            "Target": [float(index) for index in range(10)],
        }
    )

    report = generate_feature_target_relationships(
        df,
        "Target",
        problem_type="regression",
    )
    relationship = report["feature_relationships"]["Feature"]

    assert relationship["status"] == "skipped"
    assert relationship["sample_size"] == 10
    assert "Too few" in relationship["warnings"][0]


def test_relationship_report_is_json_safe_and_non_mutating():
    df = pd.DataFrame(
        {
            "Feature": range(30),
            "Category": ["A", "B", "C"] * 10,
            "Target": [0, 1] * 15,
        }
    )
    original_df = df.copy(deep=True)

    report = generate_feature_target_relationships(
        df,
        "Target",
        problem_type="classification",
    )

    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(df, original_df)
    assert "Target" not in report["feature_relationships"]


def test_relationship_input_validation():
    df = pd.DataFrame(
        {
            "Feature": range(20),
            "Target": [0, 1] * 10,
        }
    )

    with pytest.raises(ValueError, match="does not exist"):
        generate_feature_target_relationships(
            df,
            "UnknownTarget",
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_feature_target_relationships(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )