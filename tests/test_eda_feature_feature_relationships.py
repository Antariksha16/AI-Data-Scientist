import json

import pandas as pd
import pytest

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda_relationships import (
    generate_feature_feature_relationships,
)


def test_numerical_matrices_and_redundancy_groups():
    df = pd.DataFrame(
        {
            "FeatureA": range(1, 101),
            "FeatureB": [2 * value for value in range(1, 101)],
            "Noise": [(value * 17) % 31 for value in range(100)],
            "Target": [0, 1] * 50,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )

    assert report["pearson_matrix"]["FeatureA"][
        "FeatureB"
    ] == 1.0
    assert report["spearman_matrix"]["FeatureA"][
        "FeatureB"
    ] == 1.0
    assert report["high_association_pair_count"] >= 1
    assert ["FeatureA", "FeatureB"] in report[
        "redundancy_groups"
    ]
    assert report["automatic_feature_removal_performed"] is False


def test_categorical_cramers_v_matrix():
    df = pd.DataFrame(
        {
            "CategoryA": ["A", "B", "C"] * 30,
            "CategoryB": ["X", "Y", "Z"] * 30,
            "Target": [0, 1] * 45,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )

    assert report["cramers_v_matrix"]["CategoryA"][
        "CategoryB"
    ] == 1.0


def test_mixed_numerical_categorical_eta_matrix():
    df = pd.DataFrame(
        {
            "Department": ["IT", "HR", "Sales"] * 30,
            "Score": [
                score
                for _ in range(30)
                for score in (90.0, 40.0, 65.0)
            ],
            "Target": [0, 1] * 45,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )

    assert report["mixed_eta_matrix"]["Score"][
        "Department"
    ] == 1.0


def test_vif_detects_perfect_multicollinearity():
    df = pd.DataFrame(
        {
            "FeatureA": range(1, 101),
            "FeatureB": [3 * value for value in range(1, 101)],
            "FeatureC": [(value * 7) % 19 for value in range(100)],
            "Target": [0, 1] * 50,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )
    vif_report = report["multicollinearity_report"]

    assert vif_report["status"] == "analyzed"
    assert "FeatureA" in vif_report["high_vif_features"]
    assert "FeatureB" in vif_report["high_vif_features"]

    statistics = {
        item["feature"]: item
        for item in vif_report["vif_statistics"]
    }
    assert statistics["FeatureA"]["is_infinite"] is True
    assert statistics["FeatureB"]["is_infinite"] is True
    assert statistics["FeatureA"]["vif"] is None


def test_high_cardinality_and_feature_limit_safeguards():
    df = pd.DataFrame(
        {
            "FeatureA": range(120),
            "FeatureB": [value % 7 for value in range(120)],
            "FeatureC": [value % 11 for value in range(120)],
            "RecordCode": [
                f"record_{value}"
                for value in range(120)
            ],
            "Target": [0, 1] * 60,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
        EDAConfig(
            max_pairwise_features=2,
            max_relationship_categories=100,
        ),
    )

    assert report["selected_feature_count"] == 2
    assert "RecordCode" in report[
        "excluded_high_cardinality_features"
    ]
    assert len(report["omitted_due_feature_limit"]) == 1


def test_datetime_feature_participates_as_numerical():
    df = pd.DataFrame(
        {
            "EventDate": pd.date_range(
                "2025-01-01",
                periods=40,
                freq="D",
            ),
            "Elapsed": range(40),
            "Target": [0, 1] * 20,
        }
    )

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )

    assert "EventDate" in report["datetime_features"]
    assert report["pearson_matrix"]["EventDate"][
        "Elapsed"
    ] == 1.0


def test_pairwise_report_is_json_safe_and_non_mutating():
    df = pd.DataFrame(
        {
            "FeatureA": range(30),
            "FeatureB": [value % 5 for value in range(30)],
            "Category": ["A", "B", "C"] * 10,
            "Target": [0, 1] * 15,
        }
    )
    original_df = df.copy(deep=True)

    report = generate_feature_feature_relationships(
        df,
        "Target",
    )

    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(df, original_df)
    assert "Target" not in report["selected_features"]


def test_pairwise_input_validation():
    df = pd.DataFrame(
        {
            "Feature": range(20),
            "Target": [0, 1] * 10,
        }
    )

    with pytest.raises(ValueError, match="does not exist"):
        generate_feature_feature_relationships(
            df,
            "UnknownTarget",
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_feature_feature_relationships(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )