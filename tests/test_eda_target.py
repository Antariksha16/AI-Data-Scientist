import json

import pandas as pd
import pytest

from backend.app.pipeline.eda_target import (
    generate_target_statistics,
)


def test_severe_binary_class_imbalance():
    df = pd.DataFrame(
        {
            "Feature": range(100),
            "Outcome": ["Approved"] * 95
            + ["Rejected"] * 5,
        }
    )

    report = generate_target_statistics(
        df,
        "Outcome",
    )
    statistics = report["classification_statistics"]

    assert report["problem_type"] == "classification"
    assert report["problem_type_source"] == "target_analyzer"
    assert report["training_usable"] is True
    assert statistics["target_structure"] == "binary"
    assert statistics["majority_class_count"] == 95
    assert statistics["minority_class_count"] == 5
    assert statistics["imbalance_level"] == "severe"
    assert statistics["is_imbalanced"] is True
    assert statistics["automatic_resampling_performed"] is False


def test_balanced_multiclass_target():
    df = pd.DataFrame(
        {
            "Feature": range(30),
            "Outcome": ["A", "B", "C"] * 10,
        }
    )

    report = generate_target_statistics(
        df,
        "Outcome",
    )
    statistics = report["classification_statistics"]

    assert statistics["target_structure"] == "multiclass"
    assert statistics["class_count"] == 3
    assert statistics["imbalance_level"] == "balanced"
    assert statistics["minority_to_majority_ratio"] == 1.0
    assert statistics["normalized_entropy"] == 1.0


def test_regression_target_statistics():
    df = pd.DataFrame(
        {
            "Feature": range(22),
            "Salary": [
                30_000 + index * 500
                for index in range(21)
            ]
            + [1_000_000],
        }
    )

    report = generate_target_statistics(
        df,
        "Salary",
        problem_type="regression",
    )
    statistics = report["regression_statistics"]

    assert report["problem_type"] == "regression"
    assert report["problem_type_source"] == "user_override"
    assert report["training_usable"] is True
    assert statistics["finite_count"] == 22
    assert statistics["unique_finite_values"] == 22
    assert statistics["outlier_count"] == 1
    assert statistics["distribution_shape"] == "right_skewed"
    assert (
        statistics["target_transformation_recommendation"]
        == "log1p_review"
    )
    assert (
        statistics["automatic_target_transformation_performed"]
        is False
    )


def test_missing_target_rows_are_reported_and_excluded():
    df = pd.DataFrame(
        {
            "Feature": [10, 20, 30, 40, 50, 60],
            "Outcome": ["Yes", "No", None, "Yes", None, "No"],
        }
    )

    report = generate_target_statistics(
        df,
        "Outcome",
    )

    assert report["total_rows"] == 6
    assert report["observed_target_rows"] == 4
    assert report["missing_target_rows"] == 2
    assert report["missing_target_percentage"] == 33.33
    assert sum(
        item["count"]
        for item in report["classification_statistics"][
            "class_distribution"
        ]
    ) == 4


def test_constant_target_is_not_training_usable():
    df = pd.DataFrame(
        {
            "Feature": range(10),
            "Outcome": ["Same"] * 10,
        }
    )

    report = generate_target_statistics(
        df,
        "Outcome",
        problem_type="classification",
    )

    assert report["training_usable"] is False
    assert report["classification_statistics"][
        "target_structure"
    ] == "single_class"


def test_invalid_regression_values_are_not_training_usable():
    df = pd.DataFrame(
        {
            "Feature": [1, 2, 3, 4],
            "Target": [10.0, 20.0, float("inf"), 40.0],
        }
    )

    report = generate_target_statistics(
        df,
        "Target",
        problem_type="regression",
    )

    assert report["training_usable"] is False
    assert report["regression_statistics"][
        "infinite_values"
    ] == 1


def test_target_report_is_json_safe_and_input_is_unchanged():
    df = pd.DataFrame(
        {
            "Feature": [1, 2, 3, 4],
            "Target": [0, 1, 0, 1],
        }
    )
    original_df = df.copy(deep=True)

    report = generate_target_statistics(
        df,
        "Target",
        problem_type="classification",
    )

    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(df, original_df)
    assert report[
        "automatic_target_modification_performed"
    ] is False


def test_target_statistics_validate_input():
    df = pd.DataFrame(
        {
            "Feature": [1, 2],
            "Target": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="does not exist"):
        generate_target_statistics(
            df,
            "UnknownTarget",
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_target_statistics(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )

    with pytest.raises(
        ValueError,
        match="classification.*regression",
    ):
        generate_target_statistics(
            df,
            "Target",
            problem_type="clustering",
        )