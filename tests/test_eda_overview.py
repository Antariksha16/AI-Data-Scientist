import json

import pandas as pd
import pytest

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda_overview import (
    generate_dataset_overview,
)


def _profiles_by_name(report: dict) -> dict:
    return {
        profile["column"]: profile
        for profile in report["column_profiles"]
    }


def test_dataset_shape_memory_and_target_role():
    df = pd.DataFrame(
        {
            "Feature": range(30),
            "Category": ["A", "B", "C"] * 10,
            "Target": [0, 1] * 15,
        }
    )

    report = generate_dataset_overview(df, "Target")
    profiles = _profiles_by_name(report)

    assert report["shape"]["rows"] == 30
    assert report["shape"]["columns"] == 3
    assert report["shape"]["candidate_features"] == 2
    assert report["memory"]["memory_usage_bytes"] > 0
    assert profiles["Target"]["role"] == "target"
    assert profiles["Feature"]["role"] == "feature"
    assert report["basic_modeling_readiness"]["ready"] is True


def test_missingness_patterns_and_high_missing_columns():
    df = pd.DataFrame(
        {
            "MostlyMissing": [None] * 15
            + ["A", "B", "C", "D", "E"],
            "PartlyMissing": [1, None] * 10,
            "Target": [0, 1] * 10,
        }
    )

    report = generate_dataset_overview(df, "Target")
    missingness = report["missingness"]

    assert missingness["total_missing_cells"] == 25
    assert "MostlyMissing" in missingness[
        "high_missing_columns"
    ]
    assert missingness["high_missing_rows"] == 7
    assert len(missingness["top_missingness_patterns"]) > 0


def test_duplicate_rows_and_identical_columns():
    df = pd.DataFrame(
        {
            "FeatureA": [1, 2, 1, 2, 1, 2] * 4,
            "FeatureCopy": [1, 2, 1, 2, 1, 2] * 4,
            "Target": [0, 1, 0, 1, 0, 1] * 4,
        }
    )

    report = generate_dataset_overview(df, "Target")
    duplicates = report["duplicates"]

    assert duplicates["duplicate_rows"] == 22
    assert any(
        {
            pair["column_1"],
            pair["column_2"],
        }
        == {"FeatureA", "FeatureCopy"}
        for pair in duplicates["identical_column_pairs"]
    )


def test_semantic_type_inference():
    df = pd.DataFrame(
        {
            "Continuous": [
                10.5 + index * 0.37
                for index in range(60)
            ],
            "Discrete": [index % 5 for index in range(60)],
            "Category": ["A", "B", "C"] * 20,
            "Active": [True, False] * 30,
            "EventDate": [
                f"2025-01-{(index % 28) + 1:02d}"
                for index in range(60)
            ],
            "LongText": [
                "This is a detailed free-text description number "
                f"{index} with additional words."
                for index in range(60)
            ],
            "Target": [0, 1] * 30,
        }
    )

    report = generate_dataset_overview(
        df,
        "Target",
        EDAConfig(
            high_cardinality_count_threshold=20,
            high_cardinality_ratio_threshold=0.50,
        ),
    )
    profiles = _profiles_by_name(report)

    assert profiles["Continuous"]["semantic_type"] == (
        "continuous_numerical"
    )
    assert profiles["Discrete"]["semantic_type"] == (
        "discrete_numerical"
    )
    assert profiles["Category"]["semantic_type"] == "categorical"
    assert profiles["Active"]["semantic_type"] == "boolean"
    assert profiles["EventDate"]["semantic_type"] == "datetime"
    assert profiles["LongText"]["semantic_type"] == "free_text"


def test_identifier_and_constant_quality_flags():
    df = pd.DataFrame(
        {
            "EmployeeID": range(1, 31),
            "ConstantColumn": ["same"] * 30,
            "Feature": [value % 7 for value in range(30)],
            "Target": [0, 1] * 15,
        }
    )

    report = generate_dataset_overview(df, "Target")

    assert "EmployeeID" in report["quality_flags"][
        "identifier_candidates"
    ]
    assert "EmployeeID" not in report["quality_flags"][
        "high_cardinality_columns"
    ]
    assert "ConstantColumn" in report["quality_flags"][
        "constant_columns"
    ]


def test_duplicate_column_names_are_reported_as_blocker():
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

    report = generate_dataset_overview(df, "Target")

    assert report["schema"][
        "column_names_are_unique"
    ] is False
    assert report["schema"]["duplicate_column_names"] == [
        "Feature"
    ]
    assert report["basic_modeling_readiness"]["ready"] is False
    assert "duplicate column names" in report[
        "basic_modeling_readiness"
    ]["blockers"]


def test_overview_is_json_safe_and_non_mutating():
    df = pd.DataFrame(
        {
            "Feature": [1.0, 2.0, None, 4.0] * 5,
            "Category": ["A", "B", "A", None] * 5,
            "Target": [0, 1, 0, 1] * 5,
        }
    )
    original_df = df.copy(deep=True)

    report = generate_dataset_overview(df, "Target")

    json.dumps(report, allow_nan=False)
    pd.testing.assert_frame_equal(df, original_df)
    assert report["automatic_data_modification_performed"] is False


def test_overview_input_validation():
    df = pd.DataFrame(
        {
            "Feature": range(20),
            "Target": [0, 1] * 10,
        }
    )

    with pytest.raises(ValueError, match="does not exist"):
        generate_dataset_overview(
            df,
            "UnknownTarget",
        )

    with pytest.raises(ValueError, match="Dataset is empty"):
        generate_dataset_overview(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )