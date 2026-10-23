import pandas as pd
import pytest

from ml_pipeline.cleaner import create_cleaning_plan
from ml_pipeline.outliers import analyze_outliers
from ml_pipeline.preprocessing import (
    prepare_dataset_for_modeling,
)


def test_cleaning_plan_flags_quality_issues():
    df = pd.DataFrame(
        {
            "RecordID": range(1, 61),
            "Region": ["North", "South", "West"] * 20,
            "Outcome": [
                "Approved",
                "Rejected",
                "Pending",
            ] * 20,
            "ContinuousScore": [
                1000.5 + i * 23.75
                for i in range(60)
            ],
            "ConstantColumn": ["same"] * 60,
            "MostlyMissing": (
                [None] * 55
                + ["A", "B", "C", "D", "E"]
            ),
            "Description": [
                f"item_{i}"
                for i in range(60)
            ],
        }
    )

    plan = create_cleaning_plan(
        df,
        "Outcome",
    )

    assert set(plan["drop_columns"]) == {
        "RecordID",
        "ConstantColumn",
        "MostlyMissing",
    }

    assert "Description" in plan["review_columns"]
    assert "Region" in plan["kept_features"]
    assert "Outcome" not in plan["kept_features"]


def test_classification_preprocessing():
    df = pd.DataFrame(
        {
            "RecordID": range(1, 9),
            "Age": [
                20,
                30,
                None,
                40,
                35,
                50,
                45,
                55,
            ],
            "City": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "B",
                "B",
                "A",
            ],
            "Target": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    result = prepare_dataset_for_modeling(
        df,
        "Target",
    )

    transformed = result[
        "preprocessor"
    ].fit_transform(result["X"])

    metadata = result["report"][
        "preprocessing_metadata"
    ]

    assert (
        result["report"]["target_analysis"][
            "problem_type"
        ]
        == "classification"
    )

    assert "RecordID" not in metadata["model_features"]
    assert "Target" not in metadata["model_features"]
    assert int(pd.isna(transformed).sum()) == 0

    new_data = result["X"].head(1).copy()
    new_data["Age"] = None
    new_data["City"] = "UnseenCity"

    transformed_new = result[
        "preprocessor"
    ].transform(new_data)

    assert (
        transformed_new.shape[1]
        == transformed.shape[1]
    )

    assert int(
        pd.isna(transformed_new).sum()
    ) == 0

def test_regression_preprocessing():
    df = pd.DataFrame(
        {
            "EmployeeID": range(1, 21),
            "Experience": [
                i % 5
                for i in range(20)
            ],
            "Department": [
                "IT",
                "HR",
                "Sales",
                "Finance",
            ] * 5,
            "Salary": [
                30000.5 + i * 1000.25
                for i in range(20)
            ],
        }
    )

    result = prepare_dataset_for_modeling(
        df,
        "Salary",
    )

    transformed = result[
        "preprocessor"
    ].fit_transform(result["X"])

    metadata = result["report"][
        "preprocessing_metadata"
    ]

    problem_type = result["report"][
        "target_analysis"
    ]["problem_type"]

    assert problem_type == "regression"
    assert "EmployeeID" not in metadata["model_features"]
    assert "Salary" not in metadata["model_features"]
    assert int(pd.isna(transformed).sum()) == 0


def test_missing_target_rows_are_removed():
    df = pd.DataFrame(
        {
            "Feature": [
                10,
                12,
                11,
                13,
                12,
                14,
            ],
            "Category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "Target": [
                0,
                1,
                None,
                0,
                1,
                None,
            ],
        }
    )

    result = prepare_dataset_for_modeling(
        df,
        "Target",
    )

    report = result["report"]

    assert report["original_rows"] == 6
    assert report["training_rows"] == 4
    assert report["target_missing_rows_removed"] == 2
    assert result["y"].isnull().sum() == 0


def test_potential_leakage_blocks_preparation():
    df = pd.DataFrame(
        {
            "Feature": [
                10,
                20,
                15,
                25,
                12,
                22,
            ],
            "TargetCopy": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "Target": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Potential target leakage",
    ):
        prepare_dataset_for_modeling(
            df,
            "Target",
        )


def test_outliers_recommend_robust_scaling():
    df = pd.DataFrame(
        {
            "Measurement": (
                list(range(1, 20))
                + [1000]
            ),
            "Target": [0, 1] * 10,
        }
    )

    plan = create_cleaning_plan(
        df,
        "Target",
    )

    report = analyze_outliers(
        df,
        plan,
    )

    assert (
        "Measurement"
        in report["robust_scaler_features"]
    )

    assert report["rows_removed"] == 0