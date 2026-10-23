import json

import pandas as pd
import pytest

from ml_pipeline.eda_config import EDAConfig
from ml_pipeline.eda_statistics import (
    generate_categorical_statistics,
    generate_datetime_statistics,
)


def _build_datetime_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NativeDate": pd.date_range(
                "2025-01-01",
                periods=12,
                freq="D",
            ),
            "CreatedAt": [
                f"2025-02-{day:02d} 10:30:00"
                for day in range(1, 13)
            ],
            "Region": ["North", "South"] * 6,
            "Description": [
                f"item_{index}"
                for index in range(12)
            ],
            "Target": [0, 1] * 6,
        }
    )


def test_datetime_statistics_detect_native_and_text_dates():
    df = _build_datetime_dataframe()

    report = generate_datetime_statistics(
        df,
        "Target",
    )

    assert report["datetime_feature_count"] == 2
    assert report["datetime_columns"] == [
        "NativeDate",
        "CreatedAt",
    ]
    assert report["native_datetime_columns"] == [
        "NativeDate"
    ]
    assert report["inferred_datetime_columns"] == [
        "CreatedAt"
    ]

    native_stats = report["datetime_statistics"][
        "NativeDate"
    ]

    assert native_stats["parse_success_percentage"] == 100
    assert native_stats["time_span_days"] == 11
    assert native_stats["regular_interval"] is True
    assert native_stats["is_monotonic_increasing"] is True


def test_datetime_columns_are_not_categorical():
    df = _build_datetime_dataframe()

    report = generate_categorical_statistics(
        df,
        "Target",
    )

    assert "CreatedAt" not in report[
        "categorical_columns"
    ]
    assert "Region" in report["categorical_columns"]
    assert "Description" in report[
        "categorical_columns"
    ]


def test_datetime_target_is_excluded():
    df = _build_datetime_dataframe().rename(
        columns={"NativeDate": "TargetDate"}
    )

    report = generate_datetime_statistics(
        df,
        "TargetDate",
    )

    assert "TargetDate" not in report[
        "datetime_columns"
    ]


def test_datetime_report_is_strictly_json_safe():
    df = _build_datetime_dataframe()

    report = generate_datetime_statistics(
        df,
        "Target",
    )

    json.dumps(report, allow_nan=False)


def test_datetime_inference_respects_threshold():
    df = pd.DataFrame(
        {
            "MostlyDates": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "not-a-date",
            ],
            "Target": [0, 1] * 5,
        }
    )

    default_report = generate_datetime_statistics(
        df,
        "Target",
    )

    strict_report = generate_datetime_statistics(
        df,
        "Target",
        EDAConfig(datetime_parse_threshold=1.0),
    )

    assert "MostlyDates" in default_report[
        "datetime_columns"
    ]
    assert "MostlyDates" not in strict_report[
        "datetime_columns"
    ]


def test_datetime_statistics_validate_input():
    df = _build_datetime_dataframe()

    with pytest.raises(
        ValueError,
        match="does not exist",
    ):
        generate_datetime_statistics(
            df,
            "UnknownTarget",
        )

    with pytest.raises(
        ValueError,
        match="Dataset is empty",
    ):
        generate_datetime_statistics(
            pd.DataFrame(columns=["Target"]),
            "Target",
        )