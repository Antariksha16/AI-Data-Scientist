"""Regenerate synthetic section fixtures from the project's real EDA modules.

Optional, from the project root in the Python venv:
    PYTHONPATH=. MPLBACKEND=Agg python frontend/tests/generate_eda_fixtures.py

Normal npm tests use the committed JSON; Python is not required for npm test.
These fixtures exercise real statistical/insight/chart sections, not HTTP or
the preprocessing orchestration (covered separately by backend/user tests).
No uploaded dataset or original rows are stored in the fixtures.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from backend.app.pipeline.eda_overview import generate_dataset_overview
from backend.app.pipeline.eda_target import generate_target_statistics
from backend.app.pipeline.eda_statistics import (
    generate_categorical_statistics,
    generate_datetime_statistics,
    generate_numerical_statistics,
)
from backend.app.pipeline.eda_relationships import (
    generate_feature_feature_relationships,
    generate_feature_target_relationships,
)
from backend.app.pipeline.eda_insights import generate_eda_insights
from backend.app.pipeline.eda_visualizations import generate_eda_visualizations


def main() -> None:
    destination = Path(__file__).resolve().parent / "fixtures"
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for problem_type in ("classification", "regression"):
        df = pd.DataFrame({
            "Measurement": rng.normal(30, 5, 48),
            "Region": ["North", "South", "West"] * 16,
            "ObservedAt": pd.date_range("2025-01-01", periods=48),
            "Outcome": ([0, 1] * 24 if problem_type == "classification"
                        else rng.normal(150, 25, 48)),
        })
        df.loc[0, "Measurement"] = None
        target = "Outcome"
        overview = generate_dataset_overview(df, target)
        report = {
            "report_status": "complete", "target_column": target,
            "automatic_data_modification_performed": False,
            "summary": {"rows": len(df), "candidate_features": 3,
                        "problem_type": problem_type,
                        "total_missing_cells": 1, "duplicate_rows": 0},
            "dataset_overview": overview,
            "target_statistics": generate_target_statistics(df, target, problem_type=problem_type),
            "numerical_statistics": generate_numerical_statistics(df, target),
            "categorical_statistics": generate_categorical_statistics(df, target),
            "datetime_statistics": generate_datetime_statistics(df, target),
            "feature_target_relationships": generate_feature_target_relationships(df, target, problem_type=problem_type),
            "feature_feature_relationships": generate_feature_feature_relationships(df, target),
            "preprocessing_context": {"cleaning_plan": None, "outlier_report": None, "leakage_report": None},
            "warnings": overview.get("warnings", []),
            "recommendations": overview.get("recommendations", []),
            "limitations": ["Synthetic module-contract fixture; preprocessing context is intentionally not generated."],
            "failed_sections": [], "partial_sections": [], "section_errors": [],
            "section_status": {name: "success" for name in (
                "dataset_overview", "target_statistics", "numerical_statistics",
                "categorical_statistics", "datetime_statistics", "feature_target_relationships",
                "feature_feature_relationships",
            )},
        }
        report["insights"] = generate_eda_insights(report)
        analysis_id = ("1" if problem_type == "classification" else "2") * 32
        with TemporaryDirectory(prefix="m10b-eda-fixture-") as output:
            manifest = generate_eda_visualizations(df, target, output, eda_report=report)
            manifest["output_directory"] = None
            for plot in manifest["generated_plots"]:
                plot.pop("file_path", None)
                plot["artifact_url"] = f"/analysis/eda/artifacts/{analysis_id}/{plot['file_name']}"
            report["visualizations"] = manifest
        payload = {"filename": "synthetic.csv", "selected_target": target,
                   "analysis_id": analysis_id, "report": report}
        path = destination / f"eda-{problem_type}.json"
        path.write_text(json.dumps(payload, allow_nan=False, indent=2), encoding="utf-8")
        print(problem_type, "rows:", len(df), "plots:", manifest["generated_plot_count"],
              "plot errors:", manifest["plot_error_count"])


if __name__ == "__main__":
    main()
