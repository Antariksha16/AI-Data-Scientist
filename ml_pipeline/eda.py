import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_pipeline.eda_config import EDAConfig
from ml_pipeline.cleaner import create_cleaning_plan
from ml_pipeline.eda_overview import (
    generate_dataset_overview,
)
from ml_pipeline.eda_insights import (
    generate_eda_insights,
)
from ml_pipeline.eda_relationships import (
    generate_feature_feature_relationships,
    generate_feature_target_relationships,
)
from ml_pipeline.eda_statistics import (
    generate_categorical_statistics,
    generate_datetime_statistics,
    generate_numerical_statistics,
)
from ml_pipeline.eda_target import (
    generate_target_statistics,
)
from ml_pipeline.eda_visualizations import (
    generate_eda_visualizations,
)
from ml_pipeline.leakage import (
    detect_potential_leakage,
)
from ml_pipeline.outliers import analyze_outliers


EDA_REPORT_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    """Recursively convert report values into strict JSON values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def _unique_text_items(items: list[Any]) -> list[str]:
    """Deduplicate human-readable messages while preserving order."""

    unique_items = []
    seen = set()

    for item in items:
        text = str(item)
        if text not in seen:
            seen.add(text)
            unique_items.append(text)

    return unique_items


def generate_eda_report(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig | None = None,
    problem_type: str | None = None,
    include_feature_relationships: bool = True,
    include_preprocessing_context: bool = True,
    fail_fast: bool = False,
    include_insights: bool = True,
    include_visualizations: bool = False,
    visualization_output_directory: str | Path | None = None,
    visualization_image_format: str = "png",
    visualization_dpi: int = 120,
) -> dict:
    """
    Generate one generalized, JSON-safe tabular EDA report.

    By default, a noncritical section failure produces a partial
    report with explicit section errors. Set ``fail_fast=True``
    during development when the original exception should be raised.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if (
        include_visualizations
        and visualization_output_directory is None
    ):
        raise ValueError(
            "visualization_output_directory is required when "
            "include_visualizations=True."
        )

    config = config or EDAConfig()

    sections: dict[str, Any] = {}
    section_status: dict[str, str] = {}
    section_errors = []

    def run_section(
        section_name: str,
        operation: Callable[[], Any],
    ) -> Any:
        try:
            result = operation()
        except Exception as error:
            if fail_fast:
                raise

            sections[section_name] = None
            section_status[section_name] = "error"
            section_errors.append(
                {
                    "section": section_name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            return None

        sections[section_name] = result
        section_status[section_name] = "success"
        return result

    dataset_overview = run_section(
        "dataset_overview",
        lambda: generate_dataset_overview(
            df,
            target_column,
            config,
        ),
    )

    target_statistics = run_section(
        "target_statistics",
        lambda: generate_target_statistics(
            df,
            target_column,
            config=config,
            problem_type=problem_type,
        ),
    )

    numerical_statistics = run_section(
        "numerical_statistics",
        lambda: generate_numerical_statistics(
            df,
            target_column,
            config,
        ),
    )

    categorical_statistics = run_section(
        "categorical_statistics",
        lambda: generate_categorical_statistics(
            df,
            target_column,
            config,
        ),
    )

    datetime_statistics = run_section(
        "datetime_statistics",
        lambda: generate_datetime_statistics(
            df,
            target_column,
            config,
        ),
    )

    if include_feature_relationships:
        feature_target_relationships = run_section(
            "feature_target_relationships",
            lambda: generate_feature_target_relationships(
                df,
                target_column,
                config=config,
                problem_type=problem_type,
            ),
        )

        feature_feature_relationships = run_section(
            "feature_feature_relationships",
            lambda: generate_feature_feature_relationships(
                df,
                target_column,
                config,
            ),
        )
    else:
        feature_target_relationships = None
        feature_feature_relationships = None
        sections["feature_target_relationships"] = None
        sections["feature_feature_relationships"] = None
        section_status[
            "feature_target_relationships"
        ] = "skipped_by_configuration"
        section_status[
            "feature_feature_relationships"
        ] = "skipped_by_configuration"

    if include_preprocessing_context:
        cleaning_plan = run_section(
            "cleaning_plan",
            lambda: create_cleaning_plan(
                df,
                target_column,
            ),
        )

        leakage_report = run_section(
            "leakage_report",
            lambda: detect_potential_leakage(
                df,
                target_column,
            ),
        )

        if cleaning_plan is not None:
            outlier_report = run_section(
                "outlier_report",
                lambda: analyze_outliers(
                    df,
                    cleaning_plan,
                ),
            )
        else:
            outlier_report = None
            sections["outlier_report"] = None
            section_status[
                "outlier_report"
            ] = "skipped_dependency_error"
    else:
        cleaning_plan = None
        leakage_report = None
        outlier_report = None

        for section_name in [
            "cleaning_plan",
            "leakage_report",
            "outlier_report",
        ]:
            sections[section_name] = None
            section_status[
                section_name
            ] = "skipped_by_configuration"

    successful_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "success"
    ]
    failed_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "error"
    ]
    skipped_sections = [
        section_name
        for section_name, status in section_status.items()
        if status.startswith("skipped")
    ]

    if failed_sections:
        report_status = (
            "partial"
            if successful_sections
            else "failed"
        )
    else:
        report_status = "complete"

    warnings = []
    recommendations = []
    limitations = [
        (
            "EDA findings describe statistical patterns and do not "
            "establish causation."
        ),
        (
            "Semantic, temporal, privacy, and domain leakage still "
            "require human review."
        ),
    ]

    if dataset_overview is not None:
        warnings.extend(
            dataset_overview.get("warnings", [])
        )

    if target_statistics is not None:
        warnings.extend(
            target_statistics.get("warnings", [])
        )
        recommendations.extend(
            target_statistics.get("recommendations", [])
        )

    if feature_target_relationships is not None:
        limitations.extend(
            feature_target_relationships.get(
                "limitations",
                [],
            )
        )

    if feature_feature_relationships is not None:
        limitations.extend(
            feature_feature_relationships.get(
                "limitations",
                [],
            )
        )

    if leakage_report is not None:
        limitations.extend(
            leakage_report.get("limitations", [])
        )

    summary = {
        "rows": (
            dataset_overview["shape"]["rows"]
            if dataset_overview is not None
            else len(df)
        ),
        "columns": (
            dataset_overview["shape"]["columns"]
            if dataset_overview is not None
            else len(df.columns)
        ),
        "candidate_features": len(df.columns) - 1,
        "problem_type": (
            target_statistics.get("problem_type")
            if target_statistics is not None
            else problem_type
        ),
        "basic_modeling_ready": (
            dataset_overview[
                "basic_modeling_readiness"
            ]["ready"]
            if dataset_overview is not None
            else None
        ),
        "numerical_feature_count": (
            numerical_statistics.get(
                "numerical_feature_count"
            )
            if numerical_statistics is not None
            else None
        ),
        "categorical_feature_count": (
            categorical_statistics.get(
                "categorical_feature_count"
            )
            if categorical_statistics is not None
            else None
        ),
        "datetime_feature_count": (
            datetime_statistics.get(
                "datetime_feature_count"
            )
            if datetime_statistics is not None
            else None
        ),
        "total_missing_cells": (
            dataset_overview["missingness"][
                "total_missing_cells"
            ]
            if dataset_overview is not None
            else None
        ),
        "duplicate_rows": (
            dataset_overview["duplicates"][
                "duplicate_rows"
            ]
            if dataset_overview is not None
            else None
        ),
        "analyzed_feature_target_relationships": (
            feature_target_relationships.get(
                "analyzed_feature_count"
            )
            if feature_target_relationships is not None
            else None
        ),
        "high_feature_feature_associations": (
            feature_feature_relationships.get(
                "high_association_pair_count"
            )
            if feature_feature_relationships is not None
            else None
        ),
        "potential_leakage_columns": (
            leakage_report.get(
                "potential_leakage_columns",
                [],
            )
            if leakage_report is not None
            else None
        ),
        "planned_dropped_columns": (
            cleaning_plan.get("drop_columns", [])
            if cleaning_plan is not None
            else None
        ),
    }

    report = {
        "report_type": "automated_tabular_eda",
        "report_version": EDA_REPORT_VERSION,
        "report_status": report_status,
        "target_column": target_column,
        "configuration": config.to_dict(),
        "execution_options": {
            "problem_type_override": problem_type,
            "include_feature_relationships": (
                include_feature_relationships
            ),
            "include_preprocessing_context": (
                include_preprocessing_context
            ),
            "fail_fast": fail_fast,
            "include_insights": include_insights,
            "include_visualizations": include_visualizations,
            "visualization_output_directory": (
                str(visualization_output_directory)
                if visualization_output_directory is not None
                else None
            ),
            "visualization_image_format": (
                visualization_image_format
            ),
            "visualization_dpi": visualization_dpi,
        },
        "section_status": section_status,
        "successful_sections": successful_sections,
        "failed_sections": failed_sections,
        "skipped_sections": skipped_sections,
        "section_errors": section_errors,
        "summary": summary,
        "dataset_overview": dataset_overview,
        "target_statistics": target_statistics,
        "numerical_statistics": numerical_statistics,
        "categorical_statistics": categorical_statistics,
        "datetime_statistics": datetime_statistics,
        "feature_target_relationships": (
            feature_target_relationships
        ),
        "feature_feature_relationships": (
            feature_feature_relationships
        ),
        "preprocessing_context": {
            "cleaning_plan": cleaning_plan,
            "leakage_report": leakage_report,
            "outlier_report": outlier_report,
        },
        "warnings": _unique_text_items(warnings),
        "recommendations": _unique_text_items(
            recommendations
        ),
        "limitations": _unique_text_items(limitations),
        "automatic_data_modification_performed": False,
    }

    if include_insights:
        try:
            insight_report = generate_eda_insights(report)
        except Exception as error:
            if fail_fast:
                raise

            insight_report = None
            sections["insights"] = None
            section_status["insights"] = "error"
            section_errors.append(
                {
                    "section": "insights",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        else:
            sections["insights"] = insight_report
            section_status["insights"] = "success"
    else:
        insight_report = None
        sections["insights"] = None
        section_status[
            "insights"
        ] = "skipped_by_configuration"

    successful_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "success"
    ]
    failed_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "error"
    ]
    skipped_sections = [
        section_name
        for section_name, status in section_status.items()
        if status.startswith("skipped")
    ]

    if failed_sections:
        report_status = (
            "partial"
            if successful_sections
            else "failed"
        )
    else:
        report_status = "complete"

    summary.update(
        {
            "generated_insights": (
                insight_report.get(
                    "generated_insight_count"
                )
                if insight_report is not None
                else None
            ),
            "critical_insights": (
                insight_report.get(
                    "severity_counts",
                    {},
                ).get("critical")
                if insight_report is not None
                else None
            ),
            "high_insights": (
                insight_report.get(
                    "severity_counts",
                    {},
                ).get("high")
                if insight_report is not None
                else None
            ),
        }
    )

    report.update(
        {
            "report_status": report_status,
            "section_status": section_status,
            "successful_sections": successful_sections,
            "failed_sections": failed_sections,
            "skipped_sections": skipped_sections,
            "section_errors": section_errors,
            "summary": summary,
            "insights": insight_report,
        }
    )

    if include_visualizations:
        try:
            visualization_report = generate_eda_visualizations(
                df,
                target_column,
                visualization_output_directory,
                eda_report=report,
                config=config,
                problem_type=problem_type,
                image_format=visualization_image_format,
                dpi=visualization_dpi,
                fail_fast=fail_fast,
            )
        except Exception as error:
            if fail_fast:
                raise

            visualization_report = None
            sections["visualizations"] = None
            section_status["visualizations"] = "error"
            section_errors.append(
                {
                    "section": "visualizations",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        else:
            sections["visualizations"] = visualization_report
            visualization_status = visualization_report.get(
                "visualization_status"
            )

            if visualization_status == "complete":
                section_status["visualizations"] = "success"
            elif visualization_status == "partial":
                section_status["visualizations"] = "partial"
            else:
                section_status["visualizations"] = "error"
                section_errors.append(
                    {
                        "section": "visualizations",
                        "error_type": (
                            "VisualizationGenerationError"
                        ),
                        "message": (
                            "All requested visualization plots "
                            "failed. See the visualization manifest."
                        ),
                    }
                )
    else:
        visualization_report = None
        sections["visualizations"] = None
        section_status[
            "visualizations"
        ] = "skipped_by_configuration"

    successful_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "success"
    ]
    partial_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "partial"
    ]
    failed_sections = [
        section_name
        for section_name, status in section_status.items()
        if status == "error"
    ]
    skipped_sections = [
        section_name
        for section_name, status in section_status.items()
        if status.startswith("skipped")
    ]

    if failed_sections or partial_sections:
        report_status = (
            "partial"
            if successful_sections or partial_sections
            else "failed"
        )
    else:
        report_status = "complete"

    summary.update(
        {
            "generated_visualizations": (
                visualization_report.get(
                    "generated_plot_count"
                )
                if visualization_report is not None
                else None
            ),
            "skipped_visualizations": (
                visualization_report.get(
                    "skipped_plot_count"
                )
                if visualization_report is not None
                else None
            ),
            "visualization_errors": (
                visualization_report.get(
                    "plot_error_count"
                )
                if visualization_report is not None
                else None
            ),
        }
    )

    report.update(
        {
            "report_status": report_status,
            "section_status": section_status,
            "successful_sections": successful_sections,
            "partial_sections": partial_sections,
            "failed_sections": failed_sections,
            "skipped_sections": skipped_sections,
            "section_errors": section_errors,
            "summary": summary,
            "visualizations": visualization_report,
        }
    )

    return _json_safe(report)