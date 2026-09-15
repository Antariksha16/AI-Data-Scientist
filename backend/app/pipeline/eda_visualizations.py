import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.app.core.eda_config import EDAConfig
from backend.app.pipeline.eda_target import (
    generate_target_statistics,
)


VISUALIZATION_REPORT_VERSION = "1.0"
SUPPORTED_IMAGE_FORMATS = {"png", "svg"}

_PLOT_LOCK = threading.RLock()
_COLORS = [
    "#2563EB",
    "#F59E0B",
    "#10B981",
    "#EF4444",
    "#8B5CF6",
    "#06B6D4",
    "#EC4899",
    "#64748B",
]


class _PlotNotApplicable(Exception):
    """Indicate that a plot is valid but not applicable."""


def _json_safe(value: Any) -> Any:
    """Recursively convert values into strict JSON-compatible data."""

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


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _display_label(value: Any, maximum_length: int = 28) -> str:
    """Create a bounded label that is safe for plot rendering."""

    if value is None:
        text = "Missing"
    else:
        text = str(value)

    text = text.replace("\n", " ").replace("\r", " ").strip()

    if len(text) > maximum_length:
        return f"{text[: maximum_length - 1]}â€¦"

    return text


def _unique_display_labels(
    values: Any,
    maximum_length: int = 28,
) -> list[str]:
    """Create readable labels without merging distinct long values."""

    labels = []
    occurrences: dict[str, int] = {}

    for value in values:
        base = _display_label(value, maximum_length)
        occurrences[base] = occurrences.get(base, 0) + 1
        count = occurrences[base]
        labels.append(base if count == 1 else f"{base} [{count}]")

    return labels


def _unused_aggregate_label(index: Any, base: str) -> str:
    """Return an aggregate label that cannot overwrite a real category."""

    existing = {str(value) for value in index}
    candidate = base

    while candidate in existing:
        candidate = f"Aggregated {candidate}"

    return candidate


def _finite_numeric(series: pd.Series) -> pd.Series:
    """Convert values to numeric and remove NaN and infinity."""

    return pd.to_numeric(
        series,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()


def _sample_dataframe(
    df: pd.DataFrame,
    config: EDAConfig,
) -> tuple[pd.DataFrame, bool]:
    """Return a deterministic row sample for raw-observation plots."""

    if len(df) <= config.max_plot_rows:
        return df.copy(deep=False), False

    sampled = df.sample(
        n=config.max_plot_rows,
        random_state=config.random_state,
        replace=False,
    ).sort_index()

    return sampled, True


def _resolve_problem_type(
    df: pd.DataFrame,
    target_column: str,
    config: EDAConfig,
    eda_report: dict,
    problem_type: str | None,
) -> str:
    """Resolve classification versus regression consistently."""

    if problem_type is not None:
        resolved = problem_type
    else:
        resolved = _as_dict(
            eda_report.get("summary")
        ).get("problem_type")

    if resolved is None:
        resolved = generate_target_statistics(
            df,
            target_column,
            config=config,
        ).get("problem_type")

    if resolved not in {"classification", "regression"}:
        raise ValueError(
            "problem_type must resolve to 'classification' or "
            "'regression'."
        )

    return resolved


def _feature_lists(
    df: pd.DataFrame,
    target_column: str,
    eda_report: dict,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Resolve ordered numerical, categorical, and datetime features."""

    numerical = _as_list(
        _as_dict(
            eda_report.get("numerical_statistics")
        ).get("numerical_columns")
    )
    categorical = _as_list(
        _as_dict(
            eda_report.get("categorical_statistics")
        ).get("categorical_columns")
    )
    datetime_columns = _as_list(
        _as_dict(
            eda_report.get("datetime_statistics")
        ).get("datetime_columns")
    )

    if not numerical:
        numerical = [
            column
            for column in df.select_dtypes(
                include=[np.number]
            ).columns
            if column != target_column
        ]

    if not datetime_columns:
        datetime_columns = [
            column
            for column in df.columns
            if column != target_column
            and pd.api.types.is_datetime64_any_dtype(
                df[column].dtype
            )
        ]

    if not categorical:
        categorical = [
            column
            for column in df.columns
            if column != target_column
            and column not in numerical
            and column not in datetime_columns
            and (
                pd.api.types.is_object_dtype(df[column].dtype)
                or pd.api.types.is_string_dtype(df[column].dtype)
                or pd.api.types.is_bool_dtype(df[column].dtype)
                or isinstance(
                    df[column].dtype,
                    pd.CategoricalDtype,
                )
            )
        ]

    def valid_unique(columns: list[Any]) -> list[Any]:
        output = []
        seen = set()

        for column in columns:
            if (
                column in df.columns
                and column != target_column
                and column not in seen
            ):
                output.append(column)
                seen.add(column)

        return output

    datetime_columns = valid_unique(datetime_columns)
    numerical = [
        column
        for column in valid_unique(numerical)
        if column not in datetime_columns
    ]
    categorical = [
        column
        for column in valid_unique(categorical)
        if column not in datetime_columns
        and column not in numerical
    ]

    return numerical, categorical, datetime_columns


def _relationship_features(
    df: pd.DataFrame,
    target_column: str,
    eda_report: dict,
    numerical: list[Any],
    categorical: list[Any],
    datetime_columns: list[Any],
    maximum_features: int,
) -> list[dict]:
    """Select ranked relationship features with safe fallbacks."""

    ranked = _as_list(
        _as_dict(
            eda_report.get("feature_target_relationships")
        ).get("ranked_features")
    )
    selected = []
    seen = set()

    for item in ranked:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature")
        if (
            feature not in df.columns
            or feature == target_column
            or feature in seen
        ):
            continue

        feature_type = item.get("feature_type")
        if feature in datetime_columns:
            feature_type = "datetime"
        elif feature_type not in {
            "numerical",
            "categorical",
            "datetime",
        }:
            feature_type = (
                "numerical"
                if feature in numerical
                else "categorical"
            )

        selected.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "primary_score": item.get("primary_score"),
            }
        )
        seen.add(feature)

        if len(selected) >= maximum_features:
            return selected

    for feature_type, features in [
        ("numerical", numerical),
        ("categorical", categorical),
        ("datetime", datetime_columns),
    ]:
        for feature in features:
            if feature in seen:
                continue

            selected.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "primary_score": None,
                }
            )
            seen.add(feature)

            if len(selected) >= maximum_features:
                return selected

    return selected


def _subplot_grid(
    feature_count: int,
    maximum_columns: int = 3,
    width: float = 5.0,
    height: float = 3.8,
) -> tuple[Any, list[Any]]:
    """Create a compact grid and return a flat axes list."""

    columns = min(maximum_columns, max(feature_count, 1))
    rows = math.ceil(max(feature_count, 1) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * width, rows * height),
        squeeze=False,
    )

    return figure, list(axes.ravel())


def _finalize_figure(figure: Any, title: str) -> None:
    """Apply consistent layout and title settings."""

    figure.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    figure.tight_layout()


def _save_figure_atomically(
    figure: Any,
    final_path: Path,
    image_format: str,
    dpi: int,
) -> None:
    """Write a complete image before replacing the destination file."""

    temporary_path = final_path.with_name(
        f".{final_path.stem}.{uuid.uuid4().hex}.tmp."
        f"{image_format}"
    )

    try:
        figure.savefig(
            temporary_path,
            format=image_format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def generate_eda_visualizations(
    df: pd.DataFrame,
    target_column: str,
    output_directory: str | Path,
    eda_report: dict | None = None,
    config: EDAConfig | None = None,
    problem_type: str | None = None,
    image_format: str = "png",
    dpi: int = 120,
    fail_fast: bool = False,
) -> dict:
    """
    Generate bounded, server-safe EDA plots and a JSON-safe manifest.

    Plot generation is isolated per chart. By default, one plot failure is
    recorded without discarding successfully generated images. The source
    DataFrame is never modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not isinstance(dpi, int) or isinstance(dpi, bool):
        raise TypeError("dpi must be an integer.")

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero.")

    normalized_format = str(image_format).lower().lstrip(".")
    if normalized_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            "image_format must be one of: "
            f"{sorted(SUPPORTED_IMAGE_FORMATS)}."
        )

    if not isinstance(fail_fast, bool):
        raise TypeError("fail_fast must be a boolean.")

    config = config or EDAConfig()
    eda_report = eda_report or {}

    report_target = eda_report.get("target_column")
    if (
        report_target is not None
        and report_target != target_column
    ):
        raise ValueError(
            "The EDA report target_column does not match the "
            "visualization target_column."
        )

    output_path = Path(output_directory).expanduser()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(
            "output_directory exists but is not a directory."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    resolved_problem_type = _resolve_problem_type(
        df,
        target_column,
        config,
        eda_report,
        problem_type,
    )

    numerical, categorical, datetime_columns = _feature_lists(
        df,
        target_column,
        eda_report,
    )
    numerical = numerical[: config.max_features_visualized]
    categorical = categorical[: config.max_features_visualized]
    datetime_columns = datetime_columns[
        : config.max_features_visualized
    ]

    plot_df, sampling_performed = _sample_dataframe(
        df,
        config,
    )

    generated_plots = []
    skipped_plots = []
    plot_errors = []
    overwritten_files = []

    mime_type = (
        "image/png"
        if normalized_format == "png"
        else "image/svg+xml"
    )

    def attempt_plot(
        plot_id: str,
        title: str,
        description: str,
        builder: Callable[[], tuple[Any, list[Any], int, list[str]]],
    ) -> None:
        figure = None
        final_path = output_path / (
            f"{plot_id}.{normalized_format}"
        )

        try:
            with _PLOT_LOCK:
                figure, features, rows_used, notes = builder()
                existed_before = final_path.exists()
                _save_figure_atomically(
                    figure,
                    final_path,
                    normalized_format,
                    dpi,
                )
        except _PlotNotApplicable as error:
            skipped_plots.append(
                {
                    "plot_id": plot_id,
                    "title": title,
                    "reason": str(error),
                }
            )
            return
        except Exception as error:
            if fail_fast:
                raise

            plot_errors.append(
                {
                    "plot_id": plot_id,
                    "title": title,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            return
        finally:
            if figure is not None:
                with _PLOT_LOCK:
                    plt.close(figure)

        if existed_before:
            overwritten_files.append(final_path.name)

        generated_plots.append(
            {
                "plot_id": plot_id,
                "title": title,
                "description": description,
                "status": "generated",
                "file_name": final_path.name,
                "relative_path": final_path.name,
                "file_path": str(final_path),
                "mime_type": mime_type,
                "image_format": normalized_format,
                "features": list(features),
                "rows_used": int(rows_used),
                "notes": list(notes),
                "overwrote_existing_file": existed_before,
            }
        )

    def target_distribution_plot():
        target = plot_df[target_column]

        if resolved_problem_type == "classification":
            observed = target.dropna().map(str)
            if observed.empty:
                raise _PlotNotApplicable(
                    "The target has no observed class values."
                )

            counts = observed.value_counts()
            maximum = config.max_categories_displayed
            displayed = counts.iloc[:maximum].copy()
            hidden_count = int(counts.iloc[maximum:].sum())
            if hidden_count > 0:
                aggregate_label = _unused_aggregate_label(
                    counts.index,
                    "Other classes",
                )
                displayed.loc[aggregate_label] = hidden_count

            figure, axis = plt.subplots(figsize=(9, 5.5))
            bars = axis.bar(
                range(len(displayed)),
                displayed.values,
                color=_COLORS[0],
                edgecolor="white",
            )
            axis.set_xticks(range(len(displayed)))
            axis.set_xticklabels(
                _unique_display_labels(displayed.index),
                rotation=35,
                ha="right",
            )
            axis.set_ylabel("Row count")
            axis.set_xlabel("Target class")
            axis.grid(axis="y", alpha=0.2)

            for bar, count in zip(bars, displayed.values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            notes = []
            if hidden_count > 0:
                notes.append(
                    "Less frequent classes were combined into "
                    "'Other classes' for readability."
                )
        else:
            values = _finite_numeric(target)
            if len(values) < 2:
                raise _PlotNotApplicable(
                    "The regression target has fewer than two "
                    "finite values."
                )

            figure, axes = plt.subplots(
                1,
                2,
                figsize=(12, 4.8),
                gridspec_kw={"width_ratios": [3, 1]},
            )
            axes[0].hist(
                values,
                bins=min(30, max(5, int(math.sqrt(len(values))))),
                color=_COLORS[0],
                edgecolor="white",
                alpha=0.9,
            )
            axes[0].axvline(
                values.median(),
                color=_COLORS[3],
                linestyle="--",
                label="Median",
            )
            axes[0].set_xlabel(_display_label(target_column))
            axes[0].set_ylabel("Frequency")
            axes[0].legend()
            axes[0].grid(axis="y", alpha=0.2)
            axes[1].boxplot(
                values.to_numpy(),
                orientation="vertical",
                patch_artist=True,
                boxprops={"facecolor": _COLORS[1]},
                medianprops={"color": _COLORS[3]},
            )
            axes[1].set_xticks([1])
            axes[1].set_xticklabels([_display_label(target_column)])
            axes[1].grid(axis="y", alpha=0.2)
            notes = []

        _finalize_figure(
            figure,
            f"Target Distribution: {_display_label(target_column)}",
        )
        return figure, [target_column], len(plot_df), notes

    attempt_plot(
        "target_distribution",
        "Target distribution",
        (
            "Class counts for classification or a histogram and "
            "boxplot for regression."
        ),
        target_distribution_plot,
    )

    def missingness_plot():
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values()

        if missing_counts.empty:
            raise _PlotNotApplicable(
                "No missing values were detected."
            )

        maximum = config.max_features_visualized
        missing_counts = missing_counts.iloc[-maximum:]
        percentages = missing_counts / len(df) * 100
        labels = [_display_label(item) for item in missing_counts.index]

        figure, axis = plt.subplots(
            figsize=(10, max(4.5, len(labels) * 0.45))
        )
        bars = axis.barh(
            range(len(labels)),
            percentages.values,
            color=_COLORS[3],
            alpha=0.85,
        )
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlabel("Missing values (%)")
        axis.set_xlim(0, max(100, percentages.max() * 1.15))
        axis.grid(axis="x", alpha=0.2)

        for bar, percentage in zip(bars, percentages.values):
            axis.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {percentage:.1f}%",
                va="center",
                fontsize=9,
            )

        _finalize_figure(figure, "Feature Missingness")
        notes = []
        if int((df.isna().sum() > 0).sum()) > maximum:
            notes.append(
                "Only the features with the highest missingness "
                "are displayed."
            )

        return (
            figure,
            list(missing_counts.index),
            len(df),
            notes,
        )

    attempt_plot(
        "missingness",
        "Missing-value percentages",
        "Percentage of unavailable values in affected columns.",
        missingness_plot,
    )

    def numerical_distribution_plot():
        features = numerical[: config.max_features_visualized]
        if not features:
            raise _PlotNotApplicable(
                "No numerical candidate features were detected."
            )

        figure, axes = _subplot_grid(len(features))
        plotted_features = []

        for axis, feature in zip(axes, features):
            values = _finite_numeric(plot_df[feature])
            if values.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No finite values",
                    ha="center",
                    va="center",
                )
                axis.set_axis_off()
                continue

            unique_count = int(values.nunique())
            bins = (
                min(max(unique_count, 1), 20)
                if unique_count <= 20
                else min(30, max(5, int(math.sqrt(len(values)))))
            )
            axis.hist(
                values,
                bins=bins,
                color=_COLORS[0],
                edgecolor="white",
                alpha=0.85,
            )
            axis.axvline(
                values.median(),
                color=_COLORS[3],
                linestyle="--",
                linewidth=1.2,
            )
            axis.set_title(_display_label(feature))
            axis.set_ylabel("Frequency")
            axis.grid(axis="y", alpha=0.2)
            plotted_features.append(feature)

        for axis in axes[len(features):]:
            axis.set_axis_off()

        if not plotted_features:
            plt.close(figure)
            raise _PlotNotApplicable(
                "Numerical features contain no finite values."
            )

        _finalize_figure(figure, "Numerical Feature Distributions")
        return figure, plotted_features, len(plot_df), [
            "Dashed lines show feature medians."
        ]

    attempt_plot(
        "numerical_distributions",
        "Numerical feature distributions",
        "Bounded histograms for numerical candidate features.",
        numerical_distribution_plot,
    )

    def categorical_distribution_plot():
        features = categorical[
            : min(config.max_features_visualized, 12)
        ]
        if not features:
            raise _PlotNotApplicable(
                "No categorical candidate features were detected."
            )

        figure, axes = _subplot_grid(
            len(features),
            maximum_columns=2,
            width=6.2,
            height=4.2,
        )
        plotted_features = []
        truncated = []

        for axis, feature in zip(axes, features):
            values = plot_df[feature].astype("object")
            values = values.where(values.notna(), "Missing")
            counts = values.map(str).value_counts()
            if counts.empty:
                axis.set_axis_off()
                continue

            maximum = config.max_categories_displayed
            displayed = counts.iloc[:maximum].sort_values()
            hidden = int(counts.iloc[maximum:].sum())
            if hidden > 0:
                aggregate_label = _unused_aggregate_label(
                    counts.index,
                    "Other categories",
                )
                displayed.loc[aggregate_label] = hidden
                displayed = displayed.sort_values()
                truncated.append(feature)

            axis.barh(
                range(len(displayed)),
                displayed.values,
                color=_COLORS[2],
                alpha=0.85,
            )
            axis.set_yticks(range(len(displayed)))
            axis.set_yticklabels(
                _unique_display_labels(displayed.index),
                fontsize=8,
            )
            axis.set_title(_display_label(feature))
            axis.set_xlabel("Row count")
            axis.grid(axis="x", alpha=0.2)
            plotted_features.append(feature)

        for axis in axes[len(features):]:
            axis.set_axis_off()

        if not plotted_features:
            plt.close(figure)
            raise _PlotNotApplicable(
                "Categorical features contain no displayable values."
            )

        _finalize_figure(
            figure,
            "Categorical Feature Distributions",
        )
        notes = []
        if truncated:
            notes.append(
                "Less frequent values were grouped into 'Other "
                "categories' for: "
                + ", ".join(_display_label(item) for item in truncated)
                + "."
            )

        return figure, plotted_features, len(plot_df), notes

    attempt_plot(
        "categorical_distributions",
        "Categorical feature distributions",
        "Top category counts with bounded cardinality.",
        categorical_distribution_plot,
    )

    def correlation_heatmap_plot():
        features = numerical[: config.max_pairwise_features]
        if len(features) < 2:
            raise _PlotNotApplicable(
                "At least two numerical candidate features are required."
            )

        numeric_frame = plot_df[features].apply(
            pd.to_numeric,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        usable = [
            column
            for column in numeric_frame.columns
            if numeric_frame[column].notna().sum() >= 2
            and numeric_frame[column].nunique(dropna=True) >= 2
        ]

        if len(usable) < 2:
            raise _PlotNotApplicable(
                "Fewer than two variable numerical features contain "
                "enough finite values."
            )

        correlation = numeric_frame[usable].corr(method="pearson")
        size = max(6.5, min(14.0, len(usable) * 0.65 + 3))
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(
            correlation.to_numpy(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )
        labels = [_display_label(item, 20) for item in usable]
        axis.set_xticks(range(len(usable)))
        axis.set_yticks(range(len(usable)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticklabels(labels)
        figure.colorbar(image, ax=axis, label="Pearson correlation")

        if len(usable) <= 10:
            for row in range(len(usable)):
                for column in range(len(usable)):
                    value = correlation.iloc[row, column]
                    if pd.notna(value):
                        axis.text(
                            column,
                            row,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=(
                                "white"
                                if abs(value) >= 0.6
                                else "black"
                            ),
                        )

        _finalize_figure(figure, "Numerical Correlation Heatmap")
        notes = []
        if len(numerical) > config.max_pairwise_features:
            notes.append(
                "The numerical feature set was limited by "
                "max_pairwise_features."
            )

        return figure, usable, len(plot_df), notes

    attempt_plot(
        "correlation_heatmap",
        "Numerical correlation heatmap",
        "Pearson correlation between eligible numerical features.",
        correlation_heatmap_plot,
    )

    def feature_target_relationship_plot():
        selected = _relationship_features(
            df,
            target_column,
            eda_report,
            numerical,
            categorical,
            datetime_columns,
            min(config.max_features_visualized, 6),
        )
        if not selected:
            raise _PlotNotApplicable(
                "No eligible candidate features were detected."
            )

        figure, axes = _subplot_grid(
            len(selected),
            maximum_columns=2,
            width=6.4,
            height=4.5,
        )
        plotted_features = []

        for axis, item in zip(axes, selected):
            feature = item["feature"]
            feature_type = item["feature_type"]
            subset = plot_df[[feature, target_column]].copy()
            title = _display_label(feature)
            score = item.get("primary_score")
            if score is not None:
                title += f" (score={score})"

            try:
                if resolved_problem_type == "regression":
                    target_values = pd.to_numeric(
                        subset[target_column],
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan)

                    if feature_type == "categorical":
                        categories = subset[feature].astype("object")
                        categories = categories.where(
                            categories.notna(),
                            "Missing",
                        ).map(str)
                        frame = pd.DataFrame(
                            {
                                "category": categories,
                                "target": target_values,
                            }
                        ).dropna(subset=["target"])
                        top = frame["category"].value_counts().head(
                            config.max_categories_displayed
                        ).index
                        means = (
                            frame[frame["category"].isin(top)]
                            .groupby("category")["target"]
                            .mean()
                            .sort_values()
                        )
                        if means.empty:
                            raise ValueError("No usable category groups")
                        axis.barh(
                            range(len(means)),
                            means.values,
                            color=_COLORS[2],
                        )
                        axis.set_yticks(range(len(means)))
                        axis.set_yticklabels(
                            _unique_display_labels(means.index),
                            fontsize=8,
                        )
                        axis.set_xlabel("Mean target")
                    elif feature_type == "datetime":
                        dates = pd.to_datetime(
                            subset[feature],
                            errors="coerce",
                            utc=True,
                        )
                        mask = dates.notna() & target_values.notna()
                        if int(mask.sum()) < 2:
                            raise ValueError("Insufficient datetime pairs")
                        axis.scatter(
                            dates[mask],
                            target_values[mask],
                            s=16,
                            alpha=0.5,
                            color=_COLORS[4],
                        )
                        axis.xaxis.set_major_formatter(
                            mdates.DateFormatter("%Y-%m")
                        )
                        axis.tick_params(axis="x", rotation=35)
                        axis.set_ylabel(_display_label(target_column))
                    else:
                        feature_values = pd.to_numeric(
                            subset[feature],
                            errors="coerce",
                        ).replace([np.inf, -np.inf], np.nan)
                        mask = (
                            feature_values.notna()
                            & target_values.notna()
                        )
                        if int(mask.sum()) < 2:
                            raise ValueError("Insufficient numerical pairs")
                        axis.scatter(
                            feature_values[mask],
                            target_values[mask],
                            s=16,
                            alpha=0.45,
                            color=_COLORS[0],
                        )
                        axis.set_xlabel(_display_label(feature))
                        axis.set_ylabel(_display_label(target_column))
                else:
                    target_labels = subset[target_column].astype("object")
                    target_labels = target_labels.where(
                        target_labels.notna(),
                        "Missing",
                    ).map(str)
                    top_classes = target_labels.value_counts().head(
                        min(config.max_categories_displayed, 10)
                    ).index
                    class_mask = target_labels.isin(top_classes)

                    if feature_type == "categorical":
                        feature_labels = subset[feature].astype("object")
                        feature_labels = feature_labels.where(
                            feature_labels.notna(),
                            "Missing",
                        ).map(str)
                        top_values = feature_labels.value_counts().head(
                            min(config.max_categories_displayed, 12)
                        ).index
                        frame = pd.DataFrame(
                            {
                                "feature": feature_labels,
                                "target": target_labels,
                            }
                        )
                        frame = frame[
                            frame["feature"].isin(top_values)
                            & frame["target"].isin(top_classes)
                        ]
                        table = pd.crosstab(
                            frame["feature"],
                            frame["target"],
                            normalize="index",
                        )
                        if table.empty:
                            raise ValueError("No usable category pairs")
                        bottom = np.zeros(len(table))
                        class_display_labels = dict(
                            zip(
                                table.columns,
                                _unique_display_labels(
                                    table.columns,
                                    maximum_length=20,
                                ),
                            )
                        )
                        for index, class_name in enumerate(table.columns):
                            values = table[class_name].to_numpy()
                            axis.bar(
                                range(len(table)),
                                values,
                                bottom=bottom,
                                label=class_display_labels[class_name],
                                color=_COLORS[index % len(_COLORS)],
                            )
                            bottom += values
                        axis.set_xticks(range(len(table)))
                        axis.set_xticklabels(
                            _unique_display_labels(table.index),
                            rotation=35,
                            ha="right",
                            fontsize=8,
                        )
                        axis.set_ylabel("Class proportion")
                        axis.legend(fontsize=7)
                    elif feature_type == "datetime":
                        dates = pd.to_datetime(
                            subset[feature],
                            errors="coerce",
                            utc=True,
                        )
                        codes = pd.Categorical(
                            target_labels,
                            categories=list(top_classes),
                        ).codes
                        mask = dates.notna() & class_mask & (codes >= 0)
                        if int(mask.sum()) < 2:
                            raise ValueError("Insufficient datetime pairs")
                        axis.scatter(
                            dates[mask],
                            codes[mask],
                            s=18,
                            alpha=0.5,
                            color=_COLORS[4],
                        )
                        axis.set_yticks(range(len(top_classes)))
                        axis.set_yticklabels(
                            _unique_display_labels(
                                top_classes,
                                maximum_length=20,
                            )
                        )
                        axis.xaxis.set_major_formatter(
                            mdates.DateFormatter("%Y-%m")
                        )
                        axis.tick_params(axis="x", rotation=35)
                    else:
                        feature_values = pd.to_numeric(
                            subset[feature],
                            errors="coerce",
                        ).replace([np.inf, -np.inf], np.nan)
                        groups = []
                        labels = []
                        for class_name in top_classes:
                            group = feature_values[
                                (target_labels == class_name)
                                & feature_values.notna()
                            ]
                            if not group.empty:
                                groups.append(group.to_numpy())
                                labels.append(str(class_name))
                        if not groups:
                            raise ValueError("No usable numerical groups")
                        axis.boxplot(
                            groups,
                            tick_labels=_unique_display_labels(labels),
                            patch_artist=True,
                            boxprops={"facecolor": _COLORS[1]},
                            medianprops={"color": _COLORS[3]},
                        )
                        axis.set_ylabel(_display_label(feature))
                        axis.tick_params(axis="x", rotation=35)

                axis.set_title(title)
                axis.grid(axis="y", alpha=0.2)
                plotted_features.append(feature)
            except (TypeError, ValueError):
                axis.text(
                    0.5,
                    0.5,
                    "Insufficient usable data",
                    ha="center",
                    va="center",
                )
                axis.set_title(title)
                axis.set_axis_off()

        for axis in axes[len(selected):]:
            axis.set_axis_off()

        if not plotted_features:
            plt.close(figure)
            raise _PlotNotApplicable(
                "Selected features contain insufficient paired data."
            )

        _finalize_figure(
            figure,
            "Leading Featureâ€“Target Relationships",
        )
        return figure, plotted_features, len(plot_df), [
            "Plots show association and do not establish causation."
        ]

    attempt_plot(
        "feature_target_relationships",
        "Leading feature-target relationships",
        (
            "Problem-type-aware views of the highest-ranked candidate "
            "features."
        ),
        feature_target_relationship_plot,
    )

    def datetime_overview_plot():
        features = datetime_columns[
            : min(config.max_features_visualized, 6)
        ]
        if not features:
            raise _PlotNotApplicable(
                "No datetime candidate features were detected."
            )

        figure, axes = _subplot_grid(
            len(features),
            maximum_columns=2,
            width=6.5,
            height=4.2,
        )
        plotted_features = []
        frequency_notes = []

        for axis, feature in zip(axes, features):
            parsed = pd.to_datetime(
                plot_df[feature],
                errors="coerce",
                utc=True,
            ).dropna()
            if len(parsed) < 2:
                axis.set_axis_off()
                continue

            naive = parsed.dt.tz_convert(None)
            span_days = (
                naive.max() - naive.min()
            ).total_seconds() / 86_400

            if span_days <= 90:
                frequency = "D"
                frequency_label = "daily"
            elif span_days <= 730:
                frequency = "W"
                frequency_label = "weekly"
            else:
                frequency = "ME"
                frequency_label = "monthly"

            counts = pd.Series(
                1,
                index=pd.DatetimeIndex(naive),
                dtype=float,
            ).resample(frequency).sum()
            axis.plot(
                counts.index,
                counts.values,
                color=_COLORS[4],
                linewidth=1.8,
            )
            axis.fill_between(
                counts.index,
                counts.values,
                color=_COLORS[4],
                alpha=0.15,
            )
            axis.set_title(_display_label(feature))
            axis.set_ylabel("Observed rows")
            axis.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y-%m")
            )
            axis.tick_params(axis="x", rotation=35)
            axis.grid(alpha=0.2)
            plotted_features.append(feature)
            frequency_notes.append(
                f"{_display_label(feature)} uses {frequency_label} bins."
            )

        for axis in axes[len(features):]:
            axis.set_axis_off()

        if not plotted_features:
            plt.close(figure)
            raise _PlotNotApplicable(
                "Datetime features contain fewer than two valid values."
            )

        _finalize_figure(figure, "Datetime Observation Patterns")
        return figure, plotted_features, len(plot_df), frequency_notes

    attempt_plot(
        "datetime_overview",
        "Datetime observation patterns",
        "Observed-row frequency across detected datetime features.",
        datetime_overview_plot,
    )

    if plot_errors:
        visualization_status = (
            "partial" if generated_plots else "failed"
        )
    else:
        visualization_status = "complete"

    manifest = {
        "visualization_report_type": "automated_eda_visualizations",
        "visualization_report_version": (
            VISUALIZATION_REPORT_VERSION
        ),
        "visualization_status": visualization_status,
        "target_column": target_column,
        "problem_type": resolved_problem_type,
        "output_directory": str(output_path),
        "image_format": normalized_format,
        "mime_type": mime_type,
        "dpi": dpi,
        "backend": str(matplotlib.get_backend()),
        "sampling": {
            "original_rows": len(df),
            "plotted_rows": len(plot_df),
            "sampling_performed": sampling_performed,
            "max_plot_rows": config.max_plot_rows,
            "random_state": config.random_state,
        },
        "limits": {
            "max_categories_displayed": (
                config.max_categories_displayed
            ),
            "max_features_visualized": (
                config.max_features_visualized
            ),
            "max_pairwise_features": (
                config.max_pairwise_features
            ),
        },
        "detected_features": {
            "numerical": numerical,
            "categorical": categorical,
            "datetime": datetime_columns,
        },
        "generated_plot_count": len(generated_plots),
        "skipped_plot_count": len(skipped_plots),
        "plot_error_count": len(plot_errors),
        "generated_plots": generated_plots,
        "skipped_plots": skipped_plots,
        "plot_errors": plot_errors,
        "overwritten_files": overwritten_files,
        "source_dataframe_modified": False,
        "automatic_data_modification_performed": False,
        "limitations": [
            (
                "Plots are exploratory summaries and do not establish "
                "causation or model usefulness."
            ),
            (
                "Raw-observation plots use deterministic sampling when the "
                "dataset exceeds max_plot_rows."
            ),
            (
                "Displayed categories and features are bounded for readability "
                "and performance."
            ),
            (
                "Generated files may contain sensitive labels or distributions "
                "and require the same access controls as the source dataset."
            ),
        ],
    }

    return _json_safe(manifest)
