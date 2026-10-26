import math
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SUPPORTED_IMAGE_FORMATS = {"png", "svg"}
PERMUTATION_PLOT_ID = "permutation_importance_chart"
NATIVE_PLOT_ID = "native_importance_chart"

_PLOT_LOCK = threading.RLock()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _display_label(value: Any, maximum_length: int = 34) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()

    if len(text) > maximum_length:
        return f"{text[: maximum_length - 1]}…"

    return text


def _plot_horizontal_importance(
    features: list[dict[str, Any]],
    *,
    value_key: str,
    error_key: str | None,
    title: str,
    x_label: str,
    output_path: Path,
    image_format: str,
    dpi: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Visualization already exists: {output_path.name}"
        )

    plot_rows = []

    for feature in features:
        if not isinstance(feature, dict):
            continue

        value = _safe_float(feature.get(value_key))

        if value is None:
            continue

        error = (
            _safe_float(feature.get(error_key))
            if error_key is not None
            else None
        )
        plot_rows.append(
            {
                "feature": _display_label(
                    feature.get("feature", "Unknown")
                ),
                "value": value,
                "error": max(error or 0.0, 0.0),
            }
        )

    if not plot_rows:
        raise ValueError("No finite feature importance values exist.")

    plot_rows.reverse()
    labels = [row["feature"] for row in plot_rows]
    values = [row["value"] for row in plot_rows]
    errors = [row["error"] for row in plot_rows]
    colors = [
        "#2563EB" if value >= 0 else "#F97316"
        for value in values
    ]
    figure_height = max(4.0, 1.8 + 0.48 * len(plot_rows))
    temporary_path = output_path.with_name(
        f".{output_path.stem}-{uuid4().hex}.{image_format}"
    )

    try:
        with _PLOT_LOCK:
            figure, axis = plt.subplots(
                figsize=(10, figure_height),
                constrained_layout=True,
            )
            positions = list(range(len(plot_rows)))
            axis.barh(
                positions,
                values,
                xerr=errors if error_key is not None else None,
                color=colors,
                alpha=0.9,
                capsize=3,
            )
            axis.set_yticks(positions, labels=labels)
            axis.axvline(0, color="#334155", linewidth=0.9)
            axis.set_title(title, fontsize=14, fontweight="bold")
            axis.set_xlabel(x_label)
            axis.grid(
                axis="x",
                linestyle="--",
                alpha=0.3,
            )
            figure.savefig(
                temporary_path,
                format=image_format,
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _plot_filename(plot_id: str, image_format: str) -> str:
    return f"{plot_id}.{image_format}"


def generate_explainability_visualizations(
    explanation: dict[str, Any],
    output_directory: str | Path,
    *,
    image_format: str = "png",
    dpi: int = 120,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate aggregate feature-importance charts without raw rows."""

    if not isinstance(explanation, dict):
        raise TypeError("explanation must be a dictionary.")

    if explanation.get("explainability_status") != "complete":
        raise ValueError("A complete explanation report is required.")

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError("image_format must be 'png' or 'svg'.")

    if not 72 <= dpi <= 300:
        raise ValueError("dpi must be between 72 and 300.")

    output_path = Path(output_directory).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    source_explanation = deepcopy(explanation)
    generated_plots = []
    skipped_plots = []
    plot_errors = []

    permutation = explanation.get("permutation_importance", {})
    permutation_features = permutation.get("features", [])
    permutation_filename = _plot_filename(
        PERMUTATION_PLOT_ID,
        image_format,
    )

    if permutation_features:
        try:
            _plot_horizontal_importance(
                permutation_features,
                value_key="importance_mean",
                error_key="importance_standard_deviation",
                title="Permutation Feature Importance",
                x_label=permutation.get(
                    "importance_unit",
                    "Performance decrease after shuffling",
                ),
                output_path=output_path / permutation_filename,
                image_format=image_format,
                dpi=dpi,
                overwrite=overwrite,
            )
            generated_plots.append(
                {
                    "plot_id": PERMUTATION_PLOT_ID,
                    "file_name": permutation_filename,
                }
            )
        except Exception as error:
            plot_errors.append(
                {
                    "plot_id": PERMUTATION_PLOT_ID,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    else:
        skipped_plots.append(
            {
                "plot_id": PERMUTATION_PLOT_ID,
                "reason": "No permutation importance values exist.",
            }
        )

    native = explanation.get("native_importance", {})
    native_features = native.get("features", [])
    native_filename = _plot_filename(
        NATIVE_PLOT_ID,
        image_format,
    )

    if native.get("status") == "complete" and native_features:
        try:
            _plot_horizontal_importance(
                native_features,
                value_key="importance",
                error_key=None,
                title="Native Model Feature Importance",
                x_label=native.get("method") or "Native importance",
                output_path=output_path / native_filename,
                image_format=image_format,
                dpi=dpi,
                overwrite=overwrite,
            )
            generated_plots.append(
                {
                    "plot_id": NATIVE_PLOT_ID,
                    "file_name": native_filename,
                }
            )
        except Exception as error:
            plot_errors.append(
                {
                    "plot_id": NATIVE_PLOT_ID,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    else:
        skipped_plots.append(
            {
                "plot_id": NATIVE_PLOT_ID,
                "reason": native.get("reason")
                or "Native importance is unavailable.",
            }
        )

    if plot_errors and generated_plots:
        status = "partial"
    elif plot_errors:
        status = "failed"
    elif generated_plots:
        status = "complete"
    else:
        status = "skipped"

    return {
        "visualization_status": status,
        "image_format": image_format,
        "dpi": dpi,
        "generated_plot_count": len(generated_plots),
        "skipped_plot_count": len(skipped_plots),
        "plot_error_count": len(plot_errors),
        "generated_plots": generated_plots,
        "skipped_plots": skipped_plots,
        "plot_errors": plot_errors,
        "raw_holdout_rows_used": False,
        "source_explanation_modified": (
            explanation != source_explanation
        ),
    }
