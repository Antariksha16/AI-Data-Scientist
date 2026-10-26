import json
from copy import deepcopy

import pytest

from ml_pipeline.explainability_visualizations import (
    NATIVE_PLOT_ID,
    PERMUTATION_PLOT_ID,
    generate_explainability_visualizations,
)


def _explanation(native_available=True):
    native = {
        "status": "complete",
        "method": "absolute_coefficients",
        "reason": None,
        "features": [
            {"feature": "Age", "importance": 0.8},
            {"feature": "Department_IT", "importance": 0.3},
        ],
    }

    if not native_available:
        native = {
            "status": "unavailable",
            "method": None,
            "reason": "Estimator has no native importance.",
            "features": [],
        }

    return {
        "explainability_status": "complete",
        "permutation_importance": {
            "importance_unit": "balanced_accuracy_decrease",
            "features": [
                {
                    "feature": "Age",
                    "importance_mean": 0.12,
                    "importance_standard_deviation": 0.02,
                },
                {
                    "feature": "Department",
                    "importance_mean": -0.01,
                    "importance_standard_deviation": 0.01,
                },
            ],
        },
        "native_importance": native,
    }


def test_png_visualizations_are_generated_and_json_safe(tmp_path):
    explanation = _explanation()
    original = deepcopy(explanation)
    report = generate_explainability_visualizations(
        explanation,
        tmp_path,
    )

    assert report["visualization_status"] == "complete"
    assert report["generated_plot_count"] == 2
    assert report["plot_error_count"] == 0
    assert report["raw_holdout_rows_used"] is False
    assert report["source_explanation_modified"] is False
    assert explanation == original

    for plot in report["generated_plots"]:
        file_path = tmp_path / plot["file_name"]
        assert file_path.is_file()
        assert file_path.stat().st_size > 0
        assert file_path.read_bytes().startswith(b"\x89PNG")

    json.dumps(report, allow_nan=False)


def test_unavailable_native_importance_is_safely_skipped(tmp_path):
    report = generate_explainability_visualizations(
        _explanation(native_available=False),
        tmp_path,
    )

    assert report["visualization_status"] == "complete"
    assert report["generated_plot_count"] == 1
    assert report["skipped_plot_count"] == 1
    assert report["generated_plots"][0]["plot_id"] == (
        PERMUTATION_PLOT_ID
    )
    assert report["skipped_plots"][0]["plot_id"] == NATIVE_PLOT_ID


def test_svg_output_is_supported(tmp_path):
    report = generate_explainability_visualizations(
        _explanation(),
        tmp_path,
        image_format="svg",
        dpi=144,
    )

    assert report["image_format"] == "svg"
    assert report["dpi"] == 144

    for plot in report["generated_plots"]:
        content = (tmp_path / plot["file_name"]).read_text(
            encoding="utf-8"
        )
        assert "<svg" in content


def test_existing_charts_require_explicit_overwrite(tmp_path):
    explanation = _explanation()
    generate_explainability_visualizations(explanation, tmp_path)
    report = generate_explainability_visualizations(
        explanation,
        tmp_path,
    )

    assert report["visualization_status"] == "failed"
    assert report["generated_plot_count"] == 0
    assert report["plot_error_count"] == 2

    overwritten = generate_explainability_visualizations(
        explanation,
        tmp_path,
        overwrite=True,
    )
    assert overwritten["visualization_status"] == "complete"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"image_format": "pdf"}, "image_format"),
        ({"dpi": 400}, "dpi"),
    ],
)
def test_invalid_visualization_options_are_rejected(
    tmp_path,
    options,
    message,
):
    with pytest.raises(ValueError, match=message):
        generate_explainability_visualizations(
            _explanation(),
            tmp_path,
            **options,
        )
