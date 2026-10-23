"""Generate bounded, human-readable Markdown and PDF model reports."""

from __future__ import annotations

import hashlib
import html
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPORT_VERSION = "1.0"
MARKDOWN_REPORT_FILENAME = "model_report.md"
PDF_REPORT_FILENAME = "model_report.pdf"
SUPPORTED_FORMATS = ("markdown", "pdf")
MAX_TABLE_ROWS = 30
MAX_LIST_ITEMS = 30
MAX_TEXT_LENGTH = 1_000

_NAVY = colors.HexColor("#12304A")
_BLUE = colors.HexColor("#1D6FA5")
_PALE_BLUE = colors.HexColor("#EAF4FA")
_PALE_GREEN = colors.HexColor("#EAF7F0")
_PALE_AMBER = colors.HexColor("#FFF4D8")
_INK = colors.HexColor("#243746")
_MUTED = colors.HexColor("#607482")
_RULE = colors.HexColor("#CBD9E2")
_WHITE = colors.white


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _plain_text(value: Any, *, limit: int = MAX_TEXT_LENGTH) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, str)):
        text = str(value)
    elif isinstance(value, (list, tuple, set)):
        text = ", ".join(_plain_text(item, limit=120) for item in value)
    else:
        text = str(value)

    text = " ".join(text.replace("\x00", "").split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _humanize(value: Any) -> str:
    return _plain_text(value).replace("_", " ").strip().capitalize()


def _markdown_text(value: Any) -> str:
    text = _plain_text(value)
    for token in ("\\", "|", "*", "_", "`", "[", "]", "<", ">"):
        text = text.replace(token, f"\\{token}")
    return text.replace("\n", " ")


def _paragraph_text(value: Any) -> str:
    return html.escape(_plain_text(value), quote=True)


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=".professional-report-",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(name)


def _validated_formats(formats: Sequence[str]) -> tuple[str, ...]:
    if isinstance(formats, (str, bytes)):
        raise TypeError("formats must be a sequence of report formats.")

    normalized = tuple(dict.fromkeys(str(item).lower() for item in formats))
    if not normalized:
        raise ValueError("At least one report format is required.")

    unsupported = [item for item in normalized if item not in SUPPORTED_FORMATS]
    if unsupported:
        raise ValueError(f"Unsupported report formats: {unsupported}")
    return normalized


def _validated_charts(
    chart_paths: dict[str, str | Path] | None,
    output_directory: Path,
) -> list[dict[str, Any]]:
    charts = []
    for chart_id, raw_path in sorted((chart_paths or {}).items()):
        if not isinstance(chart_id, str) or re.fullmatch(
            r"[a-z][a-z0-9_]*", chart_id
        ) is None:
            raise ValueError("Chart identifiers must be safe snake_case names.")

        path = Path(raw_path).resolve()
        if path.parent != output_directory:
            raise ValueError("Report charts must be inside output_directory.")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Report chart was not found: {path.name}")
        if path.suffix.lower() not in {".png", ".svg"}:
            raise ValueError("Report charts must use PNG or SVG format.")

        charts.append(
            {
                "chart_id": chart_id,
                "title": _humanize(chart_id),
                "file_name": path.name,
                "path": path,
                "embedded_in_pdf": path.suffix.lower() == ".png",
            }
        )
    return charts


def _metric_label(metric: Any) -> str:
    labels = {
        "balanced_accuracy": "Balanced accuracy",
        "accuracy": "Accuracy",
        "precision_weighted": "Weighted precision",
        "recall_weighted": "Weighted recall",
        "f1_weighted": "Weighted F1",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
    }
    return labels.get(str(metric), _humanize(metric))


def _markdown_table(headers: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    bounded_rows = list(rows)[:MAX_TABLE_ROWS]
    header = "| " + " | ".join(_markdown_text(item) for item in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_markdown_text(item) for item in row) + " |"
        for row in bounded_rows
    ]
    return "\n".join([header, rule, *body])


def _bullet_lines(items: Iterable[Any], *, empty: str) -> list[str]:
    bounded = list(items)[:MAX_LIST_ITEMS]
    if not bounded:
        return [f"- {empty}"]
    return [f"- {_markdown_text(item)}" for item in bounded]


def _dataset_summary(
    training: dict[str, Any],
    eda: dict[str, Any],
) -> dict[str, Any]:
    split = _as_dict(training.get("split"))
    overview = _as_dict(eda.get("dataset_overview"))
    shape = _as_dict(overview.get("shape"))
    missingness = _as_dict(overview.get("missingness"))
    duplicates = _as_dict(overview.get("duplicates"))
    return {
        "rows": shape.get("rows", split.get("original_rows")),
        "columns": shape.get("columns"),
        "usable_rows": split.get("usable_rows"),
        "training_rows": split.get("training_rows"),
        "holdout_rows": split.get("test_rows"),
        "missing_target_rows_removed": split.get(
            "missing_target_rows_removed"
        ),
        "missing_cells": missingness.get("total_missing_cells"),
        "missing_percentage": missingness.get("total_missing_percentage"),
        "duplicate_rows": duplicates.get("duplicate_rows"),
    }


def _preprocessing_summary(training: dict[str, Any]) -> dict[str, Any]:
    preprocessing = _as_dict(training.get("preprocessing_report"))
    metadata = _as_dict(preprocessing.get("preprocessing_metadata"))
    cleaning = _as_dict(preprocessing.get("cleaning_plan"))
    leakage = _as_dict(preprocessing.get("leakage_report"))
    outliers = _as_dict(preprocessing.get("outlier_report"))
    return {
        "model_features": _as_list(metadata.get("model_features")),
        "numerical_features": _as_list(metadata.get("numerical_features")),
        "categorical_features": _as_list(metadata.get("categorical_features")),
        "standard_scaler_features": _as_list(
            metadata.get("standard_scaler_features")
        ),
        "robust_scaler_features": _as_list(
            metadata.get(
                "robust_scaler_features",
                outliers.get("robust_scaler_features", []),
            )
        ),
        "excluded_review_features": _as_list(
            metadata.get("excluded_review_features")
        ),
        "drop_columns": _as_list(cleaning.get("drop_columns")),
        "imputation_plan": _as_dict(cleaning.get("imputation_plan")),
        "leakage_columns": _as_list(
            leakage.get("potential_leakage_columns")
        ),
        "rows_removed_for_outliers": outliers.get("rows_removed", 0),
    }


def _model_rows(training: dict[str, Any]) -> list[list[Any]]:
    rationale = _as_dict(training.get("model_selection_rationale"))
    ranking = _as_list(rationale.get("model_ranking"))
    if ranking:
        return [
            [
                item.get("rank"),
                item.get("display_name"),
                item.get("family"),
                item.get("selected_variant"),
                item.get("cross_validation_primary_score"),
                item.get("cross_validation_standard_deviation"),
            ]
            for item in ranking[:MAX_TABLE_ROWS]
            if isinstance(item, dict)
        ]

    metric = training.get("primary_metric")
    rows = []
    for comparison in _as_list(training.get("model_comparisons")):
        if not isinstance(comparison, dict) or comparison.get("is_baseline"):
            continue
        metric_report = _as_dict(_as_dict(comparison.get("metrics")).get(metric))
        rows.append(
            [
                len(rows) + 1,
                comparison.get("display_name"),
                "Not available",
                "untuned",
                metric_report.get("mean"),
                metric_report.get("standard_deviation"),
            ]
        )
    return rows


def _eda_insights(eda: dict[str, Any]) -> list[dict[str, Any]]:
    insight_report = _as_dict(eda.get("insights"))
    return [
        item
        for item in _as_list(insight_report.get("prioritized_insights"))
        if isinstance(item, dict)
    ][:MAX_LIST_ITEMS]


def _permutation_rows(training: dict[str, Any]) -> list[list[Any]]:
    explanation = _as_dict(training.get("explainability"))
    permutation = _as_dict(explanation.get("permutation_importance"))
    entries = _as_list(
        permutation.get(
            "feature_importances",
            permutation.get("ranked_features", permutation.get("features", [])),
        )
    )
    rows = []
    for item in entries[:MAX_TABLE_ROWS]:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("rank", len(rows) + 1),
                item.get("feature", item.get("feature_name")),
                item.get("importance_mean", item.get("mean_importance")),
                item.get("importance_standard_deviation", item.get("standard_deviation")),
            ]
        )
    return rows


def _markdown_report(
    training: dict[str, Any],
    eda: dict[str, Any],
    charts: list[dict[str, Any]],
    *,
    title: str,
    dataset_filename: str | None,
) -> str:
    best = _as_dict(training.get("best_model"))
    rationale = _as_dict(training.get("model_selection_rationale"))
    winner = _as_dict(rationale.get("winner"))
    runner = _as_dict(rationale.get("runner_up"))
    confidence = _as_dict(rationale.get("decision_confidence"))
    holdout = _as_dict(training.get("holdout_metrics"))
    feature_selection = _as_dict(training.get("feature_selection"))
    recommendation = _as_dict(training.get("model_recommendation"))
    preprocessing = _preprocessing_summary(training)
    dataset = _dataset_summary(training, eda)
    split = _as_dict(training.get("split"))
    cv = _as_dict(training.get("cross_validation"))
    tuning = _as_dict(training.get("hyperparameter_tuning"))
    explanation = _as_dict(training.get("explainability"))
    baseline = _as_dict(rationale.get("baseline_assessment"))

    winner_name = winner.get("display_name", best.get("display_name"))
    winner_score = winner.get(
        "cross_validation_primary_score",
        best.get("cross_validation_primary_score"),
    )
    metric = training.get("primary_metric")
    metric_label = _metric_label(metric)

    lines = [
        f"# {_markdown_text(title)}",
        "",
        "> A decision-focused, privacy-conscious record of automated EDA, preprocessing, model comparison, validation, and explainability.",
        "",
        "## Executive summary",
        "",
        _markdown_table(
            ["Item", "Result"],
            [
                ["Dataset", dataset_filename or "Not supplied"],
                ["Target", training.get("target_column")],
                ["Problem type", _humanize(training.get("problem_type"))],
                ["Selected model", winner_name],
                ["Selected variant", winner.get("selected_variant", best.get("selected_variant"))],
                [f"Cross-validation {metric_label}", winner_score],
                ["Decision confidence", confidence.get("level")],
                ["Training status", training.get("training_status")],
            ],
        ),
        "",
    ]

    summary = _as_list(rationale.get("summary"))
    lines.extend(_bullet_lines(summary, empty="No selection summary was available."))
    lines.extend(
        [
            "",
            "## Dataset and target overview",
            "",
            _markdown_table(
                ["Measure", "Value"],
                [
                    ["Original rows", dataset.get("rows")],
                    ["Original columns", dataset.get("columns")],
                    ["Usable rows", dataset.get("usable_rows")],
                    ["Training rows", dataset.get("training_rows")],
                    ["Untouched holdout rows", dataset.get("holdout_rows")],
                    ["Missing target rows removed", dataset.get("missing_target_rows_removed")],
                    ["Missing cells", dataset.get("missing_cells")],
                    ["Missing-cell percentage", dataset.get("missing_percentage")],
                    ["Duplicate rows", dataset.get("duplicate_rows")],
                ],
            ),
            "",
            f"The target `{_markdown_text(training.get('target_column'))}` was treated as **{_markdown_text(_humanize(training.get('problem_type')))}** ({_markdown_text(training.get('problem_type_source'))}). The holdout used for final evaluation contained {_markdown_text(split.get('test_rows'))} rows and did not influence recommendation, comparison, tuning, or winner selection.",
            "",
            "## Data quality and preprocessing",
            "",
            _markdown_table(
                ["Decision", "Applied to modeling"],
                [
                    ["Numerical features", preprocessing["numerical_features"]],
                    ["Categorical features", preprocessing["categorical_features"]],
                    ["Median imputation and standard scaling", preprocessing["standard_scaler_features"]],
                    ["Median imputation and robust scaling", preprocessing["robust_scaler_features"]],
                    ["Dropped columns", preprocessing["drop_columns"]],
                    ["Manual-review exclusions", preprocessing["excluded_review_features"]],
                    ["Potential leakage columns", preprocessing["leakage_columns"]],
                    ["Rows removed for outliers", preprocessing["rows_removed_for_outliers"]],
                ],
            ),
            "",
            "Missing numerical values are imputed with the median, categorical values with the most frequent value, and categorical features are one-hot encoded with unknown categories ignored. Outliers are retained; affected numerical features use robust scaling when recommended.",
            "",
            "## Exploratory findings",
            "",
        ]
    )

    insights = _eda_insights(eda)
    if insights:
        for insight in insights:
            severity = _humanize(insight.get("severity"))
            finding = insight.get("title", insight.get("finding", insight.get("message")))
            recommendation_text = insight.get("recommendation")
            lines.append(f"- **{_markdown_text(severity)}:** {_markdown_text(finding)}")
            if recommendation_text:
                lines.append(f"  - Action: {_markdown_text(recommendation_text)}")
    else:
        lines.append("- No standalone EDA insight report was supplied for this export.")

    model_rows = _model_rows(training)
    lines.extend(
        [
            "",
            "## Candidate strategy and model comparison",
            "",
            f"Selection mode: **{_markdown_text(_humanize(recommendation.get('selection_mode')))}**. The recommender considered {_markdown_text(recommendation.get('registered_non_baseline_model_count'))} registered non-baseline candidates and selected {_markdown_text(recommendation.get('selected_model_count'))} relevant, diverse candidates using training-split characteristics only.",
            "",
            _markdown_table(
                ["Rank", "Model", "Family", "Variant", f"CV {metric_label}", "CV std. dev."],
                model_rows,
            ) if model_rows else "No successful non-baseline comparisons were available.",
            "",
            "### Why this model was selected",
            "",
            f"**{_markdown_text(winner_name)}** ranked first on training-only cross-validation. Its {_markdown_text(metric_label)} was {_markdown_text(winner_score)} using {_markdown_text(cv.get('used_folds'))}-fold {_markdown_text(cv.get('splitter'))} validation.",
            "",
        ]
    )
    lines.extend(
        _bullet_lines(
            [_humanize(reason) for reason in _as_list(winner.get("selected_because"))],
            empty="The winner-reason list was not available.",
        )
    )
    if runner:
        lines.extend(
            [
                "",
                f"The runner-up was **{_markdown_text(runner.get('display_name'))}**, with a score gap of {_markdown_text(runner.get('score_gap_from_winner'))}. The winner was preferred because it produced the better validated primary score after bounded tuning; the holdout was not used to break the tie.",
            ]
        )

    lines.extend(
        [
            "",
            "### Baseline and confidence",
            "",
            _markdown_table(
                ["Evidence", "Result"],
                [
                    ["Dummy baseline score", baseline.get("baseline_primary_score")],
                    ["Winner beats baseline", baseline.get("winner_beats_baseline")],
                    ["Absolute improvement", baseline.get("absolute_improvement")],
                    ["Relative improvement", baseline.get("relative_improvement")],
                    ["Decision confidence", confidence.get("level")],
                    ["Close competition", confidence.get("close_competition")],
                ],
            ),
            "",
            "Confidence describes evidence from this model-selection run; it is not a guarantee of future production performance.",
            "",
            "## Hyperparameter tuning",
            "",
            _markdown_table(
                ["Item", "Value"],
                [
                    ["Status", tuning.get("status")],
                    ["Attempted models", tuning.get("attempted_model_count")],
                    ["Successful models", tuning.get("successful_model_count")],
                    ["Failed models", tuning.get("failed_model_count")],
                    ["Selected model", tuning.get("selected_model_name")],
                    ["Selected variant", tuning.get("selected_variant")],
                ],
            ),
            "",
            "## Untouched holdout evaluation",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [[_metric_label(key), value] for key, value in holdout.items() if key != "confusion_matrix"],
            ),
            "",
            "The holdout was evaluated once after model selection and tuning. It was not used by the recommender or cross-validation ranking.",
            "",
            "## Feature selection and explainability",
            "",
            _markdown_table(
                ["Feature-selection item", "Value"],
                [
                    ["Method", feature_selection.get("method")],
                    ["Configured percentile", feature_selection.get("percentile")],
                    ["Selected feature count", feature_selection.get("selected_feature_count")],
                    ["Selected transformed features", feature_selection.get("selected_feature_names")],
                    ["Explainability status", explanation.get("explainability_status")],
                    ["Evaluation scope", explanation.get("evaluation_scope")],
                    ["Used for model selection", explanation.get("used_for_model_selection")],
                ],
            ),
            "",
        ]
    )

    permutation_rows = _permutation_rows(training)
    if permutation_rows:
        lines.extend(
            [
                "### Permutation importance",
                "",
                _markdown_table(
                    ["Rank", "Raw feature", "Mean importance", "Std. dev."],
                    permutation_rows,
                ),
                "",
            ]
        )

    if charts:
        lines.extend(["## Visual evidence", ""])
        for chart in charts[:MAX_LIST_ITEMS]:
            lines.extend(
                [
                    f"### {_markdown_text(chart['title'])}",
                    "",
                    f"![{_markdown_text(chart['title'])}]({_markdown_text(chart['file_name'])})",
                    "",
                ]
            )

    warnings = [
        *_as_list(eda.get("warnings")),
        *_as_list(confidence.get("warnings")),
        *_as_list(_as_dict(eda.get("insights")).get("limitations")),
        *_as_list(eda.get("limitations")),
    ]
    lines.extend(["## Warnings, limitations, and responsible use", ""])
    lines.extend(
        _bullet_lines(
            [_humanize(item) for item in warnings],
            empty="No additional automated warning was recorded.",
        )
    )
    lines.extend(
        [
            "- Automated analysis does not replace domain, privacy, fairness, causal, or regulatory review.",
            "- Performance may change under data drift, population shift, label changes, or altered collection processes.",
            "- Joblib model files use pickle internally and must be loaded only from a trusted, hash-verified source.",
            "",
            "## Deployment and prediction guidance",
            "",
            "Use the exported `model_pipeline.joblib` rather than applying preprocessing manually. Validate the artifact manifest and SHA-256 hashes before loading it. Supply raw feature columns through the prediction schema endpoint; the fitted pipeline performs the same imputation, encoding, scaling, feature selection, and model inference used during validation.",
            "",
            "---",
            "",
            f"Report schema version: `{REPORT_VERSION}`. Private holdout rows included: `False`. Raw dataset rows included: `False`.",
            "",
        ]
    )
    return "\n".join(lines)


def _font_names() -> tuple[str, str]:
    regular_candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    bold_candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    regular = next((Path(path) for path in regular_candidates if Path(path).exists()), None)
    bold = next((Path(path) for path in bold_candidates if Path(path).exists()), None)
    if regular is None or bold is None:
        return "Helvetica", "Helvetica-Bold"

    try:
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
    except Exception:
        return "Helvetica", "Helvetica-Bold"
    return "ReportSans", "ReportSans-Bold"


def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _font_names()
    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "ReportCoverTitle", parent=sample["Title"], fontName=bold,
            fontSize=25, leading=30, textColor=_NAVY, alignment=TA_LEFT,
            spaceAfter=8 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "ReportCoverSubtitle", parent=sample["BodyText"], fontName=regular,
            fontSize=11, leading=16, textColor=_MUTED, spaceAfter=7 * mm,
        ),
        "h1": ParagraphStyle(
            "ReportH1", parent=sample["Heading1"], fontName=bold,
            fontSize=15, leading=19, textColor=_NAVY, spaceBefore=5 * mm,
            spaceAfter=3 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=sample["Heading2"], fontName=bold,
            fontSize=10.5, leading=14, textColor=_BLUE, spaceBefore=3 * mm,
            spaceAfter=2 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=sample["BodyText"], fontName=regular,
            fontSize=8.5, leading=12, textColor=_INK, spaceAfter=2.2 * mm,
        ),
        "small": ParagraphStyle(
            "ReportSmall", parent=sample["BodyText"], fontName=regular,
            fontSize=7.2, leading=9.5, textColor=_MUTED,
        ),
        "table": ParagraphStyle(
            "ReportTable", parent=sample["BodyText"], fontName=regular,
            fontSize=7.2, leading=9.2, textColor=_INK,
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader", parent=sample["BodyText"], fontName=bold,
            fontSize=7.2, leading=9.2, textColor=_WHITE,
        ),
        "callout": ParagraphStyle(
            "ReportCallout", parent=sample["BodyText"], fontName=regular,
            fontSize=9, leading=13, textColor=_NAVY,
        ),
    }


def _pdf_table(
    headers: Sequence[Any],
    rows: Iterable[Sequence[Any]],
    styles: dict[str, ParagraphStyle],
    *,
    widths: Sequence[float] | None = None,
) -> Table:
    body = list(rows)[:MAX_TABLE_ROWS]
    data = [
        [Paragraph(_paragraph_text(item), styles["table_header"]) for item in headers]
    ]
    data.extend(
        [Paragraph(_paragraph_text(item), styles["table"]) for item in row]
        for row in body
    )
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, _RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, colors.HexColor("#F6F9FB")]),
            ]
        )
    )
    return table


def _pdf_bullets(
    items: Iterable[Any],
    styles: dict[str, ParagraphStyle],
    *,
    empty: str,
) -> list[Paragraph]:
    bounded = list(items)[:MAX_LIST_ITEMS]
    if not bounded:
        bounded = [empty]
    return [
        Paragraph(f"<bullet>•</bullet>{_paragraph_text(item)}", styles["body"])
        for item in bounded
    ]


def _header_footer(canvas, document, *, title: str) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFillColor(_MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, height - 11 * mm, _plain_text(title, limit=74))
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.drawString(18 * mm, 9 * mm, "AI Data Scientist • Automated evidence report")
    canvas.restoreState()


def _chart_flowable(chart: dict[str, Any], styles: dict[str, ParagraphStyle]):
    if not chart["embedded_in_pdf"]:
        return [
            Paragraph(_paragraph_text(chart["title"]), styles["h2"]),
            Paragraph(
                "This SVG chart is included in the downloadable bundle but is not embedded in this PDF.",
                styles["body"],
            ),
        ]

    image = Image(str(chart["path"]))
    max_width = 166 * mm
    max_height = 170 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return [
        Paragraph(_paragraph_text(chart["title"]), styles["h2"]),
        Spacer(1, 1.5 * mm),
        image,
        Spacer(1, 4 * mm),
    ]


def _build_pdf(
    output_path: Path,
    training: dict[str, Any],
    eda: dict[str, Any],
    charts: list[dict[str, Any]],
    *,
    title: str,
    dataset_filename: str | None,
) -> None:
    styles = _styles()
    frame = Frame(18 * mm, 15 * mm, 174 * mm, 267 * mm, id="content")
    template = PageTemplate(
        id="main",
        frames=[frame],
        onPage=lambda canvas, doc: _header_footer(canvas, doc, title=title),
    )
    document = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="AI Data Scientist",
        subject="Automated machine-learning evidence report",
    )
    document.addPageTemplates([template])

    best = _as_dict(training.get("best_model"))
    rationale = _as_dict(training.get("model_selection_rationale"))
    winner = _as_dict(rationale.get("winner"))
    runner = _as_dict(rationale.get("runner_up"))
    confidence = _as_dict(rationale.get("decision_confidence"))
    baseline = _as_dict(rationale.get("baseline_assessment"))
    feature_selection = _as_dict(training.get("feature_selection"))
    holdout = _as_dict(training.get("holdout_metrics"))
    recommendation = _as_dict(training.get("model_recommendation"))
    tuning = _as_dict(training.get("hyperparameter_tuning"))
    explanation = _as_dict(training.get("explainability"))
    preprocessing = _preprocessing_summary(training)
    dataset = _dataset_summary(training, eda)
    split = _as_dict(training.get("split"))
    cv = _as_dict(training.get("cross_validation"))
    metric = training.get("primary_metric")
    metric_label = _metric_label(metric)
    winner_name = winner.get("display_name", best.get("display_name"))
    winner_score = winner.get(
        "cross_validation_primary_score", best.get("cross_validation_primary_score")
    )

    story = [
        Spacer(1, 16 * mm),
        Paragraph(_paragraph_text(title), styles["cover_title"]),
        Paragraph(
            "Automated EDA, preprocessing, candidate recommendation, model comparison, validation, and explainability",
            styles["cover_subtitle"],
        ),
        Table(
            [
                [Paragraph("TARGET", styles["small"]), Paragraph("SELECTED MODEL", styles["small"])],
                [Paragraph(_paragraph_text(training.get("target_column")), styles["callout"]), Paragraph(_paragraph_text(winner_name), styles["callout"])],
                [Paragraph("PROBLEM TYPE", styles["small"]), Paragraph(f"CV {_paragraph_text(metric_label.upper())}", styles["small"])],
                [Paragraph(_paragraph_text(_humanize(training.get("problem_type"))), styles["callout"]), Paragraph(_paragraph_text(winner_score), styles["callout"])],
            ],
            colWidths=[83 * mm, 83 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), _PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.8, _BLUE),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, _RULE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Spacer(1, 10 * mm),
        Paragraph(
            _paragraph_text(
                f"Dataset: {dataset_filename or 'not supplied'} • Decision confidence: {confidence.get('level', 'not available')} • Private holdout rows included: no"
            ),
            styles["body"],
        ),
        Spacer(1, 35 * mm),
        Paragraph(
            "This document explains what the system found, what it changed, which models it compared, why the winner was selected, and where human review is still required.",
            styles["cover_subtitle"],
        ),
        PageBreak(),
        Paragraph("1. Executive summary", styles["h1"]),
        _pdf_table(
            ["Item", "Result"],
            [
                ["Dataset", dataset_filename or "Not supplied"],
                ["Target", training.get("target_column")],
                ["Problem type", _humanize(training.get("problem_type"))],
                ["Selected model", winner_name],
                ["Selected variant", winner.get("selected_variant", best.get("selected_variant"))],
                [f"Cross-validation {metric_label}", winner_score],
                ["Decision confidence", confidence.get("level")],
                ["Training status", training.get("training_status")],
            ],
            styles,
            widths=[55 * mm, 111 * mm],
        ),
        Spacer(1, 3 * mm),
        *_pdf_bullets(
            _as_list(rationale.get("summary")), styles,
            empty="No selection summary was available.",
        ),
        Paragraph("2. Dataset and target overview", styles["h1"]),
        _pdf_table(
            ["Measure", "Value", "Measure", "Value"],
            [
                ["Original rows", dataset.get("rows"), "Original columns", dataset.get("columns")],
                ["Usable rows", dataset.get("usable_rows"), "Training rows", dataset.get("training_rows")],
                ["Holdout rows", dataset.get("holdout_rows"), "Missing target rows removed", dataset.get("missing_target_rows_removed")],
                ["Missing cells", dataset.get("missing_cells"), "Duplicate rows", dataset.get("duplicate_rows")],
            ],
            styles,
            widths=[40 * mm, 42 * mm, 42 * mm, 42 * mm],
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            _paragraph_text(
                f"The target '{training.get('target_column')}' was treated as {_humanize(training.get('problem_type'))}. The untouched holdout contained {split.get('test_rows')} rows and was evaluated only after selection and tuning."
            ),
            styles["body"],
        ),
        Paragraph("3. Data quality and preprocessing", styles["h1"]),
        _pdf_table(
            ["Decision", "Applied to modeling"],
            [
                ["Numerical features", preprocessing["numerical_features"]],
                ["Categorical features", preprocessing["categorical_features"]],
                ["Standard-scaled features", preprocessing["standard_scaler_features"]],
                ["Robust-scaled features", preprocessing["robust_scaler_features"]],
                ["Dropped columns", preprocessing["drop_columns"]],
                ["Manual-review exclusions", preprocessing["excluded_review_features"]],
                ["Potential leakage columns", preprocessing["leakage_columns"]],
                ["Rows removed for outliers", preprocessing["rows_removed_for_outliers"]],
            ],
            styles,
            widths=[52 * mm, 114 * mm],
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            "Numerical missing values use median imputation; categorical missing values use the most frequent value. Categorical inputs are one-hot encoded. Outliers remain in the data, with robust scaling used when recommended.",
            styles["body"],
        ),
        Paragraph("4. Exploratory findings", styles["h1"]),
    ]

    insights = _eda_insights(eda)
    if insights:
        insight_rows = [
            [
                item.get("priority_rank"),
                _humanize(item.get("severity")),
                item.get("title", item.get("finding", item.get("message"))),
                item.get("recommendation"),
            ]
            for item in insights
        ]
        story.append(
            _pdf_table(
                ["Priority", "Severity", "Finding", "Recommended action"],
                insight_rows,
                styles,
                widths=[16 * mm, 24 * mm, 61 * mm, 65 * mm],
            )
        )
    else:
        story.append(Paragraph("No standalone EDA insight report was supplied for this export.", styles["body"]))

    model_rows = _model_rows(training)
    story.extend(
        [
            Paragraph("5. Candidate strategy and comparison", styles["h1"]),
            Paragraph(
                _paragraph_text(
                    f"Selection mode: {_humanize(recommendation.get('selection_mode'))}. The dataset-aware recommender considered {recommendation.get('registered_non_baseline_model_count')} registered non-baseline candidates and selected {recommendation.get('selected_model_count')} relevant, diverse candidates using training-split information only."
                ),
                styles["body"],
            ),
        ]
    )
    if model_rows:
        story.append(
            _pdf_table(
                ["Rank", "Model", "Family", "Variant", f"CV {metric_label}", "Std. dev."],
                model_rows,
                styles,
                widths=[12 * mm, 48 * mm, 27 * mm, 23 * mm, 29 * mm, 27 * mm],
            )
        )

    story.extend(
        [
            Paragraph("6. Final model decision", styles["h1"]),
            Table(
                [[Paragraph(
                    f"<b>{_paragraph_text(winner_name)}</b><br/>ranked first using training-only cross-validation. The holdout did not select the winner.",
                    styles["callout"],
                )]],
                colWidths=[166 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), _PALE_GREEN),
                        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#4D9C73")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 2 * mm),
            *_pdf_bullets(
                [_humanize(reason) for reason in _as_list(winner.get("selected_because"))],
                styles,
                empty="The winner-reason list was not available.",
            ),
        ]
    )
    if runner:
        story.append(
            Paragraph(
                _paragraph_text(
                    f"Runner-up: {runner.get('display_name')} (gap from winner: {runner.get('score_gap_from_winner')}). The winner had the better validated primary score after bounded tuning."
                ),
                styles["body"],
            )
        )

    story.extend(
        [
            _pdf_table(
                ["Evidence", "Result"],
                [
                    ["Dummy baseline score", baseline.get("baseline_primary_score")],
                    ["Winner beats baseline", baseline.get("winner_beats_baseline")],
                    ["Absolute improvement", baseline.get("absolute_improvement")],
                    ["Relative improvement", baseline.get("relative_improvement")],
                    ["Decision confidence", confidence.get("level")],
                    ["Close competition", confidence.get("close_competition")],
                ],
                styles,
                widths=[70 * mm, 96 * mm],
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                "Confidence describes this selection run and is not a guarantee of future production performance.",
                styles["small"],
            ),
            Paragraph("7. Tuning and holdout evaluation", styles["h1"]),
            _pdf_table(
                ["Tuning item", "Value", "Holdout metric", "Value"],
                [
                    ["Status", tuning.get("status"), _metric_label(metric), holdout.get(metric)],
                    ["Attempted models", tuning.get("attempted_model_count"), "Rows", split.get("test_rows")],
                    ["Successful models", tuning.get("successful_model_count"), "Used for selection", False],
                    ["Selected variant", tuning.get("selected_variant"), "Stratified split", split.get("stratified")],
                ],
                styles,
                widths=[42 * mm, 40 * mm, 44 * mm, 40 * mm],
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                _paragraph_text(
                    "Additional holdout metrics: "
                    + "; ".join(
                        f"{_metric_label(key)}={_plain_text(value)}"
                        for key, value in holdout.items()
                        if key not in {metric, "confusion_matrix", "class_labels"}
                    )
                ),
                styles["body"],
            ),
            Paragraph("8. Feature selection and explainability", styles["h1"]),
            _pdf_table(
                ["Item", "Value"],
                [
                    ["Feature-selection method", feature_selection.get("method")],
                    ["Selected feature count", feature_selection.get("selected_feature_count")],
                    ["Selected transformed features", feature_selection.get("selected_feature_names")],
                    ["Explainability status", explanation.get("explainability_status")],
                    ["Evaluation scope", explanation.get("evaluation_scope")],
                    ["Used for model selection", explanation.get("used_for_model_selection")],
                ],
                styles,
                widths=[55 * mm, 111 * mm],
            ),
        ]
    )

    permutation_rows = _permutation_rows(training)
    if permutation_rows:
        story.extend(
            [
                Paragraph("Permutation importance", styles["h2"]),
                _pdf_table(
                    ["Rank", "Raw feature", "Mean importance", "Std. dev."],
                    permutation_rows,
                    styles,
                    widths=[18 * mm, 64 * mm, 43 * mm, 41 * mm],
                ),
            ]
        )

    if charts:
        story.extend([PageBreak(), Paragraph("9. Visual evidence", styles["h1"])])
        for chart in charts[:MAX_LIST_ITEMS]:
            story.extend(_chart_flowable(chart, styles))

    warning_items = [
        *_as_list(eda.get("warnings")),
        *_as_list(confidence.get("warnings")),
        *_as_list(_as_dict(eda.get("insights")).get("limitations")),
        *_as_list(eda.get("limitations")),
    ]
    story.extend(
        [
            Paragraph("10. Warnings, limitations, and responsible use", styles["h1"]),
            *_pdf_bullets(
                [_humanize(item) for item in warning_items], styles,
                empty="No additional automated warning was recorded.",
            ),
            Paragraph(
                "Automated analysis does not replace domain, privacy, fairness, causal, or regulatory review. Performance may change under drift or population shift.",
                styles["body"],
            ),
            Paragraph("11. Deployment and prediction guidance", styles["h1"]),
            Paragraph(
                "Use the exported model pipeline rather than recreating preprocessing manually. Verify the artifact manifest and SHA-256 hashes before loading. Supply raw feature columns through the prediction schema endpoint. Joblib uses pickle internally, so load only trusted artifacts.",
                styles["body"],
            ),
            Spacer(1, 4 * mm),
            Table(
                [[Paragraph(
                    f"Report schema {REPORT_VERSION} • Raw dataset rows included: no • Private holdout rows included: no",
                    styles["small"],
                )]],
                colWidths=[166 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), _PALE_AMBER),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C99A2E")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
        ]
    )
    document.build(story)


def generate_professional_report(
    training_report: dict[str, Any],
    output_directory: str | Path,
    *,
    eda_report: dict[str, Any] | None = None,
    chart_paths: dict[str, str | Path] | None = None,
    dataset_filename: str | None = None,
    title: str | None = None,
    formats: Sequence[str] = SUPPORTED_FORMATS,
) -> dict[str, Any]:
    """Create Markdown/PDF evidence reports without mutating inputs."""

    if not isinstance(training_report, dict):
        raise TypeError("training_report must be a dictionary.")
    if eda_report is not None and not isinstance(eda_report, dict):
        raise TypeError("eda_report must be a dictionary or None.")

    required = {
        "training_status",
        "target_column",
        "problem_type",
        "best_model",
        "holdout_metrics",
    }
    missing = sorted(required - training_report.keys())
    if missing:
        raise ValueError(f"training_report is missing required fields: {missing}")

    resolved_formats = _validated_formats(formats)
    output_path = Path(output_directory).resolve()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_directory exists but is not a directory.")
    output_path.mkdir(parents=True, exist_ok=True)

    source_training = deepcopy(training_report)
    source_eda = deepcopy(eda_report or {})
    report_title = _plain_text(
        title or "Automated Machine Learning Report",
        limit=120,
    )
    charts = _validated_charts(chart_paths, output_path)
    files: dict[str, dict[str, Any]] = {}

    if "markdown" in resolved_formats:
        markdown_path = output_path / MARKDOWN_REPORT_FILENAME
        temporary = _temporary_path(output_path, ".md")
        try:
            markdown = _markdown_report(
                source_training,
                source_eda,
                charts,
                title=report_title,
                dataset_filename=dataset_filename,
            )
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(markdown)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, markdown_path)
        finally:
            temporary.unlink(missing_ok=True)

        files["professional_report_markdown"] = {
            "file_name": markdown_path.name,
            "format": "markdown",
            "media_type": "text/markdown",
            "sha256": _sha256(markdown_path),
            "size_bytes": markdown_path.stat().st_size,
        }

    if "pdf" in resolved_formats:
        pdf_path = output_path / PDF_REPORT_FILENAME
        temporary = _temporary_path(output_path, ".pdf")
        try:
            _build_pdf(
                temporary,
                source_training,
                source_eda,
                charts,
                title=report_title,
                dataset_filename=dataset_filename,
            )
            os.replace(temporary, pdf_path)
        finally:
            temporary.unlink(missing_ok=True)

        files["professional_report_pdf"] = {
            "file_name": pdf_path.name,
            "format": "pdf",
            "media_type": "application/pdf",
            "sha256": _sha256(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
        }

    return {
        "report_status": "complete",
        "report_version": REPORT_VERSION,
        "title": report_title,
        "formats": list(resolved_formats),
        "files": files,
        "chart_count": len(charts),
        "embedded_pdf_chart_count": sum(
            bool(chart["embedded_in_pdf"]) for chart in charts
        ),
        "source_training_report_modified": training_report != source_training,
        "source_eda_report_modified": (eda_report or {}) != source_eda,
        "raw_dataset_rows_included": False,
        "private_holdout_rows_included": False,
    }
