from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.core import dataset_files
from ml_pipeline.eda_config import EDAConfig
from ml_pipeline.analyzer import analyze_target
from ml_pipeline.eda import generate_eda_report
from ml_pipeline.model_artifacts import (
    MANIFEST_FILENAME,
    PIPELINE_FILENAME,
    REPORT_FILENAME,
    export_model_artifacts,
)
from ml_pipeline.explainability_config import (
    ExplainabilityConfig,
)
from ml_pipeline.explainability_visualizations import (
    generate_explainability_visualizations,
)
from ml_pipeline.model_explainability import (
    generate_model_explanations,
)
from ml_pipeline.professional_report import (
    generate_professional_report,
)
from ml_pipeline.inference import (
    DEFAULT_MAX_PREDICTION_ROWS,
    InferenceArtifactError,
    InferenceInputError,
    describe_model_input,
    predict_with_model_artifact,
)
from ml_pipeline.training import train_and_compare_models
from ml_pipeline.training_config import TrainingConfig
from ml_pipeline.tuning_config import TuningConfig


router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


class TargetSelectionRequest(BaseModel):
    filename: str = Field(min_length=1)
    target_column: str = Field(min_length=1)


class EDARequest(BaseModel):
    """Options accepted by the generalized automated EDA endpoint."""

    filename: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    problem_type: Literal["classification", "regression"] | None = None
    include_feature_relationships: bool = True
    include_preprocessing_context: bool = True
    include_insights: bool = True
    include_visualizations: bool = True
    visualization_image_format: Literal["png", "svg"] = "png"
    visualization_dpi: int = Field(default=120, ge=72, le=300)
    configuration_overrides: dict[str, Any] = Field(
        default_factory=dict
    )


class TrainingRequest(BaseModel):
    """Bounded options for generalized model training and export."""

    filename: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    problem_type: Literal["classification", "regression"] | None = None
    requested_models: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=5,
    )
    max_recommended_models: int | None = Field(
        default=None,
        ge=1,
        le=6,
    )
    run_tuning: bool = True
    tuning_iterations: int = Field(default=12, ge=1, le=50)
    tuning_top_models: int = Field(default=2, ge=1, le=4)
    include_explanations: bool = True
    explainability_repeats: int = Field(default=5, ge=1, le=20)
    explainability_max_rows: int = Field(
        default=2_000,
        ge=10,
        le=10_000,
    )
    explainability_top_n: int = Field(default=20, ge=1, le=100)
    include_explanation_visualizations: bool = False
    explanation_image_format: Literal["png", "svg"] = "png"
    explanation_visualization_dpi: int = Field(
        default=120,
        ge=72,
        le=300,
    )
    include_professional_report: bool = False
    professional_report_formats: list[
        Literal["markdown", "pdf"]
    ] = Field(
        default_factory=lambda: ["markdown", "pdf"],
        min_length=1,
        max_length=2,
    )
    professional_report_title: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
    )
    include_eda_visualizations_in_report: bool = True
    export_artifacts: bool = True
    allow_potential_leakage: bool = False
    training_configuration_overrides: dict[str, Any] = Field(
        default_factory=dict
    )


class PredictionRequest(BaseModel):
    """Bounded raw records for one verified trained model."""

    training_id: str = Field(min_length=32, max_length=32)
    records: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=DEFAULT_MAX_PREDICTION_ROWS,
    )
    include_probabilities: bool = True


def _load_uploaded_dataframe(filename: str):
    """Resolve and load an uploaded dataset with API error mapping."""

    try:
        file_path = dataset_files.resolve_uploaded_dataset(filename)
        dataframe = dataset_files.read_dataset(file_path)
    except dataset_files.DatasetFilenameError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except dataset_files.DatasetReadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return file_path.name, dataframe


def _build_eda_config(overrides: dict[str, Any]) -> EDAConfig:
    """Validate request-specific EDA thresholds using EDAConfig."""

    try:
        return EDAConfig(**overrides)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid EDA configuration: {error}",
        ) from error


def _build_training_config(overrides: dict[str, Any]) -> TrainingConfig:
    """Validate request-specific training controls."""

    try:
        config = TrainingConfig(**overrides)
        config.validate()
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid training configuration: {error}",
        ) from error

    if config.n_jobs != 1:
        raise HTTPException(
            status_code=400,
            detail="API training requires n_jobs=1.",
        )

    return config


def _public_eda_report(report: dict, analysis_id: str | None) -> dict:
    """Remove server paths and attach safe URLs to generated plots."""

    public_report = deepcopy(report)
    execution_options = public_report.get("execution_options")

    if isinstance(execution_options, dict):
        execution_options["visualization_output_directory"] = None

    visualizations = public_report.get("visualizations")
    if not isinstance(visualizations, dict):
        return public_report

    visualizations["output_directory"] = None

    for plot in visualizations.get("generated_plots", []):
        if not isinstance(plot, dict):
            continue

        plot.pop("file_path", None)
        file_name = plot.get("file_name")

        if analysis_id and isinstance(file_name, str):
            plot["artifact_url"] = (
                "/analysis/eda/artifacts/"
                f"{analysis_id}/{quote(file_name)}"
            )

    return public_report


def _public_model_artifacts(
    export_result: dict,
    training_id: str,
) -> dict:
    """Return artifact metadata and URLs without server paths."""

    manifest = deepcopy(export_result["manifest"])
    base_url = f"/analysis/train/artifacts/{training_id}"
    downloads = []

    for artifact_type, file_name in (
        ("pipeline", PIPELINE_FILENAME),
        ("training_report", REPORT_FILENAME),
    ):
        download = {
            "artifact_type": artifact_type,
            "file_name": file_name,
            "artifact_url": f"{base_url}/{quote(file_name)}",
        }

        manifest_entry = manifest.get("files", {}).get(
            artifact_type
        )
        if isinstance(manifest_entry, dict):
            download["sha256"] = manifest_entry.get("sha256")
            download["size_bytes"] = manifest_entry.get(
                "size_bytes"
            )

        downloads.append(download)

    manifest_files = manifest.get("files", {})

    for artifact_type in sorted(manifest_files):
        if artifact_type in {"pipeline", "training_report"}:
            continue

        manifest_entry = manifest_files[artifact_type]

        if not isinstance(manifest_entry, dict):
            continue

        file_name = manifest_entry.get("file_name")

        if not isinstance(file_name, str):
            continue

        downloads.append(
            {
                "artifact_type": artifact_type,
                "file_name": file_name,
                "artifact_url": f"{base_url}/{quote(file_name)}",
                "sha256": manifest_entry.get("sha256"),
                "size_bytes": manifest_entry.get("size_bytes"),
            }
        )

    downloads.append(
        {
            "artifact_type": "manifest",
            "file_name": MANIFEST_FILENAME,
            "artifact_url": (
                f"{base_url}/{quote(MANIFEST_FILENAME)}"
            ),
        }
    )

    return {
        "artifact_status": export_result["artifact_status"],
        "prediction_equivalence_verified": manifest[
            "prediction_equivalence_verified"
        ],
        "verification_rows": manifest["verification_rows"],
        "manifest": manifest,
        "downloads": downloads,
    }


def _resolve_model_manifest(training_id: str) -> Path:
    """Resolve one server-owned model manifest with API errors."""

    try:
        return dataset_files.resolve_model_artifact(
            training_id,
            MANIFEST_FILENAME,
        )
    except dataset_files.ArtifactPathError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@router.post("/target")
def analyze_selected_target(request: TargetSelectionRequest):
    """Validate and analyze a target explicitly selected by the user."""

    safe_filename, df = _load_uploaded_dataframe(request.filename)

    try:
        target_analysis = analyze_target(df, request.target_column)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    candidate_features = [
        column
        for column in df.columns
        if column != request.target_column
    ]

    return {
        "filename": safe_filename,
        "selected_target": request.target_column,
        "target_analysis": target_analysis,
        "candidate_feature_count": len(candidate_features),
        "candidate_features": candidate_features,
        "message": (
            "Target validated successfully. Candidate features "
            "are not yet the final training features."
        ),
    }


@router.post("/eda")
def analyze_dataset_with_eda(request: EDARequest):
    """Generate a generalized, JSON-safe EDA report for one dataset."""

    safe_filename, df = _load_uploaded_dataframe(request.filename)
    config = _build_eda_config(request.configuration_overrides)

    if request.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{request.target_column}' does not "
                "exist in the dataset."
            ),
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    analysis_id = None
    output_directory: Path | None = None

    if request.include_visualizations:
        try:
            analysis_id, output_directory = (
                dataset_files.create_eda_output_directory()
            )
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail="Unable to create visualization storage.",
            ) from error

    try:
        report = generate_eda_report(
            df,
            request.target_column,
            config=config,
            problem_type=request.problem_type,
            include_feature_relationships=(
                request.include_feature_relationships
            ),
            include_preprocessing_context=(
                request.include_preprocessing_context
            ),
            include_insights=request.include_insights,
            include_visualizations=request.include_visualizations,
            visualization_output_directory=output_directory,
            visualization_image_format=(
                request.visualization_image_format
            ),
            visualization_dpi=request.visualization_dpi,
            fail_fast=False,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="EDA generation failed unexpectedly.",
        ) from error

    public_report = _public_eda_report(report, analysis_id)

    return {
        "filename": safe_filename,
        "selected_target": request.target_column,
        "analysis_id": analysis_id,
        "artifact_base_url": (
            f"/analysis/eda/artifacts/{analysis_id}"
            if analysis_id is not None
            else None
        ),
        "report": public_report,
        "message": (
            "EDA completed. Findings and preprocessing actions remain "
            "recommendations until explicitly approved."
        ),
    }


@router.get("/eda/artifacts/{analysis_id}/{artifact_filename}")
def download_eda_artifact(
    analysis_id: str,
    artifact_filename: str,
):
    """Return one plot belonging to a validated EDA analysis run."""

    try:
        artifact_path = dataset_files.resolve_eda_artifact(
            analysis_id,
            artifact_filename,
        )
    except dataset_files.ArtifactPathError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    media_type = (
        "image/png"
        if artifact_path.suffix.lower() == ".png"
        else "image/svg+xml"
    )

    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=artifact_path.name,
    )


@router.post("/train")
def train_dataset_models(request: TrainingRequest):
    """Train, compare, optionally tune, and export one fitted pipeline."""

    if request.include_professional_report and not request.export_artifacts:
        raise HTTPException(
            status_code=400,
            detail=(
                "Professional reports require export_artifacts=True."
            ),
        )

    safe_filename, df = _load_uploaded_dataframe(request.filename)
    training_overrides = dict(
        request.training_configuration_overrides
    )

    if request.max_recommended_models is not None:
        training_overrides["max_recommended_models"] = (
            request.max_recommended_models
        )

    config = _build_training_config(training_overrides)
    tuning_config = None

    if request.run_tuning:
        tuning_config = TuningConfig(
            n_iter=request.tuning_iterations,
            top_models=request.tuning_top_models,
            random_state=config.random_state,
            n_jobs=1,
        )

    try:
        training_result = train_and_compare_models(
            df,
            request.target_column,
            problem_type=request.problem_type,
            config=config,
            tuning_config=tuning_config,
            requested_models=request.requested_models,
            allow_potential_leakage=(
                request.allow_potential_leakage
            ),
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Model training failed unexpectedly.",
        ) from error

    if request.include_explanations:
        explainability_config = ExplainabilityConfig(
            n_repeats=request.explainability_repeats,
            random_state=config.random_state,
            max_rows=request.explainability_max_rows,
            top_n=request.explainability_top_n,
            n_jobs=1,
        )

        try:
            explanation = generate_model_explanations(
                training_result,
                config=explainability_config,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Model explainability failed: {error}",
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Model explainability failed unexpectedly.",
            ) from error

        training_result["report"]["explainability"] = explanation
    else:
        training_result["report"]["explainability"] = {
            "explainability_status": "skipped",
            "reason": "disabled_by_request",
        }

    training_id = None
    public_artifacts = None
    explanation = training_result["report"]["explainability"]

    if request.export_artifacts:
        try:
            training_id, output_directory = (
                dataset_files.create_model_output_directory()
            )
            supplementary_artifacts = {}
            report_chart_paths = {}

            if (
                (
                    request.include_explanation_visualizations
                    or request.include_professional_report
                )
                and explanation.get("explainability_status")
                == "complete"
            ):
                visualization_report = (
                    generate_explainability_visualizations(
                        explanation,
                        output_directory,
                        image_format=(
                            "png"
                            if (
                                request.include_professional_report
                                and "pdf"
                                in request.professional_report_formats
                            )
                            else request.explanation_image_format
                        ),
                        dpi=(
                            request.explanation_visualization_dpi
                        ),
                    )
                )
                explanation["visualizations"] = visualization_report

                for plot in visualization_report[
                    "generated_plots"
                ]:
                    plot_path = output_directory / plot["file_name"]
                    supplementary_artifacts[plot["plot_id"]] = plot_path
                    report_chart_paths[plot["plot_id"]] = plot_path
            elif (
                request.include_explanation_visualizations
                or request.include_professional_report
            ):
                explanation["visualizations"] = {
                    "visualization_status": "skipped",
                    "reason": "explanation_unavailable",
                    "generated_plot_count": 0,
                    "generated_plots": [],
                    "raw_holdout_rows_used": False,
                }

            if request.include_professional_report:
                report_eda = generate_eda_report(
                    df,
                    request.target_column,
                    problem_type=training_result["report"][
                        "problem_type"
                    ],
                    include_feature_relationships=True,
                    include_preprocessing_context=True,
                    include_insights=True,
                    include_visualizations=(
                        request.include_eda_visualizations_in_report
                    ),
                    visualization_output_directory=(
                        output_directory
                    ),
                    visualization_image_format="png",
                    visualization_dpi=120,
                    fail_fast=False,
                )

                eda_visualizations = report_eda.get(
                    "visualizations"
                )
                if isinstance(eda_visualizations, dict):
                    for plot in eda_visualizations.get(
                        "generated_plots",
                        [],
                    ):
                        if not isinstance(plot, dict):
                            continue
                        plot_id = plot.get("plot_id")
                        file_name = plot.get("file_name")
                        if not (
                            isinstance(plot_id, str)
                            and isinstance(file_name, str)
                        ):
                            continue
                        plot_path = output_directory / file_name
                        artifact_type = f"eda_{plot_id}"
                        supplementary_artifacts[
                            artifact_type
                        ] = plot_path
                        report_chart_paths[artifact_type] = plot_path

                professional_report = generate_professional_report(
                    training_result["report"],
                    output_directory,
                    eda_report=report_eda,
                    chart_paths=report_chart_paths,
                    dataset_filename=safe_filename,
                    title=request.professional_report_title,
                    formats=request.professional_report_formats,
                )
                training_result["report"][
                    "professional_report"
                ] = professional_report

                for artifact_type, file_report in (
                    professional_report["files"].items()
                ):
                    supplementary_artifacts[artifact_type] = (
                        output_directory
                        / file_report["file_name"]
                    )

            sample_input = df.drop(
                columns=[request.target_column]
            ).head(20)
            export_result = export_model_artifacts(
                training_result,
                output_directory,
                sample_input=sample_input,
                supplementary_artifacts=supplementary_artifacts,
            )
            public_artifacts = _public_model_artifacts(
                export_result,
                training_id,
            )

            if request.include_professional_report:
                report_files = training_result["report"][
                    "professional_report"
                ]["files"]
                base_url = (
                    f"/analysis/train/artifacts/{training_id}"
                )
                for file_report in report_files.values():
                    file_report["artifact_url"] = (
                        f"{base_url}/"
                        f"{quote(file_report['file_name'])}"
                    )
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail="Unable to create model artifact storage.",
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="Model artifact export failed unexpectedly.",
            ) from error
    elif request.include_explanation_visualizations:
        explanation["visualizations"] = {
            "visualization_status": "skipped",
            "reason": "artifact_export_disabled",
            "generated_plot_count": 0,
            "generated_plots": [],
            "raw_holdout_rows_used": False,
        }

    return {
        "filename": safe_filename,
        "selected_target": request.target_column,
        "training_id": training_id,
        "artifact_base_url": (
            f"/analysis/train/artifacts/{training_id}"
            if training_id is not None
            else None
        ),
        "report": training_result["report"],
        "artifacts": public_artifacts,
        "message": (
            "Training completed. Holdout metrics and optional model "
            "explanations were calculated only after model comparison "
            "and optional tuning."
        ),
    }


@router.get("/predict/schema/{training_id}")
def inspect_prediction_schema(training_id: str):
    """Describe the raw fields accepted by one verified model."""

    manifest_path = _resolve_model_manifest(training_id)

    try:
        schema_report = describe_model_input(manifest_path)
    except InferenceArtifactError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to inspect the model input schema.",
        ) from error

    return {
        "training_id": training_id,
        **schema_report,
        "message": (
            "Supply every required feature. Optional features may be "
            "omitted and are reconstructed as missing values."
        ),
    }


@router.post("/predict")
def predict_with_trained_model(request: PredictionRequest):
    """Generate single-row or bounded batch predictions."""

    manifest_path = _resolve_model_manifest(request.training_id)

    try:
        prediction_report = predict_with_model_artifact(
            manifest_path,
            request.records,
            include_probabilities=request.include_probabilities,
            max_records=DEFAULT_MAX_PREDICTION_ROWS,
        )
    except InferenceInputError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except InferenceArtifactError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed unexpectedly.",
        ) from error

    return {
        "training_id": request.training_id,
        "report": prediction_report,
        "message": (
            "Predictions were generated by the verified fitted "
            "pipeline without retraining or updating the model."
        ),
    }


@router.get("/train/artifacts/{training_id}/{artifact_filename}")
def download_model_artifact(
    training_id: str,
    artifact_filename: str,
):
    """Download one approved artifact from a validated training run."""

    try:
        artifact_path = dataset_files.resolve_model_artifact(
            training_id,
            artifact_filename,
        )
    except dataset_files.ArtifactPathError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    media_type = (
        "application/json"
        if artifact_path.suffix.lower() == ".json"
        else "application/pdf"
        if artifact_path.suffix.lower() == ".pdf"
        else "text/markdown"
        if artifact_path.suffix.lower() == ".md"
        else "image/png"
        if artifact_path.suffix.lower() == ".png"
        else "image/svg+xml"
        if artifact_path.suffix.lower() == ".svg"
        else "application/octet-stream"
    )

    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=artifact_path.name,
    )
