import os
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
EDA_OUTPUT_DIR = BACKEND_DIR / "outputs" / "eda"
MODEL_OUTPUT_DIR = BACKEND_DIR / "outputs" / "models" / "api"

ALLOWED_DATASET_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ALLOWED_ARTIFACT_EXTENSIONS = {".png", ".svg"}
ALLOWED_MODEL_ARTIFACT_FILENAMES = {
    "model_pipeline.joblib",
    "training_report.json",
    "artifact_manifest.json",
    "permutation_importance_chart.png",
    "permutation_importance_chart.svg",
    "native_importance_chart.png",
    "native_importance_chart.svg",
    "model_report.md",
    "model_report.pdf",
    "target_distribution.png",
    "target_distribution.svg",
    "missingness.png",
    "missingness.svg",
    "numerical_distributions.png",
    "numerical_distributions.svg",
    "categorical_distributions.png",
    "categorical_distributions.svg",
    "correlation_heatmap.png",
    "correlation_heatmap.svg",
    "feature_target_relationships.png",
    "feature_target_relationships.svg",
    "datetime_overview.png",
    "datetime_overview.svg",
}


class DatasetFilenameError(ValueError):
    """Raised when a supplied dataset filename is unsafe or unsupported."""


class DatasetReadError(ValueError):
    """Raised when a supported dataset cannot be parsed."""


class ArtifactPathError(ValueError):
    """Raised when an EDA artifact identifier or filename is unsafe."""


def validate_dataset_filename(filename: str) -> str:
    """Return a safe basename for a supported tabular dataset."""

    if not isinstance(filename, str) or not filename.strip():
        raise DatasetFilenameError("A dataset filename is required.")

    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
        or "\x00" in filename
        or any(ord(character) < 32 for character in filename)
        or Path(filename).name != filename
    ):
        raise DatasetFilenameError("Invalid filename.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_DATASET_EXTENSIONS:
        raise DatasetFilenameError("Unsupported dataset format.")

    return filename


def ensure_storage_directories() -> None:
    """Create the application-owned storage directories when required."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_uploaded_dataset(filename: str) -> Path:
    """Resolve an existing uploaded dataset without allowing traversal."""

    safe_filename = validate_dataset_filename(filename)
    upload_root = UPLOAD_DIR.resolve()
    candidate = (upload_root / safe_filename).resolve()

    if candidate.parent != upload_root:
        raise DatasetFilenameError("Invalid filename.")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("Uploaded dataset was not found.")

    return candidate


def read_dataset(file_path: Path) -> pd.DataFrame:
    """Read a supported CSV or Excel dataset into a DataFrame."""

    extension = file_path.suffix.lower()
    if extension not in ALLOWED_DATASET_EXTENSIONS:
        raise DatasetFilenameError("Unsupported dataset format.")

    try:
        if extension == ".csv":
            return pd.read_csv(file_path)

        return pd.read_excel(file_path)
    except Exception as error:
        raise DatasetReadError(
            f"Unable to read dataset: {error}"
        ) from error


def create_eda_output_directory() -> tuple[str, Path]:
    """Create a collision-resistant directory for one EDA API run."""

    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _ in range(10):
        analysis_id = uuid4().hex
        output_directory = EDA_OUTPUT_DIR / analysis_id

        try:
            output_directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue

        return analysis_id, output_directory.resolve()

    raise RuntimeError("Unable to allocate an EDA output directory.")


def create_model_output_directory() -> tuple[str, Path]:
    """Create a collision-resistant directory for one training run."""

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _ in range(10):
        training_id = uuid4().hex
        output_directory = MODEL_OUTPUT_DIR / training_id

        try:
            output_directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue

        return training_id, output_directory.resolve()

    raise RuntimeError("Unable to allocate a model output directory.")


def resolve_eda_artifact(
    analysis_id: str,
    artifact_filename: str,
) -> Path:
    """Resolve one generated plot using a validated run identifier."""

    try:
        parsed_id = UUID(analysis_id)
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactPathError("Invalid analysis identifier.") from error

    if parsed_id.hex != analysis_id:
        raise ArtifactPathError("Invalid analysis identifier.")

    if (
        not artifact_filename
        or artifact_filename in {".", ".."}
        or "/" in artifact_filename
        or "\\" in artifact_filename
        or "\x00" in artifact_filename
        or Path(artifact_filename).name != artifact_filename
    ):
        raise ArtifactPathError("Invalid artifact filename.")

    if (
        Path(artifact_filename).suffix.lower()
        not in ALLOWED_ARTIFACT_EXTENSIONS
    ):
        raise ArtifactPathError("Unsupported artifact format.")

    output_root = EDA_OUTPUT_DIR.resolve()
    run_directory = (output_root / analysis_id).resolve()
    candidate = (run_directory / artifact_filename).resolve()

    if (
        run_directory.parent != output_root
        or candidate.parent != run_directory
    ):
        raise ArtifactPathError("Invalid artifact path.")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("EDA artifact was not found.")

    return candidate


def resolve_model_artifact(
    training_id: str,
    artifact_filename: str,
) -> Path:
    """Resolve one approved artifact from a validated training run."""

    try:
        parsed_id = UUID(training_id)
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactPathError("Invalid training identifier.") from error

    if parsed_id.hex != training_id:
        raise ArtifactPathError("Invalid training identifier.")

    if (
        not artifact_filename
        or artifact_filename in {".", ".."}
        or "/" in artifact_filename
        or "\\" in artifact_filename
        or "\x00" in artifact_filename
        or Path(artifact_filename).name != artifact_filename
    ):
        raise ArtifactPathError("Invalid artifact filename.")

    if artifact_filename not in ALLOWED_MODEL_ARTIFACT_FILENAMES:
        raise ArtifactPathError("Unsupported model artifact.")

    output_root = MODEL_OUTPUT_DIR.resolve()
    run_directory = (output_root / training_id).resolve()
    candidate = (run_directory / artifact_filename).resolve()

    if (
        run_directory.parent != output_root
        or candidate.parent != run_directory
    ):
        raise ArtifactPathError("Invalid artifact path.")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("Model artifact was not found.")

    return candidate


def replace_file_atomically(source: Path, destination: Path) -> None:
    """Atomically publish a fully validated uploaded file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
