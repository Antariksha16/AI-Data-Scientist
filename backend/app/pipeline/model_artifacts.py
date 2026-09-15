import hashlib
import json
import os
import platform
import re
import tempfile
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ARTIFACT_FORMAT_VERSION = "1.0"
PIPELINE_FILENAME = "model_pipeline.joblib"
REPORT_FILENAME = "training_report.json"
MANIFEST_FILENAME = "artifact_manifest.json"

_JOBLIB_NUMPY_SHAPE_WARNING = (
    r"Setting the shape on a NumPy array has been deprecated "
    r"in NumPy 2\.5\."
)


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _load_joblib(file_path: Path) -> Any:
    """Load Joblib while isolating its NumPy 2.5 deprecation."""

    # Joblib 1.5.3 assigns directly to ndarray.shape while rebuilding
    # arrays. NumPy 2.5 deprecates that internal operation. Restrict
    # the filter to Joblib's loader and the exact upstream message so
    # every other warning remains visible, including under -W error.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_JOBLIB_NUMPY_SHAPE_WARNING,
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        return joblib.load(file_path)


def _temporary_path(
    output_directory: Path,
    prefix: str,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_directory,
        prefix=prefix,
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _write_json(
    file_path: Path,
    value: dict,
) -> None:
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            value,
            file_handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file_handle.write("\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _predictions_match(
    expected: Any,
    actual: Any,
) -> bool:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)

    if expected_array.shape != actual_array.shape:
        return False

    if (
        np.issubdtype(expected_array.dtype, np.number)
        and np.issubdtype(actual_array.dtype, np.number)
    ):
        return bool(
            np.allclose(
                expected_array,
                actual_array,
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
        )

    return bool(np.array_equal(expected_array, actual_array))


def _safe_bundle_file(
    directory: Path,
    file_name: str,
) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Artifact filename must be a non-empty string.")

    if Path(file_name).name != file_name:
        raise ValueError("Artifact manifest contains an unsafe filename.")

    resolved_directory = directory.resolve()
    resolved_file = (directory / file_name).resolve()

    if resolved_file.parent != resolved_directory:
        raise ValueError("Artifact path escapes its bundle directory.")

    return resolved_file


def _build_input_schema(
    pipeline: Pipeline,
    report: dict[str, Any],
    sample_input: pd.DataFrame | None,
) -> dict[str, Any]:
    """Describe raw inference columns without storing sample values."""

    if sample_input is not None:
        raw_feature_names = [
            str(column)
            for column in sample_input.columns
        ]
        pandas_dtypes = {
            str(column): str(dtype)
            for column, dtype in sample_input.dtypes.items()
        }
        schema_source = "export_sample_columns"
    else:
        pipeline_feature_names = getattr(
            pipeline,
            "feature_names_in_",
            [],
        )
        raw_feature_names = [
            str(column)
            for column in pipeline_feature_names
        ]
        pandas_dtypes = {}
        schema_source = "fitted_pipeline"

    preprocessing_report = report.get(
        "preprocessing_report",
        {},
    )
    preprocessing_metadata = preprocessing_report.get(
        "preprocessing_metadata",
        {},
    )
    numerical_features = [
        str(column)
        for column in preprocessing_metadata.get(
            "numerical_features",
            [],
        )
    ]
    categorical_features = [
        str(column)
        for column in preprocessing_metadata.get(
            "categorical_features",
            [],
        )
    ]
    required_feature_set = set(
        numerical_features + categorical_features
    )
    required_features = [
        column
        for column in raw_feature_names
        if column in required_feature_set
    ]
    optional_features = [
        column
        for column in raw_feature_names
        if column not in required_feature_set
    ]

    return {
        "schema_version": "1.0",
        "schema_source": schema_source,
        "raw_feature_count": len(raw_feature_names),
        "raw_feature_names": raw_feature_names,
        "required_features": required_features,
        "optional_features": optional_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "pandas_dtypes": pandas_dtypes,
        "sample_values_stored": False,
    }


def load_model_artifact(
    model_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Pipeline:
    """
    Load a trusted pipeline artifact after optional hash validation.

    Joblib uses pickle internally. Never load a model artifact from an
    untrusted or user-controlled source.
    """

    resolved_path = Path(model_path).resolve()

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError("Model artifact was not found.")

    if resolved_path.suffix.lower() != ".joblib":
        raise ValueError("Model artifact must use the .joblib extension.")

    if expected_sha256 is not None:
        actual_sha256 = _sha256(resolved_path)

        if actual_sha256 != expected_sha256:
            raise ValueError("Model artifact SHA-256 verification failed.")

    pipeline = _load_joblib(resolved_path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "Model artifact does not contain a scikit-learn Pipeline."
        )

    return pipeline


def export_model_artifacts(
    training_result: dict[str, Any],
    output_directory: str | Path,
    *,
    sample_input: pd.DataFrame | None = None,
    supplementary_artifacts: dict[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically export a fitted pipeline and its JSON metadata."""

    if not isinstance(training_result, dict):
        raise TypeError("training_result must be a dictionary.")

    pipeline = training_result.get("pipeline")
    report = training_result.get("report")

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "training_result does not contain a fitted Pipeline."
        )

    if not isinstance(report, dict):
        raise TypeError(
            "training_result does not contain a report dictionary."
        )

    if sample_input is not None and not isinstance(
        sample_input,
        pd.DataFrame,
    ):
        raise TypeError("sample_input must be a Pandas DataFrame.")

    output_path = Path(output_directory).resolve()

    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_directory exists but is not a directory.")

    output_path.mkdir(parents=True, exist_ok=True)

    supplementary_paths: dict[str, Path] = {}

    for artifact_type, artifact_path in (
        supplementary_artifacts or {}
    ).items():
        if (
            not isinstance(artifact_type, str)
            or re.fullmatch(
                r"[a-z][a-z0-9_]*",
                artifact_type,
            )
            is None
            or artifact_type
            in {"pipeline", "training_report", "manifest"}
        ):
            raise ValueError(
                "Supplementary artifact type is invalid."
            )

        resolved_artifact = Path(artifact_path).resolve()

        if (
            resolved_artifact.parent != output_path
            or not resolved_artifact.exists()
            or not resolved_artifact.is_file()
        ):
            raise ValueError(
                "Supplementary artifact must be an existing file "
                "inside output_directory."
            )

        if resolved_artifact.name in {
            PIPELINE_FILENAME,
            REPORT_FILENAME,
            MANIFEST_FILENAME,
        }:
            raise ValueError(
                "Supplementary artifact uses a reserved filename."
            )

        supplementary_paths[artifact_type] = resolved_artifact

    pipeline_path = output_path / PIPELINE_FILENAME
    report_path = output_path / REPORT_FILENAME
    manifest_path = output_path / MANIFEST_FILENAME
    final_paths = [pipeline_path, report_path, manifest_path]

    existing_paths = [path for path in final_paths if path.exists()]

    if existing_paths and not overwrite:
        existing_names = [path.name for path in existing_paths]
        raise FileExistsError(
            "Artifact files already exist: "
            f"{existing_names}. Set overwrite=True to replace them."
        )

    pipeline_temporary = _temporary_path(
        output_path,
        ".model-pipeline-",
    )
    report_temporary = _temporary_path(
        output_path,
        ".training-report-",
    )
    manifest_temporary: Path | None = None

    try:
        joblib.dump(
            pipeline,
            pipeline_temporary,
            compress=3,
        )
        _write_json(report_temporary, report)

        reloaded_pipeline = _load_joblib(pipeline_temporary)

        if not isinstance(reloaded_pipeline, Pipeline):
            raise TypeError(
                "Reloaded artifact is not a scikit-learn Pipeline."
            )

        prediction_equivalence_verified = False
        verification_rows = 0

        if sample_input is not None:
            expected_predictions = pipeline.predict(sample_input)
            actual_predictions = reloaded_pipeline.predict(
                sample_input
            )

            if not _predictions_match(
                expected_predictions,
                actual_predictions,
            ):
                raise RuntimeError(
                    "Reloaded pipeline predictions do not match "
                    "the original pipeline."
                )

            prediction_equivalence_verified = True
            verification_rows = len(sample_input)

        os.replace(pipeline_temporary, pipeline_path)
        os.replace(report_temporary, report_path)

        pipeline_hash = _sha256(pipeline_path)
        report_hash = _sha256(report_path)
        best_model = report.get("best_model", {})
        feature_selection = report.get("feature_selection", {})
        input_schema = _build_input_schema(
            pipeline,
            report,
            sample_input,
        )

        manifest_files = {
            "pipeline": {
                "file_name": PIPELINE_FILENAME,
                "sha256": pipeline_hash,
                "size_bytes": pipeline_path.stat().st_size,
            },
            "training_report": {
                "file_name": REPORT_FILENAME,
                "sha256": report_hash,
                "size_bytes": report_path.stat().st_size,
            },
        }

        for artifact_type, artifact_path in sorted(
            supplementary_paths.items()
        ):
            manifest_files[artifact_type] = {
                "file_name": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            }

        manifest = {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "trusted_source_required": True,
            "target_column": report.get("target_column"),
            "problem_type": report.get("problem_type"),
            "model_name": best_model.get("model_name"),
            "model_display_name": best_model.get(
                "display_name"
            ),
            "selected_variant": best_model.get(
                "selected_variant"
            ),
            "selected_feature_count": feature_selection.get(
                "selected_feature_count"
            ),
            "selected_feature_names": feature_selection.get(
                "selected_feature_names",
                [],
            ),
            "input_schema": input_schema,
            "prediction_equivalence_verified": (
                prediction_equivalence_verified
            ),
            "verification_rows": verification_rows,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": version("scikit-learn"),
                "pandas": version("pandas"),
                "numpy": version("numpy"),
                "joblib": version("joblib"),
            },
            "files": manifest_files,
        }

        manifest_temporary = _temporary_path(
            output_path,
            ".artifact-manifest-",
        )
        _write_json(manifest_temporary, manifest)
        os.replace(manifest_temporary, manifest_path)

    finally:
        pipeline_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)

        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)

    return {
        "artifact_status": "complete",
        "output_directory": str(output_path),
        "pipeline_path": str(pipeline_path),
        "training_report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def verify_artifact_bundle(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify bundle paths, sizes, hashes, and trusted pipeline load."""

    resolved_manifest_path = Path(manifest_path).resolve()

    if (
        not resolved_manifest_path.exists()
        or not resolved_manifest_path.is_file()
    ):
        raise FileNotFoundError("Artifact manifest was not found.")

    if resolved_manifest_path.name != MANIFEST_FILENAME:
        raise ValueError(
            f"Artifact manifest must be named {MANIFEST_FILENAME}."
        )

    with resolved_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        manifest = json.load(file_handle)

    if (
        manifest.get("artifact_format_version")
        != ARTIFACT_FORMAT_VERSION
    ):
        raise ValueError("Unsupported artifact format version.")

    files = manifest.get("files")

    if not isinstance(files, dict):
        raise ValueError("Artifact manifest does not contain files.")

    required_entries = {"pipeline", "training_report"}

    if not required_entries.issubset(files):
        raise ValueError(
            "Artifact manifest is missing required file entries."
        )

    directory = resolved_manifest_path.parent
    verified_files = {}

    for entry_name in sorted(files):
        entry = files[entry_name]

        if not isinstance(entry, dict):
            raise ValueError("Artifact file entry is invalid.")

        file_path = _safe_bundle_file(
            directory,
            entry.get("file_name"),
        )

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(
                f"Artifact file was not found: {file_path.name}"
            )

        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")

        if file_path.stat().st_size != expected_size:
            raise ValueError(
                f"Artifact size verification failed: {file_path.name}"
            )

        actual_hash = _sha256(file_path)

        if actual_hash != expected_hash:
            raise ValueError(
                f"Artifact SHA-256 verification failed: "
                f"{file_path.name}"
            )

        verified_files[entry_name] = {
            "file_name": file_path.name,
            "sha256": actual_hash,
            "size_bytes": file_path.stat().st_size,
        }

    pipeline_entry = files["pipeline"]
    pipeline_path = _safe_bundle_file(
        directory,
        pipeline_entry["file_name"],
    )
    load_model_artifact(
        pipeline_path,
        expected_sha256=pipeline_entry["sha256"],
    )

    return {
        "verification_status": "complete",
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "verified_file_count": len(verified_files),
        "pipeline_load_verified": True,
        "files": verified_files,
    }
