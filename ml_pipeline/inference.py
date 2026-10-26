import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_pipeline.model_artifacts import (
    MANIFEST_FILENAME,
    load_model_artifact,
    verify_artifact_bundle,
)


DEFAULT_MAX_PREDICTION_ROWS = 1_000
SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}


class InferenceInputError(ValueError):
    """Raised when prediction records do not match the model schema."""


class InferenceArtifactError(RuntimeError):
    """Raised when a stored model bundle cannot be trusted or loaded."""


def _read_json_dictionary(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as file_handle:
            value = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceArtifactError(
            f"Unable to read verified artifact metadata: "
            f"{file_path.name}"
        ) from error

    if not isinstance(value, dict):
        raise InferenceArtifactError(
            f"Artifact metadata must be a JSON object: "
            f"{file_path.name}"
        )

    return value


def _bundle_file_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    entry_name: str,
) -> Path:
    files = manifest.get("files")

    if not isinstance(files, dict):
        raise InferenceArtifactError(
            "Artifact manifest does not contain files."
        )

    entry = files.get(entry_name)

    if not isinstance(entry, dict):
        raise InferenceArtifactError(
            f"Artifact manifest is missing '{entry_name}'."
        )

    file_name = entry.get("file_name")

    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
    ):
        raise InferenceArtifactError(
            "Artifact manifest contains an unsafe filename."
        )

    bundle_directory = manifest_path.parent.resolve()
    file_path = (bundle_directory / file_name).resolve()

    if file_path.parent != bundle_directory:
        raise InferenceArtifactError(
            "Artifact file escapes its bundle directory."
        )

    return file_path


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item
        for item in value
    ):
        raise InferenceArtifactError(
            f"Input schema field '{field_name}' is invalid."
        )

    if len(set(value)) != len(value):
        raise InferenceArtifactError(
            f"Input schema field '{field_name}' contains duplicates."
        )

    return list(value)


def _fallback_input_schema(
    pipeline: Pipeline,
    report: dict[str, Any],
) -> dict[str, Any]:
    pipeline_features = getattr(
        pipeline,
        "feature_names_in_",
        [],
    )
    raw_feature_names = [
        str(column)
        for column in pipeline_features
    ]

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

    if not raw_feature_names:
        raw_feature_names = list(
            dict.fromkeys(
                numerical_features + categorical_features
            )
        )

    required_set = set(
        numerical_features + categorical_features
    )
    required_features = [
        column
        for column in raw_feature_names
        if column in required_set
    ]

    return {
        "schema_version": "1.0",
        "schema_source": "legacy_bundle_fallback",
        "raw_feature_count": len(raw_feature_names),
        "raw_feature_names": raw_feature_names,
        "required_features": required_features,
        "optional_features": [
            column
            for column in raw_feature_names
            if column not in required_set
        ],
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "pandas_dtypes": {},
        "sample_values_stored": False,
    }


def _validated_input_schema(
    manifest: dict[str, Any],
    pipeline: Pipeline,
    report: dict[str, Any],
) -> dict[str, Any]:
    schema = manifest.get("input_schema")

    if schema is None:
        schema = _fallback_input_schema(pipeline, report)

    if not isinstance(schema, dict):
        raise InferenceArtifactError(
            "Artifact input schema must be a dictionary."
        )

    raw_features = _string_list(
        schema.get("raw_feature_names"),
        "raw_feature_names",
    )
    required_features = _string_list(
        schema.get("required_features"),
        "required_features",
    )
    optional_features = _string_list(
        schema.get("optional_features"),
        "optional_features",
    )
    numerical_features = _string_list(
        schema.get("numerical_features"),
        "numerical_features",
    )
    categorical_features = _string_list(
        schema.get("categorical_features"),
        "categorical_features",
    )

    raw_set = set(raw_features)
    required_set = set(required_features)
    optional_set = set(optional_features)
    typed_set = set(
        numerical_features + categorical_features
    )

    if not raw_features or not required_features:
        raise InferenceArtifactError(
            "Artifact input schema contains no usable features."
        )

    if (
        not required_set.issubset(raw_set)
        or not optional_set.issubset(raw_set)
        or required_set & optional_set
        or required_set | optional_set != raw_set
        or not typed_set.issubset(required_set)
        or set(numerical_features) & set(categorical_features)
    ):
        raise InferenceArtifactError(
            "Artifact input schema has inconsistent feature roles."
        )

    pandas_dtypes = schema.get("pandas_dtypes", {})

    if not isinstance(pandas_dtypes, dict):
        raise InferenceArtifactError(
            "Artifact input schema has invalid dtype metadata."
        )

    return {
        "schema_version": str(
            schema.get("schema_version", "1.0")
        ),
        "schema_source": str(
            schema.get("schema_source", "artifact_manifest")
        ),
        "raw_feature_count": len(raw_features),
        "raw_feature_names": raw_features,
        "required_features": required_features,
        "optional_features": optional_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "pandas_dtypes": {
            str(column): str(dtype)
            for column, dtype in pandas_dtypes.items()
            if column in raw_set
        },
        "sample_values_stored": False,
    }


def load_verified_inference_bundle(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify and load one application-owned model bundle."""

    resolved_manifest_path = Path(manifest_path).resolve()

    try:
        verification = verify_artifact_bundle(
            resolved_manifest_path
        )
        manifest = _read_json_dictionary(
            resolved_manifest_path
        )
        report_path = _bundle_file_path(
            resolved_manifest_path,
            manifest,
            "training_report",
        )
        pipeline_path = _bundle_file_path(
            resolved_manifest_path,
            manifest,
            "pipeline",
        )
        report = _read_json_dictionary(report_path)
        pipeline_entry = manifest["files"]["pipeline"]
        pipeline = load_model_artifact(
            pipeline_path,
            expected_sha256=pipeline_entry.get("sha256"),
        )
        input_schema = _validated_input_schema(
            manifest,
            pipeline,
            report,
        )
    except InferenceArtifactError:
        raise
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise InferenceArtifactError(
            f"Model bundle verification failed: {error}"
        ) from error

    problem_type = manifest.get("problem_type")

    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise InferenceArtifactError(
            "Artifact problem type is unsupported."
        )

    target_column = manifest.get("target_column")

    if not isinstance(target_column, str) or not target_column:
        raise InferenceArtifactError(
            "Artifact target column is invalid."
        )

    return {
        "pipeline": pipeline,
        "manifest": manifest,
        "report": report,
        "input_schema": input_schema,
        "verification": verification,
    }


def describe_model_input(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Return public input requirements for a verified model."""

    bundle = load_verified_inference_bundle(manifest_path)
    manifest = bundle["manifest"]

    return {
        "schema_status": "complete",
        "target_column": manifest["target_column"],
        "problem_type": manifest["problem_type"],
        "model_name": manifest.get("model_name"),
        "model_display_name": manifest.get(
            "model_display_name"
        ),
        "selected_variant": manifest.get("selected_variant"),
        "input_schema": bundle["input_schema"],
        "artifact_verification": {
            "verification_status": bundle["verification"][
                "verification_status"
            ],
            "pipeline_load_verified": bundle["verification"][
                "pipeline_load_verified"
            ],
        },
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return str(value)


def _validate_json_scalar(
    value: Any,
    *,
    row_index: int,
    feature_name: str,
) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise InferenceInputError(
                f"Row {row_index} feature '{feature_name}' "
                "must contain a finite JSON value or null."
            )
        return value

    raise InferenceInputError(
        f"Row {row_index} feature '{feature_name}' must be a "
        "string, number, boolean, or null."
    )


def _prepare_prediction_frame(
    records: list[dict[str, Any]],
    input_schema: dict[str, Any],
    *,
    max_records: int,
) -> pd.DataFrame:
    if not isinstance(records, list):
        raise InferenceInputError("records must be a list.")

    if not records:
        raise InferenceInputError(
            "At least one prediction record is required."
        )

    if len(records) > max_records:
        raise InferenceInputError(
            f"Prediction requests are limited to {max_records} rows."
        )

    raw_features = input_schema["raw_feature_names"]
    raw_feature_set = set(raw_features)
    required_features = set(
        input_schema["required_features"]
    )
    normalized_records = []

    for row_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise InferenceInputError(
                f"Prediction record {row_index} must be an object."
            )

        if any(
            not isinstance(feature_name, str)
            for feature_name in record
        ):
            raise InferenceInputError(
                f"Prediction record {row_index} contains a "
                "non-string feature name."
            )

        record_features = set(record)
        missing_features = sorted(
            required_features - record_features
        )
        unknown_features = sorted(
            record_features - raw_feature_set
        )

        if missing_features:
            raise InferenceInputError(
                f"Prediction record {row_index} is missing required "
                f"features: {missing_features}"
            )

        if unknown_features:
            raise InferenceInputError(
                f"Prediction record {row_index} contains unknown "
                f"features: {unknown_features}"
            )

        normalized_record = {}

        for feature_name in raw_features:
            value = record.get(feature_name)
            normalized_record[feature_name] = (
                _validate_json_scalar(
                    value,
                    row_index=row_index,
                    feature_name=feature_name,
                )
            )

        normalized_records.append(normalized_record)

    prediction_frame = pd.DataFrame(
        normalized_records,
        columns=raw_features,
    )

    for feature_name in input_schema["numerical_features"]:
        original = prediction_frame[feature_name]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_mask = original.notna() & converted.isna()

        if bool(invalid_mask.any()):
            invalid_rows = [
                int(index)
                for index in invalid_mask[invalid_mask].index
            ]
            raise InferenceInputError(
                f"Feature '{feature_name}' must be numeric or null; "
                f"invalid rows: {invalid_rows}"
            )

        prediction_frame[feature_name] = converted

    return prediction_frame


def predict_with_model_artifact(
    manifest_path: str | Path,
    records: list[dict[str, Any]],
    *,
    include_probabilities: bool = True,
    max_records: int = DEFAULT_MAX_PREDICTION_ROWS,
) -> dict[str, Any]:
    """Predict from raw JSON-safe rows using a verified pipeline."""

    if (
        not isinstance(max_records, int)
        or isinstance(max_records, bool)
        or max_records < 1
    ):
        raise ValueError("max_records must be a positive integer.")

    if not isinstance(include_probabilities, bool):
        raise TypeError("include_probabilities must be a boolean.")

    if not isinstance(records, list):
        raise InferenceInputError("records must be a list.")

    if not records:
        raise InferenceInputError(
            "At least one prediction record is required."
        )

    if len(records) > max_records:
        raise InferenceInputError(
            f"Prediction requests are limited to {max_records} rows."
        )

    bundle = load_verified_inference_bundle(manifest_path)
    pipeline = bundle["pipeline"]
    manifest = bundle["manifest"]
    input_schema = bundle["input_schema"]
    prediction_frame = _prepare_prediction_frame(
        records,
        input_schema,
        max_records=max_records,
    )

    try:
        raw_predictions = pipeline.predict(prediction_frame)
    except Exception as error:
        raise InferenceInputError(
            "Input records are incompatible with the fitted model."
        ) from error

    problem_type = manifest["problem_type"]
    probabilities = None
    class_labels = []

    if problem_type == "regression":
        probability_status = "not_applicable"
    elif not include_probabilities:
        probability_status = "disabled"
    elif not hasattr(pipeline, "predict_proba"):
        probability_status = "unavailable"
    else:
        try:
            probability_array = np.asarray(
                pipeline.predict_proba(prediction_frame),
                dtype=float,
            )
        except Exception as error:
            raise InferenceInputError(
                "The fitted model could not calculate probabilities."
            ) from error

        raw_class_labels = getattr(
            pipeline,
            "classes_",
            None,
        )

        if raw_class_labels is None:
            estimator = pipeline.named_steps.get("model")
            raw_class_labels = getattr(
                estimator,
                "classes_",
                None,
            )

        if raw_class_labels is None:
            raise InferenceArtifactError(
                "Classifier probabilities do not expose class labels."
            )

        class_labels = [
            _json_scalar(label)
            for label in raw_class_labels
        ]

        if (
            probability_array.ndim != 2
            or probability_array.shape[0] != len(records)
            or probability_array.shape[1] != len(class_labels)
            or not np.isfinite(probability_array).all()
        ):
            raise InferenceArtifactError(
                "Classifier returned invalid probability output."
            )

        probabilities = probability_array
        probability_status = "complete"

    prediction_rows = []

    for row_index, prediction in enumerate(raw_predictions):
        row_result = {
            "row_index": row_index,
            "prediction": _json_scalar(prediction),
        }

        if probabilities is not None:
            row_probabilities = [
                {
                    "class_label": class_label,
                    "probability": float(probability),
                }
                for class_label, probability in zip(
                    class_labels,
                    probabilities[row_index],
                    strict=True,
                )
            ]
            row_result["probabilities"] = row_probabilities
            row_result["confidence"] = max(
                item["probability"]
                for item in row_probabilities
            )

        prediction_rows.append(row_result)

    return {
        "prediction_status": "complete",
        "problem_type": problem_type,
        "target_column": manifest["target_column"],
        "model_name": manifest.get("model_name"),
        "model_display_name": manifest.get(
            "model_display_name"
        ),
        "selected_variant": manifest.get("selected_variant"),
        "prediction_count": len(prediction_rows),
        "probability_status": probability_status,
        "class_labels": class_labels,
        "predictions": prediction_rows,
        "input_schema": input_schema,
        "artifact_verification": {
            "verification_status": bundle["verification"][
                "verification_status"
            ],
            "pipeline_load_verified": bundle["verification"][
                "pipeline_load_verified"
            ],
        },
        "source_records_modified": False,
    }
