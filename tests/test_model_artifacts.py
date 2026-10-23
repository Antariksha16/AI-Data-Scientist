import json
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.model_artifacts import (
    MANIFEST_FILENAME,
    PIPELINE_FILENAME,
    REPORT_FILENAME,
    export_model_artifacts,
    load_model_artifact,
    verify_artifact_bundle,
)
from ml_pipeline.training import train_and_compare_models
from tests.test_training import (
    _classification_dataframe,
    _fast_config,
    _regression_dataframe,
)


@pytest.fixture(scope="module")
def classification_training_result():
    return train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
        requested_models=["logistic_regression"],
    )


@pytest.fixture(scope="module")
def regression_training_result():
    return train_and_compare_models(
        _regression_dataframe(),
        "Salary",
        config=_fast_config(),
        requested_models=["ridge"],
    )


def _read_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _write_json(file_path: Path, value: dict) -> None:
    with file_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            value,
            file_handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


def test_classification_export_reload_and_prediction_equivalence(
    tmp_path,
    classification_training_result,
):
    sample_input = _classification_dataframe().drop(
        columns=["Target"]
    ).head(12)
    export = export_model_artifacts(
        classification_training_result,
        tmp_path / "classification_bundle",
        sample_input=sample_input,
    )

    pipeline_path = Path(export["pipeline_path"])
    report_path = Path(export["training_report_path"])
    manifest_path = Path(export["manifest_path"])

    assert pipeline_path.name == PIPELINE_FILENAME
    assert report_path.name == REPORT_FILENAME
    assert manifest_path.name == MANIFEST_FILENAME
    assert pipeline_path.is_file()
    assert report_path.is_file()
    assert manifest_path.is_file()
    assert export["manifest"][
        "prediction_equivalence_verified"
    ] is True
    assert export["manifest"]["verification_rows"] == 12

    expected_predictions = classification_training_result[
        "pipeline"
    ].predict(sample_input)
    pipeline_hash = export["manifest"]["files"][
        "pipeline"
    ]["sha256"]
    loaded_pipeline = load_model_artifact(
        pipeline_path,
        expected_sha256=pipeline_hash,
    )
    actual_predictions = loaded_pipeline.predict(sample_input)

    assert np.array_equal(
        expected_predictions,
        actual_predictions,
    )
    json.dumps(_read_json(report_path), allow_nan=False)
    json.dumps(_read_json(manifest_path), allow_nan=False)


def test_regression_export_and_bundle_verification(
    tmp_path,
    regression_training_result,
):
    sample_input = _regression_dataframe().drop(
        columns=["Salary"]
    ).head(10)
    export = export_model_artifacts(
        regression_training_result,
        tmp_path / "regression_bundle",
        sample_input=sample_input,
    )
    verification = verify_artifact_bundle(
        export["manifest_path"]
    )

    assert verification["verification_status"] == "complete"
    assert verification["verified_file_count"] == 2
    assert verification["pipeline_load_verified"] is True
    assert export["manifest"]["problem_type"] == "regression"
    assert export["manifest"]["target_column"] == "Salary"


def test_existing_artifacts_require_explicit_overwrite(
    tmp_path,
    classification_training_result,
):
    output_directory = tmp_path / "protected_bundle"
    export_model_artifacts(
        classification_training_result,
        output_directory,
    )

    with pytest.raises(FileExistsError, match="already exist"):
        export_model_artifacts(
            classification_training_result,
            output_directory,
        )

    unrelated_file = output_directory / "keep_me.txt"
    unrelated_file.write_text("preserve", encoding="utf-8")

    export_model_artifacts(
        classification_training_result,
        output_directory,
        overwrite=True,
    )

    assert unrelated_file.read_text(encoding="utf-8") == "preserve"


def test_pipeline_corruption_is_detected(
    tmp_path,
    classification_training_result,
):
    export = export_model_artifacts(
        classification_training_result,
        tmp_path / "corrupt_bundle",
    )
    pipeline_path = Path(export["pipeline_path"])

    corrupted_payload = bytearray(
        pipeline_path.read_bytes()
    )
    corrupted_payload[
        len(corrupted_payload) // 2
    ] ^= 0x01

    pipeline_path.write_bytes(corrupted_payload)

    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact_bundle(export["manifest_path"])


def test_manifest_path_traversal_is_rejected(
    tmp_path,
    classification_training_result,
):
    export = export_model_artifacts(
        classification_training_result,
        tmp_path / "unsafe_manifest_bundle",
    )
    manifest_path = Path(export["manifest_path"])
    manifest = _read_json(manifest_path)
    manifest["files"]["pipeline"]["file_name"] = (
        "../model_pipeline.joblib"
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unsafe filename"):
        verify_artifact_bundle(manifest_path)


def test_incorrect_expected_hash_blocks_model_load(
    tmp_path,
    classification_training_result,
):
    export = export_model_artifacts(
        classification_training_result,
        tmp_path / "hash_bundle",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_model_artifact(
            export["pipeline_path"],
            expected_sha256="0" * 64,
        )


def test_export_without_sample_still_verifies_pipeline_load(
    tmp_path,
    classification_training_result,
):
    export = export_model_artifacts(
        classification_training_result,
        tmp_path / "no_sample_bundle",
    )

    assert export["manifest"][
        "prediction_equivalence_verified"
    ] is False
    assert export["manifest"]["verification_rows"] == 0

    verification = verify_artifact_bundle(
        export["manifest_path"]
    )
    assert verification["pipeline_load_verified"] is True


def test_invalid_training_result_is_rejected(tmp_path):
    with pytest.raises(TypeError, match="fitted Pipeline"):
        export_model_artifacts(
            {"pipeline": None, "report": {}},
            tmp_path / "invalid_bundle",
        )
