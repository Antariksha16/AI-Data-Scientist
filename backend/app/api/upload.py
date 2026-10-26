import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core import dataset_files
from ml_pipeline.analyzer import analyze_dataset
from ml_pipeline.target_suggester import suggest_target_columns


router = APIRouter(tags=["Datasets"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _save_to_temporary_file(
    uploaded_file: UploadFile,
    extension: str,
) -> Path:
    """Stream an upload into application storage with a size limit."""

    dataset_files.ensure_storage_directories()
    temporary_path = (
        dataset_files.UPLOAD_DIR
        / f".{uuid4().hex}{extension}"
    )
    bytes_written = 0

    try:
        with temporary_path.open("wb") as buffer:
            while True:
                chunk = await uploaded_file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Dataset exceeds the 100 MB upload limit.",
                    )

                buffer.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await uploaded_file.close()

    return temporary_path


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Validate, inspect, and atomically publish a tabular dataset."""

    try:
        safe_filename = dataset_files.validate_dataset_filename(
            file.filename or ""
        )
    except dataset_files.DatasetFilenameError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    extension = Path(safe_filename).suffix.lower()
    temporary_path = await _save_to_temporary_file(file, extension)

    try:
        df = dataset_files.read_dataset(temporary_path)
        preview_json = json.loads(
            df.head(5).to_json(
                orient="records",
                date_format="iso",
            )
        )
        dataset_analysis = analyze_dataset(df)
        target_suggestions = suggest_target_columns(
            df,
            top_k=min(10, len(df.columns)),
        )
    except dataset_files.DatasetReadError as error:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Unable to analyze dataset: {error}",
        ) from error

    destination = dataset_files.UPLOAD_DIR / safe_filename

    try:
        dataset_files.replace_file_atomically(
            temporary_path,
            destination,
        )
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to store the uploaded dataset.",
        ) from error

    return {
        "filename": safe_filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "preview": preview_json,
        "dataset_analysis": dataset_analysis,
        "target_suggestions": target_suggestions,
    }