import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.analysis import router as analysis_router
from backend.app.api.upload import router as upload_router


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def parse_cors_origins(raw_origins: str | None) -> list[str]:
    """Return a bounded explicit browser-origin allowlist."""

    if raw_origins is None or not raw_origins.strip():
        return list(DEFAULT_CORS_ORIGINS)

    origins = list(
        dict.fromkeys(
            origin.strip().rstrip("/")
            for origin in raw_origins.split(",")
            if origin.strip()
        )
    )

    if not origins:
        return list(DEFAULT_CORS_ORIGINS)

    if "*" in origins:
        raise ValueError(
            "Wildcard CORS origins are not allowed."
        )

    if len(origins) > 20:
        raise ValueError(
            "At most 20 CORS origins may be configured."
        )

    if any(
        not origin.startswith(("http://", "https://"))
        for origin in origins
    ):
        raise ValueError(
            "CORS origins must use http:// or https://."
        )

    return origins


app = FastAPI(
    title="AI Data Scientist",
    description=(
        "Transparent end-to-end machine learning automation for "
        "tabular classification and regression datasets."
    ),
    version="0.10.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(
        os.getenv("AI_DATA_SCIENTIST_CORS_ORIGINS")
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)

app.include_router(upload_router)
app.include_router(analysis_router)


@app.get("/", tags=["System"])
def home():
    return {"message": "AI Data Scientist API is running"}


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy"}
