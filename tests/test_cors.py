import pytest
from fastapi.testclient import TestClient

from backend.app.main import app, parse_cors_origins


client = TestClient(app)


def test_default_cors_origins_cover_local_vite():
    assert parse_cors_origins(None) == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_cors_origin_parser_deduplicates_and_normalizes():
    assert parse_cors_origins(
        "https://example.com/, https://example.com, http://localhost:4173"
    ) == [
        "https://example.com",
        "http://localhost:4173",
    ]


@pytest.mark.parametrize(
    "origins",
    ["*", "localhost:5173", ",".join(f"https://{i}.test" for i in range(21))],
)
def test_unsafe_cors_configuration_is_rejected(origins):
    with pytest.raises(ValueError):
        parse_cors_origins(origins)


def test_vite_preflight_is_accepted():
    response = client.options(
        "/upload",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://127.0.0.1:5173"
    )
    assert "POST" in response.headers["access-control-allow-methods"]


def test_unlisted_origin_does_not_receive_cors_permission():
    response = client.options(
        "/upload",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers
