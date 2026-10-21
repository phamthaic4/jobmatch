"""API tests — the end-to-end ones need the checkpoint in models/phobert-ner.

    python -m pytest            # everything (downloads the weights on first run)
    python -m pytest -m "not slow"   # only the tests that need no weights
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

MODEL_DIR = ROOT / "models" / "phobert-ner"

JD_TEXT = (
    "Ứng tuyển Backend Developer. Yêu cầu 3 năm kinh nghiệm với Python, FastAPI, "
    "PostgreSQL, Docker và Kubernetes. Có kinh nghiệm với CI/CD, Redis, Elasticsearch."
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_reports_model_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "backend" in body


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/extract-skills" in response.json()["paths"]


def test_empty_text_is_rejected(client):
    response = client.post("/extract-skills", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.slow
def test_extract_skills_from_a_job_description(client):
    response = client.post("/extract-skills", json={"text": JD_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["skills"])
    assert "python" in body["skills"]
    assert "postgresql" in body["skills"]
    assert "docker" in body["skills"]
    assert body["latency_ms"] > 0


@pytest.mark.slow
def test_spans_point_at_the_skill_in_the_source_text(client):
    response = client.post("/extract-skills", json={"text": JD_TEXT})
    body = response.json()
    for span in body["spans"]:
        assert JD_TEXT[span["start"] : span["end"]].lower().startswith(
            span["skill"].split()[0].lower()
        )


@pytest.mark.slow
def test_normalize_flag_controls_casing(client):
    raw = client.post("/extract-skills", json={"text": "Kinh nghiệm K8S, ReactJS.", "normalize": False})
    normalized = client.post(
        "/extract-skills", json={"text": "Kinh nghiệm K8S, ReactJS.", "normalize": True}
    )
    assert "K8S" in raw.json()["skills"]
    assert "kubernetes" in normalized.json()["skills"]
