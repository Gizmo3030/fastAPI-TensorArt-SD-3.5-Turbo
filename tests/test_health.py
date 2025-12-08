import os

from fastapi.testclient import TestClient

os.environ.setdefault("TORCH_DEVICE", "cpu")

from app.main import app  # noqa: E402


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
