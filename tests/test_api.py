from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_health_probe():
    """Verify that GET / returns HTTP 200 and valid health status payload."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"].lower() in ["online", "degraded"]
    assert "checks" in data


def test_invalid_route_404():
    """Verify unhandled paths return standard HTTP 404."""
    response = client.get("/api/unknown/endpoint")
    assert response.status_code == 404
