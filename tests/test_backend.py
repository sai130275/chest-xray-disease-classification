"""
Unit and integration tests for RAD-X FastAPI backend and predictor.
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app
from backend.predictor import get_predictor
from ml.config import DISEASE_CLASSES

def create_synthetic_image_bytes(format="PNG") -> bytes:
    """Helper to create an in-memory sample RGB image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_predictor_singleton_and_inference():
    predictor = get_predictor()
    assert predictor.is_ready is True
    assert len(predictor.classes) == 15

    img_bytes = create_synthetic_image_bytes()
    result = predictor.predict(img_bytes)

    assert "predictions" in result
    assert "positive_findings" in result
    assert "is_no_finding" in result
    assert "model_info" in result
    assert "disclaimer" in result

    assert len(result["predictions"]) == 15
    for item in result["predictions"]:
        assert item["disease"] in DISEASE_CLASSES
        assert 0.0 <= item["probability"] <= 1.0
        assert 0.0 <= item["threshold"] <= 1.0
        assert isinstance(item["is_positive"], bool)

def test_api_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["classes_count"] == 15

def test_api_metadata_endpoint():
    with TestClient(app) as client:
        response = client.get("/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["architecture"] == "DenseNet121"
        assert data["num_classes"] == 15
        assert len(data["classes"]) == 15
        assert len(data["thresholds"]) == 15

def test_api_predict_success():
    img_bytes = create_synthetic_image_bytes(format="PNG")
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test_xray.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 15
        assert "RESEARCH PROTOTYPE ONLY" in data["disclaimer"]
        assert "architecture" in data["model_info"]

def test_api_predict_invalid_file_type():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"Invalid text file data", "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

def test_api_predict_corrupt_image():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\ncorruptdata", "image/png")}
        )
        assert response.status_code == 400
        assert "Invalid or corrupted" in response.json()["detail"]

if __name__ == "__main__":
    test_predictor_singleton_and_inference()
    test_api_health_endpoint()
    test_api_metadata_endpoint()
    test_api_predict_success()
    test_api_predict_invalid_file_type()
    test_api_predict_corrupt_image()
    print("test_backend.py: ALL TESTS PASSED.")
