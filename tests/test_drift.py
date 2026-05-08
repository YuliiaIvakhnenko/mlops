"""Тести DriftDetector та endpoint `/check-drift`."""

import numpy as np
from fastapi.testclient import TestClient

from app.drift import DriftDetector
from app.main import MODEL_PATH, REFERENCE_PATH, app
from ml.train import train_and_save

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)

client = TestClient(app)


def test_no_drift_on_same_distribution() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    current = rng.normal(loc=5.0, scale=1.0, size=(500, 4))

    detector = DriftDetector(reference, FEATURE_NAMES)
    result = detector.detect(current, alpha=0.001)

    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0


def test_drift_on_shifted_distribution() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=5.0, scale=1.0, size=(500, 4))
    current = rng.normal(loc=8.0, scale=1.0, size=(500, 4))

    detector = DriftDetector(reference, FEATURE_NAMES)
    result = detector.detect(current, alpha=0.05)

    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4

    for feature in FEATURE_NAMES:
        assert result["per_feature"][feature]["p_value"] < 0.05


def test_check_drift_endpoint_no_drift() -> None:
    samples = [[5.1, 3.5, 1.4, 0.2] for _ in range(10)]

    response = client.post("/check-drift", json={"samples": samples, "alpha": 0.05})

    assert response.status_code == 200
    body = response.json()
    assert body["n_samples"] == 10
    assert "per_feature" in body


def test_check_drift_endpoint_detects_shifted_data() -> None:
    samples = [[9.0, 9.0, 9.0, 9.0] for _ in range(20)]

    response = client.post("/check-drift", json={"samples": samples, "alpha": 0.05})

    assert response.status_code == 200
    body = response.json()
    assert body["drift_detected"] is True
    assert body["n_drifted_features"] >= 1
