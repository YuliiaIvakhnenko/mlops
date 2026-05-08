"""FastAPI-сервіс для інференсу Iris із Prometheus monitoring та drift detection."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .drift import DriftDetector
from .logging_config import setup_logging
from .metrics import (
    DRIFT_CHECKS,
    DRIFT_DETECTED,
    ERROR_COUNTER,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    REGISTRY,
)
from .schemas import DriftRequest, DriftResponse, IrisFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_stats.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

setup_logging()
logger = logging.getLogger("ml-api")

model: Any | None = None
drift_detector: DriftDetector | None = None


def initialize_artifacts() -> None:
    """Завантажує модель та reference-статистики."""
    global model, drift_detector

    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)

    if REFERENCE_PATH.exists():
        ref_data = joblib.load(REFERENCE_PATH)
        drift_detector = DriftDetector(
            reference=ref_data["X"],
            feature_names=ref_data["feature_names"],
        )
        logger.info(
            "startup_complete",
            extra={
                "event": "startup",
                "model_loaded": True,
                "drift_detector_ready": True,
            },
        )
    else:
        drift_detector = None
        logger.warning(
            "reference_missing",
            extra={
                "event": "startup",
                "model_loaded": True,
                "drift_detector_ready": False,
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan-хук FastAPI для одноразового завантаження артефактів."""
    initialize_artifacts()
    yield


app = FastAPI(
    title="Iris ML API with Monitoring",
    description="ML API з Prometheus-метриками, JSON-логами та drift detection",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Вимірює latency endpoint `/predict`."""
    import time

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(elapsed)

    return response


def get_model() -> Any:
    """Повертає модель або завантажує її, якщо lifespan не виконався у тесті."""
    global model

    if model is None:
        try:
            initialize_artifacts()
        except RuntimeError as exc:
            ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return model


def get_drift_detector() -> DriftDetector:
    """Повертає drift detector або створює його за потреби."""
    global drift_detector

    if drift_detector is None:
        try:
            initialize_artifacts()
        except RuntimeError as exc:
            ERROR_COUNTER.labels(error_type="drift_detector_not_ready").inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if drift_detector is None:
        ERROR_COUNTER.labels(error_type="drift_detector_not_ready").inc()
        raise HTTPException(status_code=503, detail="Drift detector is not ready")

    return drift_detector


@app.get("/")
def root() -> dict[str, str]:
    """Базовий endpoint."""
    return {
        "status": "ok",
        "service": "Iris ML API",
        "version": "2.0.0",
    }


@app.get("/health")
def health() -> dict[str, bool | str]:
    """Health endpoint."""
    current_model = get_model()
    detector_ready = drift_detector is not None

    return {
        "status": "healthy",
        "model_loaded": current_model is not None,
        "drift_detector_ready": detector_ready,
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition endpoint."""
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    """Виконує інференс моделі для одного набору ознак."""
    current_model = get_model()

    x = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]
    )

    try:
        class_id = int(current_model.predict(x)[0])
        proba = float(current_model.predict_proba(x)[0, class_id])
    except Exception as exc:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.exception("inference_failed", extra={"event": "inference_error"})
        raise HTTPException(status_code=500, detail="Inference error") from exc

    class_name = CLASS_NAMES[class_id]
    PREDICTION_COUNTER.labels(class_name=class_name, status="success").inc()
    PREDICTION_CONFIDENCE.observe(proba)

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "class_id": class_id,
            "class_name": class_name,
            "probability": round(proba, 4),
            "features": features.model_dump(),
        },
    )

    return PredictionResponse(
        class_id=class_id,
        class_name=class_name,
        probability=round(proba, 4),
    )


@app.post("/check-drift", response_model=DriftResponse)
def check_drift(payload: DriftRequest) -> DriftResponse:
    """Перевіряє batch вхідних даних на drift."""
    detector = get_drift_detector()
    DRIFT_CHECKS.inc()

    current = np.array(payload.samples)
    try:
        result = detector.detect(current, alpha=payload.alpha)
    except ValueError as exc:
        ERROR_COUNTER.labels(error_type="invalid_drift_payload").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    for feature, info in result["per_feature"].items():
        if info["drift_detected"]:
            DRIFT_DETECTED.labels(feature=feature).inc()

    logger.info(
        "drift_check",
        extra={
            "event": "drift_check",
            "n_samples": len(payload.samples),
            "alpha": payload.alpha,
            "drift_detected": result["drift_detected"],
            "drifted_features": result["drifted_features"],
        },
    )

    return DriftResponse(**result)
