"""FastAPI-сервіс для інференсу ML-моделі Iris."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from .schemas import IrisFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(
    title="Iris ML API",
    description="REST API для класифікації квіток Iris",
    version="1.0.0",
)

model: Any | None = None


def load_model_from_disk() -> Any:
    """Завантажує модель з файлу model.joblib."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def load_model() -> None:
    """Завантажує модель один раз під час старту API."""
    global model
    model = load_model_from_disk()


def get_model() -> Any:
    """Повертає модель або завантажує її, якщо startup ще не виконався."""
    global model
    if model is None:
        try:
            model = load_model_from_disk()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return model


@app.get("/")
def root() -> dict[str, str]:
    """Базовий endpoint для перевірки сервісу."""
    return {"status": "ok", "service": "Iris ML API"}


@app.get("/health")
def health() -> dict[str, bool | str]:
    """Endpoint для перевірки стану сервісу та моделі."""
    current_model = get_model()
    return {"status": "healthy", "model_loaded": current_model is not None}


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

    class_id = int(current_model.predict(x)[0])
    probabilities = current_model.predict_proba(x)[0]
    probability = float(probabilities[class_id])

    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        probability=round(probability, 4),
    )
