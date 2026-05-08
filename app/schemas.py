"""Pydantic-схеми для ML API з моніторингом."""

from typing import Dict, List

from pydantic import BaseModel, Field, conlist


class IrisFeatures(BaseModel):
    """Вхідні ознаки квітки Iris."""

    sepal_length: float = Field(..., ge=0, le=10, description="Довжина чашолистика, см")
    sepal_width: float = Field(..., ge=0, le=10, description="Ширина чашолистика, см")
    petal_length: float = Field(..., ge=0, le=10, description="Довжина пелюстки, см")
    petal_width: float = Field(..., ge=0, le=10, description="Ширина пелюстки, см")


class PredictionResponse(BaseModel):
    """Відповідь API з результатом класифікації."""

    class_id: int
    class_name: str
    probability: float


class DriftRequest(BaseModel):
    """Батч даних для перевірки data drift."""

    samples: conlist(conlist(float, min_length=4, max_length=4), min_length=10)
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Поріг значущості для KS-тесту",
    )


class FeatureDriftInfo(BaseModel):
    """Результат drift-перевірки для однієї ознаки."""

    statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    """Відповідь endpoint `/check-drift`."""

    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
