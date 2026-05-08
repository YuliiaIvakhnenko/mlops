"""Pydantic-схеми для API інференсу Iris."""

from pydantic import BaseModel, Field


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
