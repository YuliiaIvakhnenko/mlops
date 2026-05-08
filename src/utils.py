"""
Допоміжні функції для використання натренованої ML-моделі.

Файл містить завантаження моделі з pickle-файлу, завантаження моделі з MLflow
та виконання прогнозування для нових даних.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "winequality-red.csv"
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_model_from_file(path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """
    Завантажує натренований pipeline з pickle-файлу.

    Args:
        path: шлях до файлу моделі.

    Returns:
        Завантажений sklearn Pipeline.

    Raises:
        FileNotFoundError: якщо файл моделі не знайдено.
    """
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не знайдено: {model_path}. "
            "Спочатку запустіть python src/train.py або виконайте dvc pull."
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


def load_model_from_mlflow(run_id: str, artifact_path: str = "model") -> Any:
    """
    Завантажує модель з MLflow за run_id.

    Args:
        run_id: ідентифікатор MLflow run.
        artifact_path: шлях до артефакту моделі всередині run.

    Returns:
        Завантажена MLflow/sklearn модель.
    """
    try:
        import mlflow.sklearn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Пакет mlflow не встановлено. Виконайте: pip install -r requirements.txt"
        ) from exc

    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)


def predict(model: Any, x_data: pd.DataFrame) -> pd.Series:
    """
    Виконує прогнозування класу якості вина.

    Args:
        model: натренований sklearn Pipeline.
        x_data: DataFrame з ознаками без колонки quality.

    Returns:
        Series з прогнозами: 1 — хороше вино, 0 — звичайне вино.
    """
    missing_columns = set(FEATURE_COLUMNS) - set(x_data.columns)
    if missing_columns:
        raise ValueError(f"У вхідних даних відсутні колонки: {sorted(missing_columns)}")

    predictions = model.predict(x_data[FEATURE_COLUMNS])
    return pd.Series(predictions, name="predicted_quality_class")


def load_sample_data(rows: int = 5) -> pd.DataFrame:
    """
    Завантажує кілька прикладів із датасету для демонстрації predict.

    Args:
        rows: кількість рядків для прикладу.

    Returns:
        DataFrame з ознаками моделі.
    """
    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не знайдено: {DEFAULT_DATASET_PATH}. Виконайте dvc pull."
        )

    df = pd.read_csv(DEFAULT_DATASET_PATH, sep=";")
    return df[FEATURE_COLUMNS].head(rows)


if __name__ == "__main__":
    model = load_model_from_file()
    sample = load_sample_data(rows=5)
    result = predict(model, sample)

    print("Приклад вхідних даних:")
    print(sample)
    print("\nПрогноз моделі:")
    print(result)
