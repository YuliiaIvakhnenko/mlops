"""Тренування моделі Iris із reference-статистиками для drift detection."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
REFERENCE_PATH = ROOT / "reference_stats.joblib"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train_and_save(
    model_path: Path = MODEL_PATH,
    reference_path: Path = REFERENCE_PATH,
) -> float:
    """Тренує модель та зберігає модель і reference-вибірку.

    Args:
        model_path: Шлях для збереження моделі.
        reference_path: Шлях для збереження reference-даних.

    Returns:
        Accuracy моделі на тестовій вибірці.
    """
    x, y = load_iris(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    joblib.dump(model, model_path)
    joblib.dump(
        {
            "X": x_train,
            "feature_names": FEATURE_NAMES,
        },
        reference_path,
    )

    return accuracy


if __name__ == "__main__":
    acc = train_and_save()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved reference to: {REFERENCE_PATH}")
