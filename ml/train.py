"""Тренування простої моделі класифікації Iris."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


def train_and_save(model_path: Path = MODEL_PATH) -> float:
    """Навчає модель Iris та зберігає її у файл.

    Args:
        model_path: Шлях, куди потрібно зберегти навчену модель.

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

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, model_path)
    return accuracy


if __name__ == "__main__":
    acc = train_and_save()
    print(f"Model trained. Test accuracy: {acc:.4f}")
    print(f"Saved to: {MODEL_PATH}")
