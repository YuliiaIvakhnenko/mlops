"""
Підготовка датасету Wine Quality для лабораторної роботи №1.

Скрипт перевіряє наявність файлу data/raw/winequality-red.csv.
Якщо файл уже є в проєкті, він валідується та виводиться коротка інформація.
Якщо файлу немає, його можна скопіювати з локального шляху через параметр --source.

Приклади запуску:
    python src/create_dataset.py
    python src/create_dataset.py --source data/external/winequality-red.csv
    python src/create_dataset.py --source ~/Downloads/winequality-red.csv --force
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = RAW_DIR / "winequality-red.csv"
REQUIRED_COLUMNS = {
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
    "quality",
}


def validate_dataset(path: Path) -> pd.DataFrame:
    """
    Перевіряє структуру датасету Wine Quality.

    Args:
        path: шлях до CSV-файлу датасету.

    Returns:
        DataFrame із завантаженими даними.

    Raises:
        FileNotFoundError: якщо файл не знайдено.
        ValueError: якщо у файлі немає необхідних колонок.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {path}")

    df = pd.read_csv(path, sep=";")
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Датасет має неправильну структуру. "
            f"Відсутні колонки: {sorted(missing_columns)}"
        )

    return df


def prepare_dataset(source: Path | None = None, force: bool = False) -> Path:
    """
    Готує датасет у директорії data/raw.

    Якщо цільовий файл уже існує і force=False, файл не перезаписується.
    Якщо source задано, файл копіюється в data/raw/winequality-red.csv.

    Args:
        source: локальний шлях до CSV-файлу Wine Quality.
        force: чи перезаписувати наявний файл.

    Returns:
        Шлях до підготовленого датасету.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and not force:
        print(f"Датасет уже існує: {DATASET_PATH}")
    else:
        if source is None:
            raise FileNotFoundError(
                "Датасет не знайдено. Додайте файл winequality-red.csv у data/raw "
                "або передайте шлях через --source."
            )

        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Файл-джерело не знайдено: {source}")

        shutil.copy2(source, DATASET_PATH)
        print(f"Датасет скопійовано з {source} у {DATASET_PATH}")

    df = validate_dataset(DATASET_PATH)
    print("Датасет успішно підготовлено.")
    print(f"Шлях: {DATASET_PATH}")
    print(f"Розмір: {df.shape[0]} рядків, {df.shape[1]} колонок")
    print("Цільова колонка: quality")
    print("Правило класифікації у train.py: target = 1, якщо quality >= 7")

    return DATASET_PATH


def parse_args() -> argparse.Namespace:
    """Зчитує аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Підготовка Wine Quality dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Локальний шлях до файлу winequality-red.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписати data/raw/winequality-red.csv, якщо файл уже існує",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(source=args.source, force=args.force)
