import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split

from pipeline import (
    create_gradient_boosting_pipeline,
    create_random_forest_pipeline,
    create_svc_pipeline,
)


DATA_PATH = "data/raw/winequality-red.csv"
MODEL_PATH = "models/best_model.pkl"
RANDOM_STATE = 42


def load_data():
    """
    Завантажує Wine Quality Dataset і готує його для класифікації.

    У датасеті є колонка quality з оцінкою якості вина.
    Для задачі класифікації створюємо нову цільову змінну:
    1 — хороше вино, якщо quality >= 7;
    0 — звичайне вино, якщо quality < 7.

    Повертає:
        X: ознаки;
        y: цільова змінна.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл {DATA_PATH} не знайдено. "
            "Перевірте, чи датасет завантажений у папку data/raw."
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    df["target"] = (df["quality"] >= 7).astype(int)

    X = df.drop(columns=["quality", "target"])
    y = df["target"]

    return X, y


def calculate_metrics(y_test, y_pred):
    """
    Обчислює основні метрики класифікації.
    """
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def run_experiment(model_name, pipeline, params):
    """
    Запускає один MLflow experiment run.

    Параметри:
        model_name: назва моделі.
        pipeline: sklearn Pipeline.
        params: словник гіперпараметрів.

    Повертає:
        натренований pipeline і accuracy.
    """
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow.set_experiment("wine-quality-classification")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset", "Wine Quality Red")
        mlflow.log_param("target_rule", "quality >= 7")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params(params)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_precision", test_metrics["precision"])
        mlflow.log_metric("test_recall", test_metrics["recall"])
        mlflow.log_metric("test_f1", test_metrics["f1"])
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

        mlflow.sklearn.log_model(pipeline, "model")

        print("=" * 60)
        print(f"Модель: {model_name}")
        print(f"Параметри: {params}")
        print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test precision: {test_metrics['precision']:.4f}")
        print(f"Test recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"CV accuracy mean: {cv_scores.mean():.4f}")
        print("=" * 60)

        return pipeline, test_metrics["accuracy"], test_metrics


def main():
    """
    Запускає всі експерименти для варіанта 6.
    """
    os.makedirs("models", exist_ok=True)

    experiments = [
        {
            "name": "RandomForest_1",
            "pipeline": create_random_forest_pipeline(
                n_estimators=100,
                max_depth=5,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
            },
        },
        {
            "name": "RandomForest_2",
            "pipeline": create_random_forest_pipeline(
                n_estimators=200,
                max_depth=10,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
            },
        },
        {
            "name": "RandomForest_3",
            "pipeline": create_random_forest_pipeline(
                n_estimators=300,
                max_depth=None,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": None,
            },
        },
        {
            "name": "GradientBoosting_1",
            "pipeline": create_gradient_boosting_pipeline(
                n_estimators=100,
                learning_rate=0.1,
            ),
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
            },
        },
        {
            "name": "GradientBoosting_2",
            "pipeline": create_gradient_boosting_pipeline(
                n_estimators=200,
                learning_rate=0.05,
            ),
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
            },
        },
        {
            "name": "SVC_1",
            "pipeline": create_svc_pipeline(
                c=1.0,
                kernel="rbf",
            ),
            "params": {
                "C": 1.0,
                "kernel": "rbf",
            },
        },
        {
            "name": "SVC_2",
            "pipeline": create_svc_pipeline(
                c=10.0,
                kernel="rbf",
            ),
            "params": {
                "C": 10.0,
                "kernel": "rbf",
            },
        },
    ]

    best_model = None
    best_model_name = None
    best_accuracy = 0
    results = []

    for experiment in experiments:
        trained_pipeline, accuracy, metrics = run_experiment(
            model_name=experiment["name"],
            pipeline=experiment["pipeline"],
            params=experiment["params"],
        )

        results.append({
            "model": experiment["name"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "params": experiment["params"],
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_pipeline
            best_model_name = experiment["name"]

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/experiment_results.csv", index=False)

    print("\nНайкраща модель:")
    print(f"Назва: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Модель збережено у файл: {MODEL_PATH}")
    print("Результати експериментів збережено у файл: models/experiment_results.csv")


if __name__ == "__main__":
    main()