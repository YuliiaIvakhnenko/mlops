from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_random_forest_pipeline(n_estimators=100, max_depth=None):
    """
    Створює Pipeline для Random Forest Classifier.

    Параметри:
        n_estimators: кількість дерев у лісі.
        max_depth: максимальна глибина дерева.

    Повертає:
        sklearn Pipeline з масштабуванням і моделлю.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])


def create_gradient_boosting_pipeline(n_estimators=100, learning_rate=0.1):
    """
    Створює Pipeline для Gradient Boosting Classifier.

    Параметри:
        n_estimators: кількість boosting-ітерацій.
        learning_rate: швидкість навчання.

    Повертає:
        sklearn Pipeline з масштабуванням і моделлю.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        ))
    ])


def create_svc_pipeline(c=1.0, kernel="rbf"):
    """
    Створює Pipeline для Support Vector Machine.

    Параметри:
        c: параметр регуляризації.
        kernel: тип ядра SVM.

    Повертає:
        sklearn Pipeline з масштабуванням і моделлю.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=c,
            kernel=kernel,
            random_state=42
        ))
    ])


if __name__ == "__main__":
    pipeline = create_random_forest_pipeline()
    print("Pipeline успішно створено")
    print("Кроки Pipeline:", pipeline.named_steps.keys())