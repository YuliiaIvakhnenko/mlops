# Лабораторна робота №3: Моніторинг ML API та детекція проблем

> **Тема:** Моніторинг ML API та детекція проблем  
> **Мета:** розширити ML API з ЛР2 засобами observability: Prometheus-метриками, endpoint `/metrics`, детекцією data drift через KS-тест, endpoint `/check-drift`, структурованим JSON-логуванням і docker-compose для Prometheus.

![CI](https://github.com/YuliiaIvakhnenko/mlops/actions/workflows/ci.yml/badge.svg)

---

## 1. Опис проєкту

Проєкт реалізує FastAPI-сервіс для класифікації квіток Iris і доповнює його компонентами моніторингу.

У порівнянні з ЛР2 додано:

- Prometheus-метрики для прогнозів, latency, confidence, помилок і drift-перевірок;
- endpoint `GET /metrics` у Prometheus exposition format;
- endpoint `POST /check-drift` для перевірки data drift;
- клас `DriftDetector`, який виконує KS-тест для кожної ознаки;
- збереження `reference_stats.joblib` під час тренування;
- структуровані JSON-логи;
- `monitoring/prometheus.yml`;
- `monitoring/docker-compose.monitoring.yml`;
- тести для метрик і drift detection;
- опційний скрипт Evidently-звіту.

---

## 2. Стек технологій

| Технологія | Призначення |
|---|---|
| Python 3.11 | основна мова розробки |
| FastAPI | REST API |
| scikit-learn | навчання ML-моделі |
| joblib | збереження моделі та reference-даних |
| NumPy | робота з масивами |
| SciPy | KS-тест для drift detection |
| prometheus-client | Prometheus-метрики |
| python-json-logger | структуроване JSON-логування |
| pytest | автоматичне тестування |
| Docker | контейнеризація API |
| Docker Compose | запуск API разом із Prometheus |
| Prometheus | збір і перегляд метрик |
| Evidently | опційний HTML-звіт про drift |

---

## 3. Структура проєкту

```text
mlops_lab3_monitoring/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── metrics.py
│   ├── drift.py
│   └── logging_config.py
├── ml/
│   ├── __init__.py
│   └── train.py
├── monitoring/
│   ├── prometheus.yml
│   └── docker-compose.monitoring.yml
├── scripts/
│   └── evidently_report.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_drift.py
├── model.joblib
├── reference_stats.joblib
├── requirements.txt
├── requirements-evidently.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. Локальний запуск

### 4.1. Клонування репозиторію

```bash
git clone -b lr3 https://github.com/YuliiaIvakhnenko/mlops.git
cd mlops
```

### 4.2. Створення віртуального середовища

#### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 4.4. Тренування моделі та reference-статистик

```bash
python -m ml.train
```

Очікуваний результат:

```text
Model trained. Test accuracy: 0.9667
Saved model to: .../model.joblib
Saved reference to: .../reference_stats.joblib
```

Після запуску мають з'явитися два файли:

```text
model.joblib
reference_stats.joblib
```

---

## 5. Запуск API

```bash
uvicorn app.main:app --reload
```

API буде доступне:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 6. Endpoint-и API

| Метод | Endpoint | Призначення |
|---|---|---|
| GET | `/` | базова перевірка сервісу |
| GET | `/health` | стан API, моделі та drift detector |
| POST | `/predict` | інференс моделі |
| GET | `/metrics` | Prometheus-метрики |
| POST | `/check-drift` | перевірка data drift |

### 6.1. `/health`

```http
GET /health
```

Приклад відповіді:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "drift_detector_ready": true
}
```

### 6.2. `/predict`

```http
POST /predict
```

Приклад запиту:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Приклад відповіді:

```json
{
  "class_id": 0,
  "class_name": "setosa",
  "probability": 0.9786
}
```

### 6.3. `/metrics`

```http
GET /metrics
```

У відповіді мають бути Prometheus-метрики, наприклад:

```text
ml_predictions_total{class_name="setosa",status="success"} 1.0
ml_prediction_latency_seconds_count 1.0
ml_prediction_confidence_count 1.0
ml_model_loaded 1.0
```

### 6.4. `/check-drift`

```http
POST /check-drift
```

Приклад запиту без суттєвого drift:

```json
{
  "samples": [
    [5.1, 3.5, 1.4, 0.2],
    [5.0, 3.4, 1.5, 0.2],
    [5.2, 3.6, 1.4, 0.3],
    [5.1, 3.5, 1.5, 0.2],
    [5.0, 3.5, 1.4, 0.2],
    [5.2, 3.4, 1.5, 0.2],
    [5.1, 3.6, 1.4, 0.3],
    [5.0, 3.5, 1.5, 0.2],
    [5.1, 3.4, 1.4, 0.2],
    [5.2, 3.5, 1.5, 0.3]
  ],
  "alpha": 0.05
}
```

Приклад запиту з явним drift:

```json
{
  "samples": [
    [9.0, 9.0, 9.0, 9.0],
    [9.1, 9.0, 9.0, 9.0],
    [9.0, 9.1, 9.0, 9.0],
    [9.0, 9.0, 9.1, 9.0],
    [9.0, 9.0, 9.0, 9.1],
    [9.2, 9.0, 9.0, 9.0],
    [9.0, 9.2, 9.0, 9.0],
    [9.0, 9.0, 9.2, 9.0],
    [9.0, 9.0, 9.0, 9.2],
    [9.3, 9.3, 9.3, 9.3]
  ],
  "alpha": 0.05
}
```

---

## 7. Prometheus-метрики

У проєкті реалізовано такі метрики:

| Метрика | Тип | Призначення |
|---|---|---|
| `ml_predictions_total` | Counter | кількість прогнозів за класом і статусом |
| `ml_prediction_latency_seconds` | Histogram | час обробки прогнозу |
| `ml_prediction_confidence` | Histogram | розподіл упевненості моделі |
| `ml_errors_total` | Counter | кількість помилок за типом |
| `ml_model_loaded` | Gauge | чи завантажена модель |
| `ml_drift_checks_total` | Counter | кількість перевірок drift |
| `ml_drift_detected_total` | Counter | кількість виявлених drift-подій за ознакою |

Корисні PromQL-запити:

```promql
ml_predictions_total
```

```promql
rate(ml_predictions_total[1m])
```

```promql
sum by (class_name) (ml_predictions_total)
```

```promql
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
```

```promql
ml_drift_detected_total
```

---

## 8. Запуск тестів

```bash
pytest -q
```

Очікуваний результат:

```text
12 passed
```

Тести перевіряють:

- створення `model.joblib`;
- створення `reference_stats.joblib`;
- endpoint-и `/`, `/health`, `/predict`;
- endpoint `/metrics`;
- інкрементування Prometheus-лічильників;
- роботу `DriftDetector`;
- endpoint `/check-drift`.

---

## 9. Запуск Docker API

### 9.1. Збірка образу

```bash
docker build -t ml-api:lab3 .
```

### 9.2. Запуск контейнера

```bash
docker run --rm -p 8000:8000 ml-api:lab3
```

Перевірка:

```text
http://127.0.0.1:8000/health
```

---

## 10. Запуск Prometheus через Docker Compose

Перейдіть у каталог `monitoring`:

```bash
cd monitoring
```

Запустіть API + Prometheus:

```bash
docker-compose -f docker-compose.monitoring.yml up --build
```

Або у нових версіях Docker:

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

Після запуску:

```text
ML API:      http://127.0.0.1:8000
Swagger:     http://127.0.0.1:8000/docs
Metrics:     http://127.0.0.1:8000/metrics
Prometheus:  http://127.0.0.1:9090
Targets:     http://127.0.0.1:9090/targets
Graph:       http://127.0.0.1:9090/graph
```

У Prometheus на сторінці `/targets` ціль `ml-api` має бути в статусі `UP`.

---

## 11. Генерація навантаження

У Git Bash можна виконати:

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict     -H "Content-Type: application/json"     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'     > /dev/null
done
```

Після цього у Prometheus можна перевірити:

```promql
ml_predictions_total
```

або:

```promql
sum by (class_name) (ml_predictions_total)
```

---

## 12. Опційний Evidently-звіт

Для додаткового завдання встановіть залежності:

```bash
pip install -r requirements-evidently.txt
```

Запустіть:

```bash
python scripts/evidently_report.py
```

Буде створено файл:

```text
drift_report.html
```

---

## 13. GitHub Actions CI

Workflow знаходиться у файлі:

```text
.github/workflows/ci.yml
```

Він виконує:

1. встановлення залежностей;
2. тренування моделі;
3. створення reference-статистик;
4. запуск `pytest`;
5. збірку Docker-образу.

---

## 14. Контрольні питання

### 1. Для чого потрібен моніторинг ML API?

Моніторинг потрібен, щоб відстежувати технічний стан сервісу та поведінку моделі після розгортання. ML-сервіс може деградувати навіть без зміни коду, якщо змінюються вхідні дані. Тому важливо контролювати latency, кількість запитів, помилки, розподіл прогнозів, confidence та data drift.

### 2. Що таке Prometheus і як він збирає метрики?

Prometheus — це система моніторингу часових рядів. Вона працює за pull-моделлю: сама періодично звертається до endpoint `/metrics` сервісу, читає метрики у Prometheus exposition format і зберігає їх у своїй базі.

### 3. Які типи метрик Prometheus використані в роботі?

У роботі використано Counter, Histogram і Gauge. Counter рахує події, наприклад кількість прогнозів. Histogram збирає розподіли latency та confidence. Gauge показує поточний стан, наприклад чи завантажена модель.

### 4. Що таке data drift?

Data drift — це зміна статистичного розподілу вхідних даних порівняно з навчальною вибіркою. Якщо live-дані починають сильно відрізнятися від reference-даних, якість моделі може погіршитися.

### 5. Як працює KS-тест у DriftDetector?

KS-тест порівнює розподіли reference-вибірки та поточної вибірки для кожної числової ознаки. Якщо `p_value < alpha`, вважається, що розподіли значуще відрізняються, тобто для цієї ознаки виявлено drift.

### 6. Для чого потрібне структуроване JSON-логування?

Структуроване логування дозволяє записувати події у машинно-читабельному JSON-форматі. Такі логи легше фільтрувати, аналізувати та передавати в системи на кшталт ELK, Loki або CloudWatch.

---

## 15. Фактичні результати перевірки

Після локального запуску проєкту очікуються такі результати:

| Перевірка | Очікуваний результат |
|---|---|
| Тренування моделі | створено `model.joblib` і `reference_stats.joblib` |
| `/health` | `model_loaded: true`, `drift_detector_ready: true` |
| `/predict` | повертає клас Iris і probability |
| `/metrics` | містить `ml_predictions_total` та інші метрики |
| `/check-drift` | повертає `drift_detected` і деталі по ознаках |
| `pytest -q` | `12 passed` |
| Docker | API доступний на `http://127.0.0.1:8000` |
| Prometheus | target `ml-api` у статусі `UP` |

---

## 16. Висновок

У лабораторній роботі реалізовано observable ML API. Сервіс було розширено Prometheus-метриками, endpoint `/metrics`, endpoint `/check-drift`, статистичним детектором data drift на основі KS-тесту та структурованим JSON-логуванням. Додано Docker Compose конфігурацію для запуску Prometheus і ML API в одній мережі. Проєкт готовий до локального запуску, тестування, контейнеризації, збору метрик і подальшого аналізу drift.
