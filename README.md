# Лабораторна робота №3: Моніторинг ML API та детекція проблем

> **Тема:** Моніторинг ML API та детекція проблем  
> **Мета:** сформувати практичні навички побудови спостережуваного ML-сервісу: додати Prometheus-метрики, endpoint `/metrics`, перевірку data drift через KS-тест, endpoint `/check-drift`, структуроване JSON-логування та запуск Prometheus через Docker Compose.

![CI](https://github.com/YuliiaIvakhnenko/mlops/actions/workflows/ci.yml/badge.svg)

---

## 1. Опис проєкту

У межах лабораторної роботи було розширено ML API з попередньої лабораторної роботи. Базовий сервіс класифікує квітки Iris за чотирма числовими ознаками:

- `sepal_length`;
- `sepal_width`;
- `petal_length`;
- `petal_width`.

Модель навчається на стандартному датасеті Iris із бібліотеки `scikit-learn`, зберігається у файл `model.joblib`, а FastAPI-застосунок використовує її для виконання інференсу через endpoint `/predict`.

У ЛР3 до сервісу було додано:

- Prometheus-метрики для моніторингу ML API;
- endpoint `GET /metrics`;
- endpoint `POST /check-drift`;
- drift detector на основі KS-тесту;
- збереження reference-вибірки у `reference_stats.joblib`;
- структуроване JSON-логування;
- Docker Compose для запуску ML API разом із Prometheus;
- тести для API, метрик та drift detection;
- GitHub Actions workflow для автоматичної перевірки проєкту.

---

## 2. Використані технології

| Технологія | Призначення |
|---|---|
| Python 3.11 | основна мова розробки |
| FastAPI | реалізація REST API |
| Uvicorn | ASGI-сервер для запуску API |
| scikit-learn | навчання ML-моделі |
| joblib | збереження моделі та reference-даних |
| NumPy | робота з числовими масивами |
| SciPy | реалізація KS-тесту |
| prometheus-client | створення Prometheus-метрик |
| python-json-logger | структуроване JSON-логування |
| pytest | автоматичне тестування |
| Docker | контейнеризація ML API |
| Docker Compose | запуск ML API та Prometheus разом |
| Prometheus | збір і перегляд метрик |
| Evidently | опційна генерація HTML-звіту про drift |

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

## 4. Основні компоненти реалізації

### 4.1. `app/main.py`

Файл містить FastAPI-застосунок і реалізує основні endpoint-и:

- `/`;
- `/health`;
- `/predict`;
- `/metrics`;
- `/check-drift`.

Також у цьому файлі під час старту сервісу завантажуються модель `model.joblib`, reference-дані `reference_stats.joblib` і drift detector. Для вимірювання latency endpoint `/predict` використовується middleware, який фіксує час обробки запиту та записує його в Prometheus Histogram.

### 4.2. `app/metrics.py`

У цьому модулі оголошено Prometheus-метрики:

| Метрика | Тип | Призначення |
|---|---|---|
| `ml_predictions_total` | Counter | кількість прогнозів моделі за класом і статусом |
| `ml_prediction_latency_seconds` | Histogram | час обробки запиту `/predict` |
| `ml_prediction_confidence` | Histogram | розподіл значень упевненості моделі |
| `ml_errors_total` | Counter | кількість помилок за типом |
| `ml_model_loaded` | Gauge | чи завантажена модель |
| `ml_drift_checks_total` | Counter | кількість перевірок drift |
| `ml_drift_detected_total` | Counter | кількість виявлених drift-подій за ознаками |

### 4.3. `app/drift.py`

У файлі реалізовано клас `DriftDetector`. Він отримує reference-вибірку, тобто дані, на яких навчалася модель, і порівнює її з поточною live-вибіркою. Для кожної ознаки виконується KS-тест.

Логіка прийняття рішення:

```text
якщо p_value < alpha → drift виявлено
якщо p_value >= alpha → drift не виявлено
```

У роботі використано стандартне значення:

```text
alpha = 0.05
```

### 4.4. `app/logging_config.py`

Модуль налаштовує структуроване логування у форматі JSON.

Приклад лог-події:

```json
{
  "timestamp": "2026-05-09 00:18:25,405",
  "level": "INFO",
  "logger": "ml-api",
  "message": "startup_complete",
  "event": "startup",
  "model_loaded": true,
  "drift_detector_ready": true
}
```

Такі логи легше аналізувати, фільтрувати та передавати в системи централізованого логування.

### 4.5. `ml/train.py`

Скрипт тренування виконує:

1. завантаження датасету Iris;
2. поділ даних на train/test;
3. тренування `LogisticRegression`;
4. збереження моделі у `model.joblib`;
5. збереження reference-вибірки у `reference_stats.joblib`.

Файл `reference_stats.joblib` потрібен для подальшої перевірки data drift.

---

## 5. Локальний запуск

### 5.1. Клонування репозиторію

```bash
git clone -b lr3 https://github.com/YuliiaIvakhnenko/mlops.git
cd mlops
```

### 5.2. Створення віртуального середовища

#### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 5.4. Тренування моделі

```bash
python -m ml.train
```

Фактичний результат виконання:

```text
Model trained. Test accuracy: 0.9667
Saved model to: E:\sem2\mlops\mlops_lab3_monitoring\model.joblib
Saved reference to: E:\sem2\mlops\mlops_lab3_monitoring\reference_stats.joblib
```

Після виконання команди створюються файли:

```text
model.joblib
reference_stats.joblib
```

---

## 6. Запуск API локально

```bash
uvicorn app.main:app --reload
```

Після запуску API доступне за адресою:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

У Swagger UI мають бути доступні такі endpoint-и:

```text
GET  /
GET  /health
GET  /metrics
POST /predict
POST /check-drift
```

---

## 7. Endpoint-и API

### 7.1. `GET /`

Базовий endpoint для перевірки сервісу.

Приклад відповіді:

```json
{
  "status": "ok",
  "service": "Iris ML API",
  "version": "2.0.0"
}
```

### 7.2. `GET /health`

Endpoint для перевірки стану сервісу, моделі та drift detector.

Приклад відповіді:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "drift_detector_ready": true
}
```

### 7.3. `POST /predict`

Endpoint виконує інференс моделі для одного набору ознак Iris.

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

Після виконання `/predict` оновлюються метрики:

- `ml_predictions_total`;
- `ml_prediction_latency_seconds`;
- `ml_prediction_confidence`.

### 7.4. `GET /metrics`

Endpoint повертає Prometheus-метрики у форматі Prometheus exposition format.

Приклад частини відповіді:

```text
# HELP ml_predictions_total Загальна кількість прогнозів моделі
# TYPE ml_predictions_total counter
ml_predictions_total{class_name="setosa",status="success"} 50.0

# HELP ml_prediction_latency_seconds Час обробки одного прогнозу в секундах
# TYPE ml_prediction_latency_seconds histogram
ml_prediction_latency_seconds_count 50.0

# HELP ml_prediction_confidence Розподіл значень упевненості моделі
# TYPE ml_prediction_confidence histogram
ml_prediction_confidence_count 50.0

# HELP ml_model_loaded Чи успішно завантажена модель: 1 = так, 0 = ні
# TYPE ml_model_loaded gauge
ml_model_loaded 1.0
```

### 7.5. `POST /check-drift`

Endpoint приймає batch вхідних даних і перевіряє, чи відрізняється їх розподіл від reference-вибірки.

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

Фактичний результат перевірки:

```json
{
  "drift_detected": true,
  "n_drifted_features": 4,
  "drifted_features": [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
  ],
  "per_feature": {
    "sepal_length": {
      "statistic": 1.0,
      "p_value": 7.507471973909942e-15,
      "drift_detected": true
    },
    "sepal_width": {
      "statistic": 1.0,
      "p_value": 7.507471973909942e-15,
      "drift_detected": true
    },
    "petal_length": {
      "statistic": 1.0,
      "p_value": 7.507471973909942e-15,
      "drift_detected": true
    },
    "petal_width": {
      "statistic": 1.0,
      "p_value": 7.507471973909942e-15,
      "drift_detected": true
    }
  },
  "n_samples": 10,
  "alpha": 0.05
}
```

Після виконання `/check-drift` оновлюються метрики:

```text
ml_drift_checks_total 1.0
ml_drift_detected_total{feature="sepal_length"} 1.0
ml_drift_detected_total{feature="sepal_width"} 1.0
ml_drift_detected_total{feature="petal_length"} 1.0
ml_drift_detected_total{feature="petal_width"} 1.0
```

---

## 8. Запуск тестів

Для запуску тестів використовується команда:

```bash
pytest -q
```

Фактичний результат:

```text
12 passed, 1 warning in 4.59s
```

Попередження не є критичною помилкою. Воно пов'язане з deprecated-повідомленням бібліотеки `python-json-logger`.

Тести перевіряють:

- створення `model.joblib`;
- створення `reference_stats.joblib`;
- endpoint `/`;
- endpoint `/health`;
- endpoint `/predict`;
- endpoint `/metrics`;
- інкрементування Prometheus-лічильників;
- роботу `DriftDetector`;
- endpoint `/check-drift`;
- виявлення drift для зміщених даних.

---

## 9. Docker-запуск ML API

### 9.1. Збірка Docker-образу

```bash
docker build -t ml-api:lab3 .
```

### 9.2. Запуск контейнера

Якщо порт `8000` вільний:

```bash
docker run --rm -p 8000:8000 ml-api:lab3
```

Якщо порт `8000` зайнятий, можна використати порт `8001`:

```bash
docker run --rm -p 8001:8000 ml-api:lab3
```

Тоді перевірка API виконується за адресами:

```text
http://127.0.0.1:8001/health
http://127.0.0.1:8001/metrics
http://127.0.0.1:8001/docs
```

---

## 10. Запуск Prometheus через Docker Compose

Для запуску ML API разом із Prometheus потрібно перейти в каталог `monitoring`:

```bash
cd monitoring
```

Команда запуску:

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

Альтернативна команда для старіших версій Docker Compose:

```bash
docker-compose -f docker-compose.monitoring.yml up --build
```

У цій роботі зовнішній порт API було змінено на `8001`, щоб уникнути конфлікту з уже запущеним контейнером:

```yaml
ports:
  - "8001:8000"
```

Важливо: Prometheus всередині Docker-мережі все одно звертається до API за адресою:

```text
ml-api:8000
```

А для браузера на Windows API доступне через:

```text
http://127.0.0.1:8001
```

Після запуску доступні адреси:

```text
ML API:      http://127.0.0.1:8001
Swagger:     http://127.0.0.1:8001/docs
Health:      http://127.0.0.1:8001/health
Metrics:     http://127.0.0.1:8001/metrics
Prometheus:  http://127.0.0.1:9090
Targets:     http://127.0.0.1:9090/targets
Graph:       http://127.0.0.1:9090/graph
```

---

## 11. Перевірка Prometheus

### 11.1. Перевірка targets

На сторінці:

```text
http://127.0.0.1:9090/targets
```

Prometheus показав:

```text
ml-api     1/1 up
prometheus 1/1 up
```

Це означає, що Prometheus успішно збирає метрики з ML API.

### 11.2. Перевірка `ml_model_loaded`

У Prometheus Graph було виконано запит:

```promql
ml_model_loaded
```

Результат:

```text
ml_model_loaded = 1
```

Це підтверджує, що модель завантажена.

### 11.3. Перевірка `ml_predictions_total`

Після генерації 50 запитів до `/predict` було виконано запит:

```promql
ml_predictions_total
```

Результат:

```text
ml_predictions_total{class_name="setosa",status="success"} 50
```

Це підтверджує, що Prometheus збирає кількість прогнозів моделі.

### 11.4. Перевірка `ml_drift_detected_total`

Після виконання `/check-drift` було виконано запит:

```promql
ml_drift_detected_total
```

Prometheus показав 4 часові ряди:

```text
ml_drift_detected_total{feature="sepal_length"} 1
ml_drift_detected_total{feature="sepal_width"} 1
ml_drift_detected_total{feature="petal_length"} 1
ml_drift_detected_total{feature="petal_width"} 1
```

Це підтверджує, що drift detection інтегрований із системою метрик.

---

## 12. Генерація навантаження

Для генерації 50 запитів до `/predict` було використано команду:

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8001/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
    > /dev/null
done
```

Після виконання команди у `/metrics` з'явилося:

```text
ml_predictions_total{class_name="setosa",status="success"} 50.0
ml_prediction_latency_seconds_count 50.0
ml_prediction_confidence_count 50.0
```

---

## 13. Опційний Evidently-звіт

У проєкті додано скрипт:

```text
scripts/evidently_report.py
```

Він може бути використаний для генерації HTML-звіту про drift.

Для запуску потрібно встановити додаткові залежності:

```bash
pip install -r requirements-evidently.txt
```

Потім виконати:

```bash
python scripts/evidently_report.py
```

У результаті буде створено файл:

```text
drift_report.html
```

Evidently-звіт є додатковою частиною роботи та не замінює онлайн-перевірку drift через endpoint `/check-drift`.

---

## 14. GitHub Actions CI

У репозиторії налаштовано workflow:

```text
.github/workflows/ci.yml
```

Workflow запускається при `push` та `pull_request` і виконує:

1. клонування репозиторію;
2. встановлення Python 3.11;
3. встановлення залежностей;
4. тренування моделі;
5. створення `reference_stats.joblib`;
6. запуск `pytest`;
7. збірку Docker-образу.

---

## 15. Скріншоти для підтвердження

Для підтвердження виконання роботи були підготовлені такі скріншоти:

1. запуск `python -m ml.train`;
2. запуск `pytest -q`;
3. Swagger UI з endpoint-ами `/metrics` і `/check-drift`;
4. відповідь `/health`;
5. відповідь `/metrics`;
6. виконання `/check-drift` з `drift_detected: true`;
7. Docker Compose запуск ML API та Prometheus;
8. Prometheus Targets зі статусом `ml-api UP`;
9. Prometheus Graph із запитом `ml_model_loaded`;
10. Prometheus Graph із запитом `ml_predictions_total`;
11. Prometheus Graph із запитом `ml_drift_detected_total`.

---

## 16. Фактичні результати виконання

| Перевірка | Результат |
|---|---|
| Тренування моделі | `Test accuracy: 0.9667` |
| Файл моделі | `model.joblib` створено |
| Reference-дані | `reference_stats.joblib` створено |
| Тести | `12 passed, 1 warning` |
| `/health` | `model_loaded: true`, `drift_detector_ready: true` |
| `/predict` | повертає клас `setosa` |
| `/metrics` | повертає Prometheus-метрики |
| Кількість прогнозів | `ml_predictions_total = 50` |
| Latency-метрика | `ml_prediction_latency_seconds_count = 50` |
| Confidence-метрика | `ml_prediction_confidence_count = 50` |
| `/check-drift` | `drift_detected: true` |
| Drifted features | 4 ознаки |
| `ml_drift_checks_total` | `1.0` |
| `ml_drift_detected_total` | 4 часові ряди |
| Prometheus Targets | `ml-api UP`, `prometheus UP` |

---

## 17. Контрольні питання

### 1. Для чого потрібен моніторинг ML API?

Моніторинг ML API потрібен для спостереження за технічним станом сервісу та поведінкою моделі після розгортання. На відміну від звичайних програм, ML-модель може погіршувати якість прогнозів без зміни коду, якщо змінюється розподіл вхідних даних. Тому важливо контролювати кількість запитів, latency, помилки, confidence, розподіл прогнозів і data drift.

### 2. Що таке Prometheus?

Prometheus — це система моніторингу та збору часових рядів. Вона періодично опитує endpoint `/metrics` сервісу, зчитує метрики у Prometheus exposition format і зберігає їх для подальшого аналізу через PromQL.

### 3. Як Prometheus збирає метрики?

Prometheus працює за pull-моделлю. Це означає, що не сервіс надсилає метрики в Prometheus, а Prometheus сам регулярно виконує HTTP-запити до endpoint `/metrics` заданих targets.

У цій роботі Prometheus опитує:

```text
http://ml-api:8000/metrics
```

### 4. Які типи метрик Prometheus використовуються в роботі?

У роботі використовуються:

- **Counter** — лічильник подій, наприклад `ml_predictions_total`;
- **Histogram** — розподіл значень, наприклад latency або confidence;
- **Gauge** — поточний стан, наприклад `ml_model_loaded`.

### 5. Що таке data drift?

Data drift — це зміна статистичного розподілу вхідних даних порівняно з даними, на яких модель навчалася. Якщо live-дані значно відрізняються від reference-вибірки, модель може почати працювати гірше.

### 6. Як працює KS-тест у цій лабораторній роботі?

KS-тест порівнює розподіл reference-вибірки та поточної вибірки для кожної числової ознаки. Якщо `p_value < alpha`, вважається, що розподіли статистично значуще відрізняються, тобто для цієї ознаки виявлено drift.

У роботі використано:

```text
alpha = 0.05
```

### 7. Що означає `p_value`?

`p_value` показує, наскільки ймовірно отримати таку різницю між вибірками, якщо насправді вони походять з одного розподілу. Якщо `p_value` мале, наприклад менше `0.05`, є підстави вважати, що розподіли відрізняються.

### 8. Для чого потрібне структуроване JSON-логування?

Структуроване JSON-логування дозволяє зберігати події у машинно-читабельному форматі. Такі логи легше аналізувати, фільтрувати й передавати до систем централізованого логування.

### 9. Навіщо зберігати `reference_stats.joblib`?

Файл `reference_stats.joblib` містить reference-вибірку, тобто навчальні дані, з якими порівнюються live-дані під час перевірки drift. Без цього файлу endpoint `/check-drift` не зможе виконати статистичне порівняння.

### 10. Що означає статус `UP` у Prometheus Targets?

Статус `UP` означає, що Prometheus успішно підключився до target і зміг прочитати метрики з endpoint `/metrics`.

---

## 18. Висновок

У лабораторній роботі було реалізовано спостережуваний ML API для класифікації квіток Iris. Сервіс було розширено Prometheus-метриками, endpoint `/metrics`, endpoint `/check-drift`, drift detector на основі KS-тесту та структурованим JSON-логуванням.

Було налаштовано запуск ML API разом із Prometheus через Docker Compose. Prometheus успішно отримав статус `UP` для target `ml-api` і зібрав метрики `ml_model_loaded`, `ml_predictions_total` та `ml_drift_detected_total`.

Фактичні результати підтвердили коректність реалізації: модель навчилася з accuracy `0.9667`, тести пройшли успішно (`12 passed, 1 warning`), endpoint `/metrics` повернув Prometheus-метрики, а endpoint `/check-drift` виявив drift по всіх 4 ознаках.
