# Лабораторна робота №2: CI/CD та ML API

> **Тема:** CI/CD та ML API  
> **Мета:** реалізувати повний навчальний MLOps-конвеєр: ML-модель → FastAPI REST API → тести → Docker → GitHub Actions → Render.

![CI](https://github.com/YuliiaIvakhnenko/mlops/actions/workflows/ci.yml/badge.svg)

---

## 1. Опис проєкту

Цей проєкт реалізує REST API для класифікації квіток Iris за допомогою навченої ML-моделі.

Модель навчається на датасеті Iris із `scikit-learn`, зберігається у файл `model.joblib`, а потім використовується FastAPI-застосунком для інференсу через endpoint `/predict`.

Проєкт демонструє базові MLOps-практики:

- автоматизоване тренування моделі;
- REST API для інференсу;
- unit-тести та інтеграційні API-тести;
- контейнеризацію через Docker;
- CI-пайплайн через GitHub Actions;
- підготовку до деплою на Render.

---

## 2. Стек технологій

| Технологія | Призначення |
|---|---|
| Python 3.11 | основна мова розробки |
| scikit-learn | навчання ML-моделі |
| joblib | збереження моделі |
| FastAPI | створення REST API |
| Pydantic | валідація JSON-запитів |
| Uvicorn | ASGI-сервер для запуску API |
| pytest | автоматичне тестування |
| httpx / TestClient | тестування API |
| Docker | контейнеризація застосунку |
| GitHub Actions | CI-перевірки |
| Render | хмарне розгортання API |

---

## 3. Структура репозиторію

```text
mlops_lab2_ml_api/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── ml/
│   ├── __init__.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── model.joblib
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. Як запустити локально

### 4.1. Клонування репозиторію

```bash
git clone -b lr2 https://github.com/YuliiaIvakhnenko/mlops.git
cd mlops
```

### 4.2. Створення віртуального середовища

#### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
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

### 4.4. Тренування моделі

```bash
python -m ml.train
```

Очікуваний результат:

```text
Model trained. Test accuracy: 0.9333
Saved to: .../model.joblib
```

Фактичний результат під час перевірки:

```text
Model trained. Test accuracy: 0.9667
Saved to: .../model.joblib
```

Після цього в корені проєкту має з'явитися файл:

```text
model.joblib
```

### 4.5. Запуск API

```bash
uvicorn app.main:app --reload
```

API буде доступне за адресою:

```text
http://127.0.0.1:8000
```

Документація Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 5. Як працює API

### 5.1. Перевірка сервісу

```http
GET /
```

Відповідь:

```json
{
  "status": "ok",
  "service": "Iris ML API"
}
```

### 5.2. Health check

```http
GET /health
```

Відповідь:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 5.3. Передбачення класу Iris

```http
POST /predict
```

Приклад JSON-запиту:

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
  "probability": 0.98
}
```

---

## 6. Запуск тестів

Для запуску тестів використовується команда:

```bash
pytest -q
```

Очікуваний результат:

```text
6 passed
```

Фактичний результат виконання:

```text
6 passed, 4 warnings
```

Попередження не є критичними помилками та пов'язані з deprecated-повідомленнями бібліотек. Усі тести пройшли успішно.

Тести перевіряють:

- створення файлу моделі;
- коректність accuracy;
- можливість передбачення одного з трьох класів;
- endpoint `/`;
- endpoint `/health`;
- endpoint `/predict`;
- валідацію некоректного запиту з HTTP-кодом `422`.

---

## 7. Запуск через Docker

### 7.1. Збірка Docker-образу

```bash
docker build -t ml-api:lab2 .
```

### 7.2. Запуск контейнера

```bash
docker run --rm -p 8000:8000 ml-api:lab2
```

Після запуску перевірити:

```text
http://127.0.0.1:8000/health
```

Фактичний результат перевірки endpoint `/health`:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Це підтверджує, що API успішно працює всередині Docker-контейнера.

---

## 8. GitHub Actions CI

У проєкті налаштовано workflow:

```text
.github/workflows/ci.yml
```

Workflow автоматично запускається при `push` та `pull_request`.

Він виконує два jobs:

1. **Test Python project**
   - checkout репозиторію;
   - встановлення Python 3.11;
   - встановлення залежностей;
   - тренування моделі;
   - запуск `pytest`.

2. **Build Docker image**
   - збірка Docker-образу;
   - перевірка, що `Dockerfile` коректний.

Після успішного виконання workflow у вкладці **Actions** має бути зелений статус.

---

## 9. Деплой на Render

Для розгортання потрібно:

1. Перейти на [Render](https://render.com).
2. Авторизуватися через GitHub.
3. Натиснути **New → Web Service**.
4. Обрати репозиторій із цією лабораторною.
5. У полі **Environment** вибрати **Docker**.
6. Обрати тариф **Free**.
7. Натиснути **Create Web Service**.
8. Після завершення білду отримати публічний URL.

Після деплою перевірити:

```text
https://your-service-name.onrender.com/health
```

Очікувана відповідь:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Посилання на деплой

Render: https://mlops-lr2.onrender.com

Health check: https://mlops-lr2.onrender.com/health

API docs: https://mlops-lr2.onrender.com/docs

## 10. Приклад `curl`-запиту

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

Для Git Bash / Linux / macOS:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

---

## 11. Контрольні питання

### 1. У чому різниця між Continuous Integration і Continuous Deployment?

**Continuous Integration (CI)** — це практика, коли після кожного push або pull request автоматично запускаються перевірки: встановлення залежностей, тренування моделі, тести, збірка Docker-образу.

**Continuous Deployment (CD)** — це автоматичне розгортання нової версії застосунку після успішного проходження CI. Тобто CI перевіряє, що код працює, а CD доставляє цей код у середовище, доступне користувачам.

---

### 2. Поясніть взаємозв'язок Workflow, Job та Step у GitHub Actions.

**Workflow** — це весь автоматизований процес, описаний у YAML-файлі, наприклад `.github/workflows/ci.yml`.

**Job** — окрема задача всередині workflow. Наприклад, `test` або `docker-build`.

**Step** — конкретний крок всередині job. Наприклад, checkout репозиторію, встановлення Python, запуск `pytest`.

Тобто структура така:

```text
Workflow
└── Job
    └── Step
```

---

### 3. Навіщо модель завантажується у `startup`, а не всередині `predict`?

Модель завантажується один раз при старті API, щоб не читати файл `model.joblib` з диска при кожному запиті.

Якщо виконувати `joblib.load(...)` всередині `predict`, кожен запит буде повільнішим, бо API щоразу виконуватиме дискову операцію. Це збільшить час відповіді та знизить продуктивність сервісу.

---

### 4. Чому у Dockerfile спочатку копіюється `requirements.txt`, а потім код?

Docker кешує шари. Якщо спочатку скопіювати `requirements.txt` і встановити залежності, то при зміні лише коду Docker не буде повторно встановлювати всі бібліотеки.

Це прискорює повторну збірку образу, бо шар із залежностями перевикористовується.

---

### 5. Як FastAPI обробляє некоректний JSON, наприклад `"sepal_length": "abc"`?

FastAPI використовує Pydantic-схему `IrisFeatures`. У ній поле `sepal_length` має тип `float`.

Якщо клієнт надсилає рядок замість числа, Pydantic не пропускає такий запит, і FastAPI автоматично повертає HTTP-код:

```text
422 Unprocessable Entity
```

Це відбувається без ручної перевірки в коді endpoint-у.

---

## 12. Фактичні результати перевірки

Після локального запуску проєкту було отримано такі результати:

| Перевірка | Результат |
|---|---|
| Тренування моделі | `Test accuracy: 0.9667` |
| Запуск тестів | `6 passed, 4 warnings` |
| Локальний запуск API | Swagger UI доступний за `http://127.0.0.1:8000/docs` |
| Endpoint `/health` | `{"status":"healthy","model_loaded":true}` |
| Endpoint `/predict` | модель повертає клас `setosa` |
| Docker-запуск | контейнер успішно запущено на порту `8000` |

Отримані результати підтверджують, що ML-модель навчена, API працює локально, тести проходять успішно, а Docker-контейнер коректно запускає сервіс.

---

## 13. Висновок

У лабораторній роботі реалізовано повний навчальний MLOps-конвеєр. Було створено ML-модель для класифікації Iris, збережено її як артефакт `model.joblib`, розроблено REST API на FastAPI, написано тести для моделі та API, додано Dockerfile для контейнеризації та GitHub Actions workflow для автоматичного запуску перевірок. Проєкт готовий до локального запуску, тестування, контейнеризації та розгортання на Render.
