# MLOps Lab 1: Wine Quality Classification

**Автор:** Івахненко Юлія  
**Група:** ТР-51мп  
**Дата:** 04.05.2026  

## 1. Опис проєкту

Проєкт виконано в межах лабораторної роботи №1 **«Налаштування MLOps середовища»**.

Мета роботи — налаштувати базове MLOps-середовище для задачі машинного навчання: створити структуру ML-проєкту, підготувати датасет, налаштувати Git, DVC, MLflow, реалізувати `scikit-learn Pipeline`, провести серію експериментів з різними моделями та зберегти найкращу модель.

У роботі використано задачу бінарної класифікації якості червоного вина на основі набору даних **Wine Quality Dataset**.

## 2. Індивідуальний варіант

**Варіант:** 6  
**Датасет:** Wine Quality Dataset  
**Основна модель:** Random Forest Classifier  
**Альтернативні моделі:** Gradient Boosting Classifier, Support Vector Machine  

Виконані експерименти:

- 3 експерименти з Random Forest Classifier;
- 2 експерименти з Gradient Boosting Classifier;
- 2 експерименти з Support Vector Machine.

Усі експерименти логуються в MLflow.

## 3. Структура проєкту

```text
mlops_lab1_wine_quality/
├── .dvc/                         # службові файли DVC
├── data/
│   ├── raw/                       # оригінальні дані
│   │   ├── winequality-red.csv    # Wine Quality dataset
│   │   └── winequality-red.csv.dvc
│   ├── processed/                 # оброблені дані
│   └── external/                  # зовнішні дані
├── models/
│   ├── best_model.pkl             # найкраща модель
│   ├── best_model.pkl.dvc         # DVC-файл моделі
│   └── experiment_results.csv     # таблиця результатів експериментів
├── notebooks/                     # місце для Jupyter notebooks
├── src/
│   ├── create_dataset.py          # підготовка/перевірка датасету
│   ├── pipeline.py                # створення ML Pipeline
│   ├── train.py                   # навчання моделей та MLflow tracking
│   └── utils.py                   # завантаження моделі та predict
├── tests/                         # місце для тестів
├── .gitignore
├── .dvcignore
├── requirements.txt
├── mlflow.db                      # база MLflow Tracking
└── README.md
```

Окремо в архіві додано папку:

```text
dvc-storage/
```

Це локальне DVC remote-сховище, у якому збережені датасет і найкраща модель.

## 4. Датасет

У роботі використано файл:

```text
data/raw/winequality-red.csv
```

Оригінальна цільова колонка — `quality`.

Для задачі класифікації створюється нова цільова змінна:

- `1` — хороше вино, якщо `quality >= 7`;
- `0` — звичайне вино, якщо `quality < 7`.

Ознаки моделі:

- fixed acidity;
- volatile acidity;
- citric acid;
- residual sugar;
- chlorides;
- free sulfur dioxide;
- total sulfur dioxide;
- density;
- pH;
- sulphates;
- alcohol.

## 5. Встановлення

### 5.1. Клонування або розпакування проєкту

Якщо використовується GitHub:

```bash
git clone <repository-url>
cd mlops_lab1_wine_quality
```

Якщо використовується ZIP-архів, потрібно розпакувати його так, щоб структура була такою:

```text
mlops_lab1_wine_quality_completed/
├── dvc-storage/
└── mlops_lab1_wine_quality/
```

Після цього перейти в папку проєкту:

```bash
cd mlops_lab1_wine_quality_completed/mlops_lab1_wine_quality
```

### 5.2. Створення віртуального середовища

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

## 6. DVC: отримання даних і моделі

DVC уже ініціалізований у проєкті. У `.dvc/config` налаштовано локальне remote-сховище:

```text
../dvc-storage
```

Перевірити DVC remote:

```bash
dvc remote list
```

Отримати датасет і модель із DVC storage:

```bash
dvc pull
```

Перевірити статус DVC:

```bash
dvc status
```

Датасет і найкраща модель відстежуються файлами:

```text
data/raw/winequality-red.csv.dvc
models/best_model.pkl.dvc
```

## 7. Git та branching strategy

Для контролю версій використовується Git.

Базові команди ініціалізації:

```bash
git init
git add .
git commit -m "Initial MLOps lab project"
```

Рекомендована стратегія розгалуження:

```text
main          # стабільна версія проєкту
feature/dvc   # налаштування DVC
feature/mlflow # налаштування MLflow tracking
feature/pipeline # реалізація Pipeline та навчання моделей
```

Приклад роботи з feature-гілкою:

```bash
git checkout -b feature/pipeline
git add .
git commit -m "Add sklearn pipelines and training script"
git checkout main
git merge feature/pipeline
```

У Git потрібно зберігати код, README, requirements.txt, `.dvc` метафайли та `.dvc`-файли. Великі файли даних і моделей мають зберігатися через DVC.

## 8. Підготовка датасету

Перевірити наявний датасет:

```bash
python src/create_dataset.py
```

Якщо потрібно скопіювати датасет з іншого місця:

```bash
python src/create_dataset.py --source data/external/winequality-red.csv --force
```

Після додавання або зміни датасету через DVC:

```bash
dvc add data/raw/winequality-red.csv
git add data/raw/winequality-red.csv.dvc data/raw/.gitignore
git commit -m "Add Wine Quality dataset"
dvc push
```

## 9. MLflow Tracking

У проєкті використовується MLflow для логування:

- назви моделі;
- гіперпараметрів;
- train/test метрик;
- cross-validation метрик;
- sklearn Pipeline як артефакту.

Запуск MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Після запуску інтерфейс буде доступний у браузері:

```text
http://127.0.0.1:5000
```

Назва експерименту:

```text
wine-quality-classification
```

## 10. Навчання моделей

Запуск усіх експериментів:

```bash
python src/train.py
```

Скрипт виконує такі дії:

1. Завантажує датасет `data/raw/winequality-red.csv`.
2. Створює цільову змінну `target`.
3. Розділяє дані на train/test.
4. Створює sklearn Pipeline з preprocessing і моделлю.
5. Виконує cross-validation.
6. Навчає моделі.
7. Логує параметри, метрики та модель у MLflow.
8. Зберігає найкращу модель у `models/best_model.pkl`.
9. Зберігає результати в `models/experiment_results.csv`.

## 11. ML Pipeline

У файлі `src/pipeline.py` реалізовано три типи Pipeline:

- `create_random_forest_pipeline()`;
- `create_gradient_boosting_pipeline()`;
- `create_svc_pipeline()`.

Кожен Pipeline містить:

```text
StandardScaler -> Classifier
```

Використання Pipeline дозволяє уникнути data leakage, оскільки масштабування ознак навчається тільки на тренувальних даних під час `fit`.

## 12. Результати експериментів

Результати збережено у файлі:

```text
models/experiment_results.csv
```

| № | Модель | Accuracy | Precision | Recall | F1 | Параметри |
|---|---|---:|---:|---:|---:|---|
| 1 | RandomForest_1 | 0.9031 | 0.9286 | 0.3023 | 0.4561 | `n_estimators=100, max_depth=5` |
| 2 | RandomForest_2 | 0.9375 | 0.8966 | 0.6047 | 0.7222 | `n_estimators=200, max_depth=10` |
| 3 | RandomForest_3 | 0.9375 | 0.8966 | 0.6047 | 0.7222 | `n_estimators=300, max_depth=None` |
| 4 | GradientBoosting_1 | 0.9156 | 0.7857 | 0.5116 | 0.6197 | `n_estimators=100, learning_rate=0.1` |
| 5 | GradientBoosting_2 | 0.9094 | 0.7692 | 0.4651 | 0.5797 | `n_estimators=200, learning_rate=0.05` |
| 6 | SVC_1 | 0.9000 | 0.7619 | 0.3721 | 0.5000 | `C=1.0, kernel=rbf` |
| 7 | SVC_2 | 0.9000 | 0.6774 | 0.4884 | 0.5676 | `C=10.0, kernel=rbf` |

## 13. Найкраща модель

Найкращою моделлю за метрикою Accuracy стала:

```text
RandomForest_2
```

Параметри:

```text
n_estimators = 200
max_depth = 10
```

Метрики:

```text
Accuracy  = 0.9375
Precision = 0.8966
Recall    = 0.6047
F1        = 0.7222
```

Модель збережено у файлі:

```text
models/best_model.pkl
```

Модель також додано під контроль DVC:

```text
models/best_model.pkl.dvc
```

## 14. Використання моделі

Приклад використання з Python:

```python
from src.utils import load_model_from_file, load_sample_data, predict

model = load_model_from_file("models/best_model.pkl")
sample = load_sample_data(rows=5)
predictions = predict(model, sample)
print(predictions)
```

Або запуск готового прикладу:

```bash
python src/utils.py
```

Прогноз:

- `1` — хороше вино;
- `0` — звичайне вино.

## 15. Відтворення експериментів на новій машині

Повний порядок дій:

```bash
# 1. Перейти в папку проєкту
cd mlops_lab1_wine_quality_completed/mlops_lab1_wine_quality

# 2. Створити середовище
python -m venv .venv
.venv\Scripts\activate

# 3. Встановити залежності
pip install -r requirements.txt

# 4. Отримати дані та модель через DVC
dvc pull

# 5. Запустити MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 6. Запустити навчання
python src/train.py

# 7. Перевірити результати
cat models/experiment_results.csv
```

## 16. Troubleshooting

### Проблема: `dvc pull` не знаходить remote storage

Перевірити структуру папок:

```text
mlops_lab1_wine_quality_completed/
├── dvc-storage/
└── mlops_lab1_wine_quality/
```

Перевірити remote:

```bash
dvc remote list
```

За потреби змінити шлях:

```bash
dvc remote modify myremote url ../dvc-storage
```

### Проблема: не знайдено датасет

Виконати:

```bash
dvc pull
```

Або перевірити вручну, що файл існує:

```text
data/raw/winequality-red.csv
```

### Проблема: MLflow UI не відкривається

Перевірити, чи запущено команду:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Після запуску відкрити:

```text
http://127.0.0.1:5000
```

### Проблема: різні результати після повторного запуску

У коді використовується фіксований `random_state = 42`, але незначні відмінності можуть виникати через різні версії бібліотек. Для відтворюваності потрібно використовувати залежності з `requirements.txt`.

## 17. Висновок

У межах лабораторної роботи було створено повну структуру ML-проєкту, налаштовано DVC для версіонування датасету та моделі, реалізовано MLflow Tracking, створено sklearn Pipeline з preprocessing та моделями класифікації, проведено 7 експериментів, порівняно результати та збережено найкращу модель.
