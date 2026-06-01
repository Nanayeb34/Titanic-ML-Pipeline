# Titanic Survival Predictor

An end-to-end ML pipeline on the Titanic dataset.

## Project Structure

```
├── train_pipeline.py     # Preprocessing, training, evaluation, and model serialization
├── app.py                # FastAPI REST API for serving predictions
└── requirements.txt      # Project dependencies
```

## Pipeline Design

`train_pipeline.py` uses scikit-learn `Pipeline` and `ColumnTransformer` to ensure no data leakage between training and evaluation.

- **Categorical features** — `SimpleImputer(most_frequent)` → `OneHotEncoder`
- **Numerical features** — `SimpleImputer(median)` → `StandardScaler`
- **Classifier** — `RandomForestClassifier(random_state=42)`

All preprocessing is fitted exclusively on training data and applied to the test set at inference time.

**Data cleaning notes:**
- `deck` dropped due to high missingness
- `alive` dropped — it is a string encoding of the target variable and would cause direct data leakage

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train and save the model:**
```bash
python train_pipeline.py
```
Outputs a `classification_report` and ROC-AUC score, then saves the model to `titanic_pipeline.pkl`.

**Start the API:**
```bash
uvicorn app:app --reload
```
API runs on `http://localhost:8000`. Interactive docs available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Returns survival prediction and probability |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "C",
    "who": "woman",
    "adult_male": false,
    "embark_town": "Cherbourg",
    "alone": true,
    "passenger_class": "First"
  }'
```

**Example response:**
```json
{
  "survived": 1,
  "survival_probability": 0.9412
}
```