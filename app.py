import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Titanic Survival Predictor")

pipeline = joblib.load("titanic_pipeline.pkl")


class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: Optional[str] = None
    who: str
    adult_male: bool
    embark_town: Optional[str] = None
    alone: bool
    passenger_class: Optional[str] = None

    class Config:
        populate_by_name = True


@app.get("/")
def root():
    return {"message": "Titanic Survival Predictor API. POST to /predict."}


@app.post("/predict")
def predict(passenger: Passenger):
    data = passenger.model_dump()

    data["class"] = data.pop("passenger_class")

    df = pd.DataFrame([data])

    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])

    return {
        "survived": prediction,
        "survival_probability": round(probability, 4),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
