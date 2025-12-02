import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))
encoder = pickle.load(open(MODEL_DIR / "encoder.pkl", "rb"))
lb = pickle.load(open(MODEL_DIR / "lb.pkl", "rb"))

cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
def root():
    return {"message": "Welcome!"}


@app.post("/predict")
def predict(input_data: CensusInput):
    df = pd.DataFrame([input_data.dict(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)[0]
    pred_label = lb.inverse_transform(np.array([pred]))[0]
    return {"prediction": pred_label}
