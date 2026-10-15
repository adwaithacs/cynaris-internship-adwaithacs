
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="HealthWire FAQ API")

model = joblib.load("healthwire_week2_model_improved.joblib")


class Question(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "HealthWire FAQ API is running"}


@app.post("/predict")
def predict(question: Question):
    prediction = model.predict([question.text])[0]
    return {
        "question": question.text,
        "predicted_intent": prediction
    }
