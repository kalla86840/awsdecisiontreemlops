import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
clf = joblib.load(MODEL_PATH)

class Request(BaseModel):
    features: list  # list of rows

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    X = np.array(req.features)
    return {"predictions": clf.predict(X).tolist()}
