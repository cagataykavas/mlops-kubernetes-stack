from __future__ import annotations

import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/model.joblib")
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
REQUEST_COUNT = 0
REQUEST_LATENCY_SECONDS = 0.0

app = FastAPI(title="MLOps Kubernetes Demo", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/metrics")
def metrics():
    mean_latency = REQUEST_LATENCY_SECONDS / REQUEST_COUNT if REQUEST_COUNT else 0.0
    return {
        "prediction_requests_total": REQUEST_COUNT,
        "prediction_latency_seconds_mean": mean_latency,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    global REQUEST_COUNT, REQUEST_LATENCY_SECONDS
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model artifact not found; run train_model.py first")
    started = time.perf_counter()
    probability = float(MODEL.predict_proba([payload.features])[0, 1])
    REQUEST_COUNT += 1
    REQUEST_LATENCY_SECONDS += time.perf_counter() - started
    return {"probability": probability, "class": int(probability >= 0.5)}
