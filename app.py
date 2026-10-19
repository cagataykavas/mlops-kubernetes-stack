from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-dev")
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

PREDICTIONS = Counter(
    "ml_predictions_total",
    "Number of prediction requests",
    labelnames=("predicted_class", "model_version"),
)
PREDICTION_FAILURES = Counter(
    "ml_prediction_failures_total",
    "Inference attempts rejected due to model errors or invalid output",
)
LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=(0.001, 0.003, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
MODEL_READY = Gauge("ml_model_ready", "Whether the model artifact is loaded")
MODEL_READY.set(1 if MODEL is not None else 0)

app = FastAPI(title="MLOps Kubernetes Stack", version="2.0.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


@app.get("/health/live")
def liveness() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model artifact unavailable")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="model artifact not found; run train_model.py first"
        )

    started = time.perf_counter()
    try:
        try:
            probability = validated_probability(MODEL, payload.features)
        except Exception:  # noqa: BLE001 -- contain arbitrary model execution errors at the API boundary
            PREDICTION_FAILURES.inc()
            raise HTTPException(
                status_code=503, detail="model inference unavailable"
            ) from None
        predicted_class = int(probability >= 0.5)
        PREDICTIONS.labels(str(predicted_class), MODEL_VERSION).inc()
        return {
            "probability": probability,
            "class": predicted_class,
            "model_version": MODEL_VERSION,
        }
    finally:
        LATENCY.observe(time.perf_counter() - started)


def validated_probability(model, features: list[float]) -> float:
    """Enforce the binary 0/1 classifier contract before publishing a prediction."""
    classes = np.asarray(model.classes_)
    if (
        classes.shape != (2,)
        or classes.dtype.kind not in "iu"
        or set(classes.tolist()) != {0, 1}
    ):
        raise ValueError("expected integer class labels 0 and 1")

    probabilities = np.asarray(model.predict_proba([features]))
    if (
        probabilities.shape != (1, 2)
        or probabilities.dtype.kind not in "fiu"
        or not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
        or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-6)
    ):
        raise ValueError("expected one normalized binary probability row")
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return float(probabilities[0, positive_column])
