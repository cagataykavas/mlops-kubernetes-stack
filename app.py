from __future__ import annotations

import os
import time
from ipaddress import ip_address
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from drain import DrainController, ServiceDraining

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-dev")
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

PREDICTIONS = Counter(
    "ml_predictions_total",
    "Number of prediction requests",
    labelnames=("predicted_class", "model_version"),
)
LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=(0.001, 0.003, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
MODEL_READY = Gauge("ml_model_ready", "Whether the model artifact is loaded")
MODEL_READY.set(1 if MODEL is not None else 0)
ACCEPTING_PREDICTIONS = Gauge(
    "ml_accepting_predictions",
    "Whether this process accepts new prediction work",
)
ACCEPTING_PREDICTIONS.set(1)
IN_FLIGHT_PREDICTIONS = Gauge(
    "ml_inflight_predictions",
    "Prediction calls admitted and not yet completed",
)
DRAIN = DrainController()

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
    if not DRAIN.snapshot().accepting:
        raise HTTPException(status_code=503, detail="service is draining")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.get("/health")
def health() -> dict:
    drain = DRAIN.snapshot()
    return {
        "status": "ok" if MODEL is not None and drain.accepting else "degraded",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "accepting_predictions": drain.accepting,
        "in_flight_predictions": drain.in_flight,
    }


@app.post("/health/drain")
def begin_drain(
    request: Request,
    timeout_seconds: float = Query(default=20.0, ge=0.0, le=25.0),
) -> dict:
    """Close local prediction admission and wait within the pod shutdown budget."""
    try:
        is_loopback = (
            request.client is not None and ip_address(request.client.host).is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=403, detail="drain control is loopback-only")

    DRAIN.begin_draining()
    ACCEPTING_PREDICTIONS.set(0)
    snapshot = DRAIN.wait_for_idle(timeout_seconds)
    return {
        "status": "drained" if snapshot.drained else "drain_timeout",
        "accepting_predictions": snapshot.accepting,
        "in_flight_predictions": snapshot.in_flight,
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

    try:
        with DRAIN.admit():
            IN_FLIGHT_PREDICTIONS.inc()
            started = time.perf_counter()
            try:
                probability = float(MODEL.predict_proba([payload.features])[0, 1])
                predicted_class = int(probability >= 0.5)
                PREDICTIONS.labels(str(predicted_class), MODEL_VERSION).inc()
                return {
                    "probability": probability,
                    "class": predicted_class,
                    "model_version": MODEL_VERSION,
                }
            finally:
                LATENCY.observe(time.perf_counter() - started)
                IN_FLIGHT_PREDICTIONS.dec()
    except ServiceDraining as exc:
        raise HTTPException(status_code=503, detail="service is draining") from exc
