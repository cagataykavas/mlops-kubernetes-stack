from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

FEATURES = {"features": [0.2, 0.4, 0.8, 0.1]}


@pytest.fixture(scope="module")
def service():
    path = Path(__file__).resolve().parents[1] / "app.py"
    spec = spec_from_file_location("inference_service", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(service, monkeypatch):
    monkeypatch.setattr(service, "MODEL", classifier([[0.25, 0.75]]))
    with TestClient(service.app) as client:
        yield client


def classifier(output, classes=(0, 1)):
    return SimpleNamespace(classes_=np.asarray(classes), predict_proba=lambda _: output)


def counts(service):
    return (
        sum(
            REGISTRY.get_sample_value(
                "ml_predictions_total",
                {"predicted_class": str(label), "model_version": service.MODEL_VERSION},
            )
            or 0
            for label in (0, 1)
        ),
        REGISTRY.get_sample_value("ml_prediction_failures_total"),
        REGISTRY.get_sample_value("ml_prediction_latency_seconds_count"),
    )


@pytest.mark.parametrize(
    ("classes", "output", "probability", "label"),
    [
        ((0, 1), [[0.25, 0.75]], 0.75, 1),
        ((1, 0), [[0.25, 0.75]], 0.25, 0),
        ((0, 1), [[0.5, 0.5]], 0.5, 1),
        ((0, 1), [[1, 0]], 0.0, 0),
        ((0, 1), [[0, 1]], 1.0, 1),
        ((0, 1), [[0.5, 0.5000005]], 0.5000005, 1),
    ],
)
def test_admitted_output(
    client, service, monkeypatch, classes, output, probability, label
):
    monkeypatch.setattr(service, "MODEL", classifier(output, classes))
    before = counts(service)
    response = client.post("/predict", json=FEATURES)
    assert response.status_code == 200
    assert response.json() == {
        "probability": probability,
        "class": label,
        "model_version": service.MODEL_VERSION,
    }
    assert counts(service) == (before[0] + 1, before[1], before[2] + 1)


@pytest.mark.parametrize(
    "output",
    [
        [[float("nan"), 0.5]],
        [[float("inf"), 0.5]],
        [[float("-inf"), 0.5]],
        [[-0.1, 1.1]],
        [[1.1, -0.1]],
        [[0.2, 0.2]],
        [[0.5, 0.500002]],
        [["0.25", "0.75"]],
        [[False, True]],
        [[0.25 + 0j, 0.75 + 0j]],
        [[None, 0.75]],
        [0.25, 0.75],
        [[0.25]],
        [[0.25, 0.5, 0.25]],
        [[0.25, 0.75], [0.5, 0.5]],
        [[[0.25, 0.75]]],
        [],
        None,
        0.75,
    ],
)
def test_invalid_output_is_not_a_success(client, service, monkeypatch, output):
    monkeypatch.setattr(service, "MODEL", classifier(output))
    before = counts(service)
    response = client.post("/predict", json=FEATURES)
    assert response.status_code == 503
    assert response.json() == {"detail": "model inference unavailable"}
    assert counts(service) == (before[0], before[1] + 1, before[2] + 1)
    assert client.get("/health/ready").status_code == 200


@pytest.mark.parametrize(
    "classes", [(0,), (0, 0), (1, 2), ("0", "1"), (False, True), (0.0, 1.0), ((0, 1),)]
)
def test_invalid_class_contract(client, service, monkeypatch, classes):
    def must_not_run(_):
        pytest.fail("inference must not run for an incompatible class contract")

    monkeypatch.setattr(
        service,
        "MODEL",
        SimpleNamespace(classes_=np.asarray(classes), predict_proba=must_not_run),
    )
    assert client.post("/predict", json=FEATURES).status_code == 503


def test_model_exception_is_redacted_and_counted(client, service, monkeypatch):
    def fail(_):
        raise RuntimeError("sensitive internal artifact path and request contents")

    monkeypatch.setattr(service.MODEL, "predict_proba", fail)
    before = counts(service)
    response = client.post("/predict", json=FEATURES)
    assert response.status_code == 503
    assert response.json() == {"detail": "model inference unavailable"}
    assert counts(service) == (before[0], before[1] + 1, before[2] + 1)
    assert "sensitive" not in client.get("/metrics").text


def test_unloaded_model_is_not_an_inference_attempt(client, service, monkeypatch):
    monkeypatch.setattr(service, "MODEL", None)
    before = counts(service)
    assert client.post("/predict", json=FEATURES).status_code == 503
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/live").status_code == 200
    assert counts(service) == before


def test_actual_trained_artifact(client, service, monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "train_model.py"
    spec = spec_from_file_location("training_for_inference", path)
    training = module_from_spec(spec)
    spec.loader.exec_module(training)
    monkeypatch.chdir(tmp_path)
    training.main()
    model = joblib.load(tmp_path / "artifacts/model.joblib")
    monkeypatch.setattr(service, "MODEL", model)
    response = client.post("/predict", json=FEATURES)
    assert response.status_code == 200
    expected = model.predict_proba([FEATURES["features"]])[0, 1]
    assert response.json()["probability"] == pytest.approx(expected)
    assert response.json()["class"] == int(expected >= 0.5)
