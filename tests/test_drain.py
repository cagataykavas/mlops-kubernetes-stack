from __future__ import annotations

import threading
from importlib import import_module
from pathlib import Path
from sys import path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

path.insert(0, str(Path(__file__).resolve().parents[1]))
api = import_module("app")
drain = import_module("drain")
DrainController = drain.DrainController
ServiceDraining = drain.ServiceDraining


def test_container_copies_drain_runtime() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "COPY train_model.py app.py drain.py ./" in dockerfile


def test_drain_waits_for_admitted_work_and_rejects_new_work() -> None:
    controller = DrainController()
    entered = threading.Event()
    release = threading.Event()

    def active_prediction() -> None:
        with controller.admit():
            entered.set()
            release.wait(timeout=1.0)

    worker = threading.Thread(target=active_prediction)
    worker.start()
    assert entered.wait(timeout=1.0)

    started = controller.begin_draining()
    assert started.accepting is False
    assert started.in_flight == 1
    with pytest.raises(ServiceDraining, match="closed"), controller.admit():
        pass

    timed_out = controller.wait_for_idle(0.01)
    assert timed_out.drained is False
    assert timed_out.in_flight == 1

    release.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert controller.wait_for_idle(0.1).drained is True


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), float("nan"), True])
def test_drain_rejects_invalid_wait_budgets(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        DrainController().wait_for_idle(timeout)


def test_readiness_and_prediction_close_after_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        def predict_proba(self, rows):
            return [[0.75, 0.25] for _ in rows]

    controller = DrainController()
    monkeypatch.setattr(api, "DRAIN", controller)
    monkeypatch.setattr(api, "MODEL", Model())
    controller.begin_draining()

    with pytest.raises(HTTPException) as readiness_error:
        api.readiness()
    assert readiness_error.value.status_code == 503

    with pytest.raises(HTTPException) as prediction_error:
        api.predict(api.PredictionRequest(features=[0.0, 0.0, 0.0, 0.0]))
    assert prediction_error.value.status_code == 503
    assert prediction_error.value.detail == "service is draining"


def test_drain_control_is_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = DrainController()
    monkeypatch.setattr(api, "DRAIN", controller)

    with pytest.raises(HTTPException) as error:
        api.begin_drain(SimpleNamespace(client=SimpleNamespace(host="10.2.0.5")), 0.0)
    assert error.value.status_code == 403
    assert controller.snapshot().accepting is True

    response = api.begin_drain(
        SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
        0.0,
    )
    assert response == {
        "status": "drained",
        "accepting_predictions": False,
        "in_flight_predictions": 0,
    }
