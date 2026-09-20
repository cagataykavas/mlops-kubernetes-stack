import json

import pytest

from scripts.canary_gate import (
    CanaryPolicy,
    CanaryWindow,
    evaluate_canary,
    evaluate_file,
)


def window(version, requests=1000, errors=5, latency=100.0):
    return CanaryWindow(version, requests, errors, latency)


def test_promotes_candidate_within_absolute_and_relative_budgets():
    decision = evaluate_canary(
        window("v1", errors=5),
        window("v2", errors=8, latency=115),
        CanaryPolicy(),
    )
    assert decision.promote
    assert decision.reasons == ()
    assert decision.p95_latency_ratio == pytest.approx(1.15)


def test_reports_every_regression_deterministically():
    decision = evaluate_canary(
        window("v1", requests=100, errors=0, latency=50),
        window("v2", requests=100, errors=10, latency=80),
        CanaryPolicy(min_requests=500),
    )
    assert not decision.promote
    assert decision.reasons == tuple(sorted(decision.reasons))
    assert set(decision.reasons) == {
        "candidate_error_rate_exceeded",
        "error_rate_regression",
        "insufficient_baseline_requests",
        "insufficient_candidate_requests",
        "p95_latency_regression",
    }


def test_absolute_error_budget_catches_bad_candidate_when_baseline_is_also_bad():
    decision = evaluate_canary(
        window("v1", errors=100),
        window("v2", errors=90),
        CanaryPolicy(),
    )
    assert decision.reasons == ("candidate_error_rate_exceeded",)


def test_same_version_fails_closed():
    assert evaluate_canary(window("v1"), window("v1"), CanaryPolicy()).reasons == (
        "model_versions_must_differ",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: window("", requests=1),
        lambda: window("v1", requests=0),
        lambda: window("v1", errors=1001),
        lambda: window("v1", latency=float("nan")),
        lambda: CanaryPolicy(min_requests=0),
        lambda: CanaryPolicy(max_error_rate=-1),
        lambda: CanaryPolicy(max_p95_latency_ratio=0.9),
    ],
)
def test_invalid_inputs_fail_closed(factory):
    with pytest.raises(ValueError):
        factory()


def test_file_evaluation_is_json_ready(tmp_path):
    evidence = tmp_path / "canary.json"
    evidence.write_text(
        json.dumps(
            {
                "baseline": {
                    "model_version": "v1",
                    "requests": 1000,
                    "errors": 2,
                    "p95_latency_ms": 100,
                },
                "candidate": {
                    "model_version": "v2",
                    "requests": 1000,
                    "errors": 3,
                    "p95_latency_ms": 105,
                },
            }
        ),
        encoding="utf-8",
    )
    assert evaluate_file(evidence).to_dict()["promote"] is True
