from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_SPEC = spec_from_file_location("capacity_audit", Path(__file__).parents[1] / "capacity_audit.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CapacityInputError = _MODULE.CapacityInputError
CapacityPolicy = _MODULE.CapacityPolicy
audit_artifact = _MODULE.audit_artifact
load_artifact = _MODULE.load_artifact
main = _MODULE.main

NOW = datetime(2026, 9, 26, 5, 35, tzinfo=UTC)


def _artifact() -> dict:
    return {
        "schema": "mlops-kubernetes-rollout-capacity/v1",
        "generated_at": "2026-09-26T05:34:00Z",
        "workload": {
            "namespace": "ml-production",
            "name": "ml-model-service",
            "template_sha256": "a" * 64,
        },
        "deployment": {
            "desired_replicas": 4,
            "max_surge": "25%",
            "max_unavailable": "25%",
            "pod_requests": {"cpu_millicores": 500, "memory_bytes": 1_073_741_824},
        },
        "autoscaler": {"min_replicas": 2, "max_replicas": 8},
        "disruption_budget": {"mode": "min_available", "value": "75%"},
        "quota": {
            "cpu_request_millicores": 8_000,
            "memory_request_bytes": 17_179_869_184,
            "pod_count": 24,
            "reserved_cpu_millicores": 2_000,
            "reserved_memory_bytes": 4_294_967_296,
            "reserved_pod_count": 6,
        },
    }


def _audit(payload: dict, policy: CapacityPolicy | None = None) -> dict:
    return audit_artifact(payload, policy=policy, now=NOW)


def test_safe_peak_rollout_is_accepted_with_exact_headroom():
    report = _audit(_artifact())
    assert report["accepted"] is True
    assert report["calculations"] == {
        "hpa_peak_replicas": 8,
        "resolved_max_surge": 2,
        "resolved_max_unavailable": 2,
        "rollout_peak_pods": 10,
        "rollout_min_available": 6,
        "disruption_budget_required_available": 6,
        "required_cpu_millicores": 5_000,
        "required_memory_bytes": 10_737_418_240,
        "available_cpu_millicores": 6_000,
        "available_memory_bytes": 12_884_901_888,
        "available_pod_count": 18,
        "cpu_headroom_millicores": 1_000,
        "memory_headroom_bytes": 2_147_483_648,
        "pod_headroom": 8,
    }


def test_percentage_rounding_matches_kubernetes_rollout_semantics():
    payload = _artifact()
    payload["autoscaler"]["max_replicas"] = 7
    payload["disruption_budget"]["value"] = 5
    report = _audit(payload)
    assert report["calculations"]["resolved_max_surge"] == 2
    assert report["calculations"]["resolved_max_unavailable"] == 1
    assert report["calculations"]["rollout_peak_pods"] == 9


def test_exact_percentage_multiple_is_not_rounded_up_by_float_error():
    payload = _artifact()
    payload["autoscaler"]["max_replicas"] = 100
    payload["deployment"]["desired_replicas"] = 100
    payload["deployment"]["max_surge"] = "7%"
    payload["quota"].update(
        cpu_request_millicores=60_000,
        memory_request_bytes=120_000_000_000,
        pod_count=120,
        reserved_cpu_millicores=0,
        reserved_memory_bytes=0,
        reserved_pod_count=0,
    )
    report = _audit(payload)
    assert report["calculations"]["resolved_max_surge"] == 7
    assert report["calculations"]["rollout_peak_pods"] == 107


def test_hpa_peak_not_current_desired_drives_quota_envelope():
    payload = _artifact()
    payload["quota"]["cpu_request_millicores"] = 6_500
    payload["quota"]["reserved_cpu_millicores"] = 2_000
    report = _audit(payload)
    assert report["accepted"] is False
    assert report["reasons"] == ["cpu_request_quota_exceeded"]
    assert report["calculations"]["required_cpu_millicores"] == 5_000


@pytest.mark.parametrize(
    ("field", "reserved_field", "hard", "reserved", "reason"),
    [
        (
            "cpu_request_millicores",
            "reserved_cpu_millicores",
            6_500,
            2_000,
            "cpu_request_quota_exceeded",
        ),
        (
            "memory_request_bytes",
            "reserved_memory_bytes",
            12_000_000_000,
            2_000_000_000,
            "memory_request_quota_exceeded",
        ),
        ("pod_count", "reserved_pod_count", 15, 6, "pod_quota_exceeded"),
    ],
)
def test_each_quota_dimension_fails_independently(field, reserved_field, hard, reserved, reason):
    payload = _artifact()
    payload["quota"][field] = hard
    payload["quota"][reserved_field] = reserved
    report = _audit(payload)
    assert report["accepted"] is False
    assert report["reasons"] == [reason]


def test_all_policy_failures_are_reported_in_stable_order():
    payload = _artifact()
    payload["deployment"]["desired_replicas"] = 1
    payload["deployment"]["max_unavailable"] = "50%"
    payload["disruption_budget"]["value"] = "100%"
    payload["quota"].update(
        cpu_request_millicores=6_500,
        memory_request_bytes=12_000_000_000,
        pod_count=15,
    )
    report = _audit(payload)
    assert report["reasons"] == [
        "desired_replicas_outside_hpa_bounds",
        "rollout_availability_below_disruption_budget",
        "cpu_request_quota_exceeded",
        "memory_request_quota_exceeded",
        "pod_quota_exceeded",
    ]


def test_max_unavailable_pdb_uses_round_up_semantics():
    payload = _artifact()
    payload["autoscaler"]["max_replicas"] = 7
    payload["disruption_budget"] = {"mode": "max_unavailable", "value": "25%"}
    report = _audit(payload)
    assert report["calculations"]["disruption_budget_required_available"] == 5


def test_integer_strategy_values_are_supported():
    payload = _artifact()
    payload["deployment"]["max_surge"] = 3
    payload["deployment"]["max_unavailable"] = 1
    report = _audit(payload)
    assert report["calculations"]["rollout_peak_pods"] == 11
    assert report["calculations"]["rollout_min_available"] == 7


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item["workload"].update(template_sha256="bad"), "template_sha256"),
        (lambda item: item.update(generated_at="2026-09-26T05:34:00"), "timezone"),
        (lambda item: item["autoscaler"].update(min_replicas=9), "must not exceed"),
        (lambda item: item["deployment"].update(max_surge="01%"), "percentage"),
        (lambda item: item["deployment"].update(max_surge=True), "integer or a percentage"),
        (lambda item: item["deployment"].update(max_surge="0%", max_unavailable="0%"), "zero"),
        (lambda item: item["deployment"].update(max_unavailable=9), "must not exceed"),
        (lambda item: item["disruption_budget"].update(mode="either"), "mode"),
        (lambda item: item["disruption_budget"].update(value=9), "more available"),
        (
            lambda item: item["quota"].update(
                reserved_cpu_millicores=item["quota"]["cpu_request_millicores"] + 1
            ),
            "must not exceed",
        ),
        (
            lambda item: item["deployment"]["pod_requests"].update(cpu_millicores=0),
            "cpu_millicores",
        ),
        (lambda item: item.update(extra=True), "must contain exactly"),
    ],
)
def test_malformed_or_inconsistent_evidence_fails_closed(mutation, match):
    payload = _artifact()
    mutation(payload)
    with pytest.raises(CapacityInputError, match=match):
        _audit(payload)


def test_stale_and_future_evidence_is_rejected():
    stale = _artifact()
    stale["generated_at"] = "2026-09-25T05:33:59Z"
    with pytest.raises(CapacityInputError, match="stale"):
        _audit(stale)
    future = _artifact()
    future["generated_at"] = "2026-09-26T05:41:00Z"
    with pytest.raises(CapacityInputError, match="future"):
        _audit(future)


def test_report_is_deterministic_and_omits_raw_workload_identity():
    left = _audit(_artifact())
    right = _audit(deepcopy(_artifact()))
    assert left == right
    encoded = json.dumps(left)
    assert "ml-production" not in encoded
    assert "ml-model-service" not in encoded
    assert len(left["evidence_sha256"]) == 64


def test_loader_rejects_duplicate_nonfinite_and_oversize_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(CapacityInputError, match="duplicate JSON key"):
        load_artifact(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(CapacityInputError, match="non-finite JSON"):
        load_artifact(nonfinite)
    oversize = tmp_path / "oversize.json"
    oversize.write_text("{} ", encoding="utf-8")
    with pytest.raises(CapacityInputError, match="byte budget"):
        load_artifact(oversize, max_bytes=2)


def test_cli_exit_codes_and_atomic_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "capacity_audit.datetime",
        type("FixedDatetime", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}),
    )
    accepted = tmp_path / "accepted.json"
    output = tmp_path / "nested" / "report.json"
    accepted.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert main([str(accepted), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["accepted"] is True
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert list(output.parent.glob(f".{output.name}.*")) == []

    rejected_payload = _artifact()
    rejected_payload["quota"]["pod_count"] = 15
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(rejected_payload), encoding="utf-8")
    assert main([str(rejected)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "rollout_capacity_policy_rejected"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert main([str(malformed)]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "malformed_artifact"


@pytest.mark.parametrize(
    "policy",
    [
        CapacityPolicy(max_age_seconds=True),
        CapacityPolicy(max_artifact_bytes=0),
        CapacityPolicy(max_replicas="many"),  # type: ignore[arg-type]
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(CapacityInputError):
        _audit(_artifact(), policy)
