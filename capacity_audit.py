"""Fail-closed Kubernetes rollout capacity-envelope audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

ARTIFACT_SCHEMA = "mlops-kubernetes-rollout-capacity/v1"
REPORT_SCHEMA = "mlops-kubernetes-rollout-capacity-report/v1"
MAX_ARTIFACT_BYTES = 256 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PERCENT = re.compile(r"^(0|[1-9][0-9]?|100)%$")


class CapacityInputError(ValueError):
    """Raised when rollout-capacity evidence or policy is malformed."""


@dataclass(frozen=True)
class CapacityPolicy:
    max_age_seconds: int = 24 * 60 * 60
    max_future_skew_seconds: int = 300
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES
    max_replicas: int = 10_000
    max_cpu_millicores: int = 100_000_000
    max_memory_bytes: int = 1 << 60
    max_pods: int = 1_000_000

    def validate(self) -> None:
        _integer("max_age_seconds", self.max_age_seconds, 1, 365 * 24 * 60 * 60)
        _integer("max_future_skew_seconds", self.max_future_skew_seconds, 0, 86_400)
        _integer("max_artifact_bytes", self.max_artifact_bytes, 1, 4 * 1024 * 1024)
        _integer("max_replicas", self.max_replicas, 1, 1_000_000)
        _integer("max_cpu_millicores", self.max_cpu_millicores, 1, 10**12)
        _integer("max_memory_bytes", self.max_memory_bytes, 1, 1 << 63)
        _integer("max_pods", self.max_pods, 1, 100_000_000)


def _fail(message: str) -> NoReturn:
    raise CapacityInputError(message)


def _integer(name: str, value: object, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        _fail(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _object(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{name} keys must be strings")
    return value


def _identifier(value: object, name: str, *, max_length: int = 253) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        _fail(f"{name} must be a non-empty string of at most {max_length} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(f"{name} must not contain control characters")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _fail(f"{name} must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CapacityInputError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapacityInputError("artifact must be canonical JSON with finite values") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _int_or_percent(value: object, name: str, *, upper: int) -> int | str:
    if isinstance(value, bool):
        _fail(f"{name} must be an integer or a percentage")
    if isinstance(value, int):
        return _integer(name, value, 0, upper)
    if isinstance(value, str) and _PERCENT.fullmatch(value) is not None:
        return value
    _fail(f"{name} must be an integer or percentage in [0%, 100%]")


def _resolve(value: int | str, replicas: int, *, round_up: bool) -> int:
    if isinstance(value, int):
        return value
    numerator = replicas * int(value[:-1])
    return (numerator + 99) // 100 if round_up else numerator // 100


def _parse_requests(value: object, policy: CapacityPolicy) -> dict[str, int]:
    requests = _object(value, "deployment.pod_requests", {"cpu_millicores", "memory_bytes"})
    return {
        "cpu_millicores": _integer(
            "deployment.pod_requests.cpu_millicores",
            requests["cpu_millicores"],
            1,
            policy.max_cpu_millicores,
        ),
        "memory_bytes": _integer(
            "deployment.pod_requests.memory_bytes",
            requests["memory_bytes"],
            1,
            policy.max_memory_bytes,
        ),
    }


def _parse_quota(value: object, policy: CapacityPolicy) -> dict[str, int]:
    quota = _object(
        value,
        "quota",
        {
            "cpu_request_millicores",
            "memory_request_bytes",
            "pod_count",
            "reserved_cpu_millicores",
            "reserved_memory_bytes",
            "reserved_pod_count",
        },
    )
    result = {
        "cpu_request_millicores": _integer(
            "quota.cpu_request_millicores",
            quota["cpu_request_millicores"],
            1,
            policy.max_cpu_millicores,
        ),
        "memory_request_bytes": _integer(
            "quota.memory_request_bytes",
            quota["memory_request_bytes"],
            1,
            policy.max_memory_bytes,
        ),
        "pod_count": _integer("quota.pod_count", quota["pod_count"], 1, policy.max_pods),
        "reserved_cpu_millicores": _integer(
            "quota.reserved_cpu_millicores",
            quota["reserved_cpu_millicores"],
            0,
            policy.max_cpu_millicores,
        ),
        "reserved_memory_bytes": _integer(
            "quota.reserved_memory_bytes",
            quota["reserved_memory_bytes"],
            0,
            policy.max_memory_bytes,
        ),
        "reserved_pod_count": _integer(
            "quota.reserved_pod_count", quota["reserved_pod_count"], 0, policy.max_pods
        ),
    }
    pairs = (
        ("reserved_cpu_millicores", "cpu_request_millicores"),
        ("reserved_memory_bytes", "memory_request_bytes"),
        ("reserved_pod_count", "pod_count"),
    )
    for used_name, hard_name in pairs:
        if result[used_name] > result[hard_name]:
            _fail(f"quota.{used_name} must not exceed quota.{hard_name}")
    return result


def _pdb_required_available(
    value: object, replicas: int, policy: CapacityPolicy
) -> tuple[int, str, int | str]:
    budget = _object(value, "disruption_budget", {"mode", "value"})
    mode = budget["mode"]
    if mode not in {"min_available", "max_unavailable"}:
        _fail("disruption_budget.mode must be min_available or max_unavailable")
    configured = _int_or_percent(
        budget["value"], "disruption_budget.value", upper=policy.max_replicas
    )
    if mode == "min_available":
        required = _resolve(configured, replicas, round_up=True)
    else:
        allowed_unavailable = _resolve(configured, replicas, round_up=True)
        required = max(0, replicas - allowed_unavailable)
    if required > replicas:
        _fail("disruption budget requires more available pods than the HPA maximum")
    return required, str(mode), configured


def audit_artifact(
    payload: object,
    *,
    policy: CapacityPolicy | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate and evaluate one deployment/HPA/PDB/quota capacity envelope."""
    active_policy = policy or CapacityPolicy()
    active_policy.validate()
    artifact_bytes = _canonical_bytes(payload)
    if len(artifact_bytes) > active_policy.max_artifact_bytes:
        _fail("artifact exceeds max_artifact_bytes")

    root = _object(
        payload,
        "artifact",
        {
            "schema",
            "generated_at",
            "workload",
            "deployment",
            "autoscaler",
            "disruption_budget",
            "quota",
        },
    )
    if root["schema"] != ARTIFACT_SCHEMA:
        _fail(f"schema must equal {ARTIFACT_SCHEMA}")
    generated_at = _timestamp(root["generated_at"], "generated_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age_seconds = (current - generated_at).total_seconds()
    if age_seconds > active_policy.max_age_seconds:
        _fail("artifact is stale")
    if age_seconds < -active_policy.max_future_skew_seconds:
        _fail("artifact timestamp is too far in the future")

    workload = _object(root["workload"], "workload", {"namespace", "name", "template_sha256"})
    namespace = _identifier(workload["namespace"], "workload.namespace")
    name = _identifier(workload["name"], "workload.name")
    template_digest = _digest(workload["template_sha256"], "workload.template_sha256")

    deployment = _object(
        root["deployment"],
        "deployment",
        {"desired_replicas", "max_surge", "max_unavailable", "pod_requests"},
    )
    desired = _integer(
        "deployment.desired_replicas", deployment["desired_replicas"], 1, active_policy.max_replicas
    )
    max_surge = _int_or_percent(
        deployment["max_surge"], "deployment.max_surge", upper=active_policy.max_replicas
    )
    max_unavailable = _int_or_percent(
        deployment["max_unavailable"],
        "deployment.max_unavailable",
        upper=active_policy.max_replicas,
    )
    requests = _parse_requests(deployment["pod_requests"], active_policy)

    autoscaler = _object(root["autoscaler"], "autoscaler", {"min_replicas", "max_replicas"})
    hpa_min = _integer(
        "autoscaler.min_replicas", autoscaler["min_replicas"], 1, active_policy.max_replicas
    )
    hpa_max = _integer(
        "autoscaler.max_replicas", autoscaler["max_replicas"], 1, active_policy.max_replicas
    )
    if hpa_min > hpa_max:
        _fail("autoscaler.min_replicas must not exceed autoscaler.max_replicas")

    surge_at_peak = _resolve(max_surge, hpa_max, round_up=True)
    unavailable_at_peak = _resolve(max_unavailable, hpa_max, round_up=False)
    if surge_at_peak == 0 and unavailable_at_peak == 0:
        _fail("max_surge and max_unavailable cannot both resolve to zero")
    if unavailable_at_peak > hpa_max:
        _fail("max_unavailable must not exceed the HPA maximum")

    pdb_required, pdb_mode, pdb_value = _pdb_required_available(
        root["disruption_budget"], hpa_max, active_policy
    )
    quota = _parse_quota(root["quota"], active_policy)

    peak_pods = hpa_max + surge_at_peak
    rollout_min_available = hpa_max - unavailable_at_peak
    required_cpu = peak_pods * requests["cpu_millicores"]
    required_memory = peak_pods * requests["memory_bytes"]
    available_cpu = quota["cpu_request_millicores"] - quota["reserved_cpu_millicores"]
    available_memory = quota["memory_request_bytes"] - quota["reserved_memory_bytes"]
    available_pods = quota["pod_count"] - quota["reserved_pod_count"]

    reasons: list[str] = []
    if not hpa_min <= desired <= hpa_max:
        reasons.append("desired_replicas_outside_hpa_bounds")
    if rollout_min_available < pdb_required:
        reasons.append("rollout_availability_below_disruption_budget")
    if required_cpu > available_cpu:
        reasons.append("cpu_request_quota_exceeded")
    if required_memory > available_memory:
        reasons.append("memory_request_quota_exceeded")
    if peak_pods > available_pods:
        reasons.append("pod_quota_exceeded")

    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    policy_payload = asdict(active_policy)
    policy_sha256 = _sha256_json(policy_payload)
    calculations = {
        "hpa_peak_replicas": hpa_max,
        "resolved_max_surge": surge_at_peak,
        "resolved_max_unavailable": unavailable_at_peak,
        "rollout_peak_pods": peak_pods,
        "rollout_min_available": rollout_min_available,
        "disruption_budget_required_available": pdb_required,
        "required_cpu_millicores": required_cpu,
        "required_memory_bytes": required_memory,
        "available_cpu_millicores": available_cpu,
        "available_memory_bytes": available_memory,
        "available_pod_count": available_pods,
        "cpu_headroom_millicores": available_cpu - required_cpu,
        "memory_headroom_bytes": available_memory - required_memory,
        "pod_headroom": available_pods - peak_pods,
    }
    evidence = {
        "artifact_sha256": artifact_sha256,
        "policy_sha256": policy_sha256,
        "reasons": reasons,
        "calculations": calculations,
    }
    return {
        "schema": REPORT_SCHEMA,
        "accepted": not reasons,
        "reason": "accepted" if not reasons else "rollout_capacity_policy_rejected",
        "identity": {
            "namespace_sha256": _sha256_text(namespace),
            "workload_sha256": _sha256_text(name),
            "template_sha256": template_digest,
        },
        "strategy": {
            "max_surge": max_surge,
            "max_unavailable": max_unavailable,
            "disruption_budget_mode": pdb_mode,
            "disruption_budget_value": pdb_value,
        },
        "calculations": calculations,
        "reasons": reasons,
        "policy": policy_payload,
        "artifact_sha256": artifact_sha256,
        "policy_sha256": policy_sha256,
        "evidence_sha256": _sha256_json(evidence),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CapacityInputError("unable to read artifact") from exc
    if len(raw) > max_bytes:
        _fail("artifact exceeds the input byte budget")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _fail(f"non-finite JSON value: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapacityInputError("artifact must be valid UTF-8 JSON") from exc


def write_report_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        artifact = load_artifact(args.artifact)
        report = audit_artifact(artifact)
    except CapacityInputError as exc:
        print(json.dumps({"accepted": False, "reason": "malformed_artifact", "error": str(exc)}))
        return 3
    if args.output is not None:
        write_report_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
