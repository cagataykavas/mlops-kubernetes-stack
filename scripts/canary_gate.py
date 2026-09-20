from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CanaryWindow:
    model_version: str
    requests: int
    errors: int
    p95_latency_ms: float

    def __post_init__(self) -> None:
        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")
        if self.requests < 1:
            raise ValueError("requests must be positive")
        if not 0 <= self.errors <= self.requests:
            raise ValueError("errors must be between zero and requests")
        if not math.isfinite(self.p95_latency_ms) or self.p95_latency_ms <= 0:
            raise ValueError("p95_latency_ms must be finite and positive")

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanaryWindow:
        required = {"model_version", "requests", "errors", "p95_latency_ms"}
        if set(value) != required:
            raise ValueError(f"window keys must be exactly {sorted(required)}")
        if isinstance(value["requests"], bool) or not isinstance(
            value["requests"], int
        ):
            raise TypeError("requests must be an integer")
        if isinstance(value["errors"], bool) or not isinstance(value["errors"], int):
            raise TypeError("errors must be an integer")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CanaryPolicy:
    min_requests: int = 500
    max_error_rate: float = 0.02
    max_error_rate_increase: float = 0.005
    max_p95_latency_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.min_requests < 1:
            raise ValueError("min_requests must be positive")
        rates = (self.max_error_rate, self.max_error_rate_increase)
        if any(not math.isfinite(value) or value < 0 for value in rates):
            raise ValueError("error-rate limits must be finite and non-negative")
        if (
            not math.isfinite(self.max_p95_latency_ratio)
            or self.max_p95_latency_ratio < 1
        ):
            raise ValueError("max_p95_latency_ratio must be finite and at least one")


@dataclass(frozen=True, slots=True)
class CanaryDecision:
    promote: bool
    baseline_version: str
    candidate_version: str
    baseline_error_rate: float
    candidate_error_rate: float
    error_rate_increase: float
    p95_latency_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def evaluate_canary(
    baseline: CanaryWindow, candidate: CanaryWindow, policy: CanaryPolicy
) -> CanaryDecision:
    reasons: list[str] = []
    if baseline.model_version == candidate.model_version:
        reasons.append("model_versions_must_differ")
    if baseline.requests < policy.min_requests:
        reasons.append("insufficient_baseline_requests")
    if candidate.requests < policy.min_requests:
        reasons.append("insufficient_candidate_requests")

    error_increase = candidate.error_rate - baseline.error_rate
    latency_ratio = candidate.p95_latency_ms / baseline.p95_latency_ms
    if candidate.error_rate > policy.max_error_rate:
        reasons.append("candidate_error_rate_exceeded")
    if error_increase > policy.max_error_rate_increase:
        reasons.append("error_rate_regression")
    if latency_ratio > policy.max_p95_latency_ratio:
        reasons.append("p95_latency_regression")

    ordered = tuple(sorted(reasons))
    return CanaryDecision(
        promote=not ordered,
        baseline_version=baseline.model_version,
        candidate_version=candidate.model_version,
        baseline_error_rate=baseline.error_rate,
        candidate_error_rate=candidate.error_rate,
        error_rate_increase=error_increase,
        p95_latency_ratio=latency_ratio,
        reasons=ordered,
    )


def evaluate_file(
    path: str | Path, policy: CanaryPolicy | None = None
) -> CanaryDecision:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load canary evidence: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"baseline", "candidate"}:
        raise ValueError("evidence must contain exactly baseline and candidate")
    return evaluate_canary(
        CanaryWindow.from_dict(value["baseline"]),
        CanaryWindow.from_dict(value["candidate"]),
        policy or CanaryPolicy(),
    )
