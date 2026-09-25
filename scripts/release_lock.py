"""Kubernetes Lease-based serialization for deployment and rollback operations."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_HOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MAX_LEASE_BYTES = 64 * 1024


class LeaseError(RuntimeError):
    """Base class for stable release-lock failures."""


class LeaseHeldError(LeaseError):
    """Raised when another unexpired holder owns the release lease."""


class LeaseRaceError(LeaseError):
    """Raised after optimistic-concurrency retries are exhausted."""


class LeaseEvidenceError(LeaseError):
    """Raised when the Kubernetes Lease cannot be safely interpreted."""


class LeaseBackendError(LeaseError):
    """Raised for non-concurrency kubectl failures."""


class LeaseBackend(Protocol):
    def get(self, name: str) -> dict[str, Any] | None: ...

    def create(self, lease: dict[str, Any]) -> None: ...

    def replace(self, lease: dict[str, Any]) -> None: ...


class _Conflict(RuntimeError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LeaseEvidenceError("duplicate field in Lease JSON")
        result[key] = value
    return result


def _parse_json(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > _MAX_LEASE_BYTES:
        raise LeaseEvidenceError("Lease JSON exceeds byte budget")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LeaseEvidenceError("invalid Lease JSON") from exc
    if not isinstance(value, dict):
        raise LeaseEvidenceError("Lease JSON root must be an object")
    return value


class KubectlLeaseBackend:
    """Small kubectl adapter using create/replace resourceVersion semantics."""

    def __init__(
        self,
        namespace: str,
        *,
        command: str = "kubectl",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not _DNS_LABEL.fullmatch(namespace):
            raise LeaseEvidenceError("invalid namespace")
        self.namespace = namespace
        self.command = command
        self._runner = runner

    def _run(
        self, *args: str, payload: dict[str, Any] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return self._runner(
            [self.command, *args, "-n", self.namespace],
            input=None if payload is None else json.dumps(payload, separators=(",", ":")),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    def get(self, name: str) -> dict[str, Any] | None:
        result = self._run("get", "lease", name, "-o", "json")
        if result.returncode == 0:
            return _parse_json(result.stdout)
        if "NotFound" in result.stderr:
            return None
        raise LeaseBackendError("kubectl get lease failed")

    def create(self, lease: dict[str, Any]) -> None:
        result = self._run("create", "-f", "-", payload=lease)
        if result.returncode == 0:
            return
        if "AlreadyExists" in result.stderr or "Conflict" in result.stderr:
            raise _Conflict from None
        raise LeaseBackendError("kubectl create lease failed")

    def replace(self, lease: dict[str, Any]) -> None:
        result = self._run("replace", "-f", "-", payload=lease)
        if result.returncode == 0:
            return
        if "Conflict" in result.stderr or "NotFound" in result.stderr:
            raise _Conflict from None
        raise LeaseBackendError("kubectl replace lease failed")


@dataclass(frozen=True)
class LeasePolicy:
    duration_seconds: int = 600
    future_skew_seconds: int = 30
    max_attempts: int = 3

    def validate(self) -> None:
        for field in ("duration_seconds", "future_skew_seconds", "max_attempts"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise LeaseEvidenceError(f"invalid policy field: {field}")
        if not 30 <= self.duration_seconds <= 3600:
            raise LeaseEvidenceError("lease duration must be between 30 and 3600 seconds")
        if not 0 <= self.future_skew_seconds <= 300:
            raise LeaseEvidenceError("future skew must be between 0 and 300 seconds")
        if not 1 <= self.max_attempts <= 10:
            raise LeaseEvidenceError("max_attempts must be between 1 and 10")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LeaseEvidenceError(f"invalid Lease timestamp: {field}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LeaseEvidenceError(f"invalid Lease timestamp: {field}") from exc
    if parsed.tzinfo is None:
        raise LeaseEvidenceError(f"timezone missing from Lease timestamp: {field}")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class _LeaseState:
    resource_version: str
    holder: str | None
    duration_seconds: int
    renew_time: datetime
    transitions: int


def _lease_state(
    lease: dict[str, Any], *, expected_name: str, expected_namespace: str
) -> _LeaseState:
    if lease.get("apiVersion") != "coordination.k8s.io/v1" or lease.get("kind") != "Lease":
        raise LeaseEvidenceError("unexpected Lease type")
    metadata = lease.get("metadata")
    spec = lease.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise LeaseEvidenceError("Lease metadata/spec missing")
    if metadata.get("name") != expected_name or metadata.get("namespace") != expected_namespace:
        raise LeaseEvidenceError("Lease identity mismatch")
    resource_version = metadata.get("resourceVersion")
    if not isinstance(resource_version, str) or not resource_version:
        raise LeaseEvidenceError("Lease resourceVersion missing")
    holder = spec.get("holderIdentity")
    if holder is not None and (not isinstance(holder, str) or not _HOLDER.fullmatch(holder)):
        raise LeaseEvidenceError("invalid Lease holder")
    duration = spec.get("leaseDurationSeconds")
    transitions = spec.get("leaseTransitions", 0)
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 86400:
        raise LeaseEvidenceError("invalid Lease duration")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise LeaseEvidenceError("invalid Lease transition count")
    renew_time = _parse_timestamp(spec.get("renewTime"), "renewTime")
    return _LeaseState(resource_version, holder, duration, renew_time, transitions)


class LeaseCoordinator:
    """Acquire and release one namespaced Lease with optimistic concurrency."""

    def __init__(
        self,
        backend: LeaseBackend,
        *,
        name: str,
        namespace: str,
        holder: str,
        policy: LeasePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not _DNS_LABEL.fullmatch(name) or not _DNS_LABEL.fullmatch(namespace):
            raise LeaseEvidenceError("invalid Lease name or namespace")
        if not _HOLDER.fullmatch(holder):
            raise LeaseEvidenceError("invalid holder identity")
        self.backend = backend
        self.name = name
        self.namespace = namespace
        self.holder = holder
        self.policy = policy or LeasePolicy()
        self.policy.validate()
        self._clock = clock or (lambda: datetime.now(UTC))
        self.acquired = False

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise LeaseEvidenceError("clock must be timezone-aware")
        return value.astimezone(UTC)

    def _new_lease(self, now: datetime) -> dict[str, Any]:
        stamp = _timestamp(now)
        return {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "holderIdentity": self.holder,
                "leaseDurationSeconds": self.policy.duration_seconds,
                "acquireTime": stamp,
                "renewTime": stamp,
                "leaseTransitions": 0,
            },
        }

    def acquire(self) -> None:
        if self.acquired:
            raise LeaseEvidenceError("Lease is already acquired")
        for _ in range(self.policy.max_attempts):
            now = self._now()
            current = self.backend.get(self.name)
            if current is None:
                try:
                    self.backend.create(self._new_lease(now))
                except _Conflict:
                    continue
                self.acquired = True
                return
            state = _lease_state(
                current, expected_name=self.name, expected_namespace=self.namespace
            )
            if state.renew_time > now + timedelta(seconds=self.policy.future_skew_seconds):
                raise LeaseEvidenceError("Lease renewTime exceeds future-skew budget")
            expires_at = state.renew_time + timedelta(seconds=state.duration_seconds)
            if state.holder not in (None, self.holder) and now < expires_at:
                raise LeaseHeldError("release lease is held")
            takeover = state.holder not in (None, self.holder)
            updated = deepcopy(current)
            updated_spec = updated["spec"]
            updated_spec["holderIdentity"] = self.holder
            updated_spec["leaseDurationSeconds"] = self.policy.duration_seconds
            updated_spec["renewTime"] = _timestamp(now)
            if takeover or state.holder is None:
                updated_spec["acquireTime"] = _timestamp(now)
                updated_spec["leaseTransitions"] = state.transitions + 1
            try:
                self.backend.replace(updated)
            except _Conflict:
                continue
            self.acquired = True
            return
        raise LeaseRaceError("release lease changed during acquisition")

    def release(self) -> None:
        if not self.acquired:
            raise LeaseEvidenceError("Lease is not acquired")
        current = self.backend.get(self.name)
        if current is None:
            raise LeaseEvidenceError("acquired Lease disappeared")
        state = _lease_state(current, expected_name=self.name, expected_namespace=self.namespace)
        if state.holder != self.holder:
            raise LeaseHeldError("release lease ownership was lost")
        updated = deepcopy(current)
        updated["spec"].pop("holderIdentity", None)
        updated["spec"]["renewTime"] = _timestamp(self._now())
        try:
            self.backend.replace(updated)
        except _Conflict as exc:
            raise LeaseRaceError("release lease changed during release") from exc
        self.acquired = False
