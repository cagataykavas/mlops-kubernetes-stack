from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from scripts.release_lock import (
    KubectlLeaseBackend,
    LeaseBackendError,
    LeaseCoordinator,
    LeaseEvidenceError,
    LeaseHeldError,
    LeasePolicy,
    LeaseRaceError,
)

from scripts import release_lock

NOW = datetime(2026, 9, 25, 17, 0, tzinfo=UTC)


def lease(
    *,
    holder: str | None = "pipeline-a",
    renew_time: datetime = NOW,
    duration: int = 600,
    version: str = "7",
    transitions: int = 2,
) -> dict:
    spec = {
        "leaseDurationSeconds": duration,
        "acquireTime": "2026-09-25T16:00:00.000000Z",
        "renewTime": renew_time.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "leaseTransitions": transitions,
    }
    if holder is not None:
        spec["holderIdentity"] = holder
    return {
        "apiVersion": "coordination.k8s.io/v1",
        "kind": "Lease",
        "metadata": {
            "name": "model-release",
            "namespace": "ml",
            "resourceVersion": version,
        },
        "spec": spec,
    }


class FakeBackend:
    def __init__(self, current=None):
        self.current = deepcopy(current)
        self.created = []
        self.replaced = []
        self.create_conflicts = 0
        self.replace_conflicts = 0

    def get(self, name):
        assert name == "model-release"
        return deepcopy(self.current)

    def create(self, value):
        if self.create_conflicts:
            self.create_conflicts -= 1
            raise release_lock._Conflict
        self.created.append(deepcopy(value))
        self.current = deepcopy(value)
        self.current["metadata"]["resourceVersion"] = "1"

    def replace(self, value):
        if self.replace_conflicts:
            self.replace_conflicts -= 1
            raise release_lock._Conflict
        self.replaced.append(deepcopy(value))
        self.current = deepcopy(value)
        self.current["metadata"]["resourceVersion"] = str(
            int(self.current["metadata"]["resourceVersion"]) + 1
        )


def coordinator(backend, holder="pipeline-b", policy=None, now=NOW):
    return LeaseCoordinator(
        backend,
        name="model-release",
        namespace="ml",
        holder=holder,
        policy=policy,
        clock=lambda: now,
    )


def test_creates_lease_when_absent():
    backend = FakeBackend()
    lock = coordinator(backend)
    lock.acquire()
    created = backend.created[0]
    assert lock.acquired
    assert created["spec"]["holderIdentity"] == "pipeline-b"
    assert created["spec"]["leaseDurationSeconds"] == 600
    assert created["spec"]["leaseTransitions"] == 0


def test_active_foreign_holder_fails_closed_without_mutation():
    backend = FakeBackend(lease(holder="pipeline-a"))
    with pytest.raises(LeaseHeldError):
        coordinator(backend).acquire()
    assert backend.replaced == []


def test_expired_holder_is_replaced_with_resource_version_preserved():
    expired = lease(renew_time=NOW - timedelta(seconds=601))
    backend = FakeBackend(expired)
    lock = coordinator(backend)
    lock.acquire()
    replacement = backend.replaced[0]
    assert replacement["metadata"]["resourceVersion"] == "7"
    assert replacement["spec"]["holderIdentity"] == "pipeline-b"
    assert replacement["spec"]["leaseTransitions"] == 3
    assert replacement["spec"]["acquireTime"] == "2026-09-25T17:00:00.000000Z"


def test_same_holder_renews_without_transition():
    backend = FakeBackend(lease(holder="pipeline-b", transitions=4))
    lock = coordinator(backend)
    lock.acquire()
    replacement = backend.replaced[0]
    assert replacement["spec"]["leaseTransitions"] == 4
    assert replacement["spec"]["acquireTime"] == "2026-09-25T16:00:00.000000Z"


def test_released_lease_can_be_acquired_and_counts_transition():
    backend = FakeBackend(lease(holder=None, transitions=5))
    coordinator(backend).acquire()
    assert backend.replaced[0]["spec"]["leaseTransitions"] == 6


def test_future_renewal_beyond_skew_budget_is_rejected():
    backend = FakeBackend(lease(renew_time=NOW + timedelta(seconds=31)))
    with pytest.raises(LeaseEvidenceError, match="future-skew"):
        coordinator(backend).acquire()


def test_create_conflict_retries_against_observed_lease():
    class RacingBackend(FakeBackend):
        def create(self, value):
            self.current = lease(holder=None)
            raise release_lock._Conflict

    backend = RacingBackend()
    lock = coordinator(backend)
    lock.acquire()
    assert lock.acquired
    assert len(backend.replaced) == 1


def test_replace_conflicts_exhaust_bounded_attempts():
    backend = FakeBackend(lease(holder=None))
    backend.replace_conflicts = 3
    lock = coordinator(backend, policy=LeasePolicy(max_attempts=3))
    with pytest.raises(LeaseRaceError):
        lock.acquire()
    assert not lock.acquired


def test_release_clears_only_owned_holder():
    backend = FakeBackend()
    lock = coordinator(backend)
    lock.acquire()
    lock.release()
    assert not lock.acquired
    assert "holderIdentity" not in backend.replaced[-1]["spec"]
    assert backend.replaced[-1]["metadata"]["resourceVersion"] == "1"


def test_release_refuses_lost_ownership():
    backend = FakeBackend()
    lock = coordinator(backend)
    lock.acquire()
    backend.current["spec"]["holderIdentity"] = "pipeline-c"
    with pytest.raises(LeaseHeldError, match="ownership"):
        lock.release()


def test_release_conflict_is_not_silently_ignored():
    backend = FakeBackend()
    lock = coordinator(backend)
    lock.acquire()
    backend.replace_conflicts = 1
    with pytest.raises(LeaseRaceError):
        lock.release()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Bad_Name"},
        {"namespace": "Bad_Name"},
        {"holder": "contains spaces"},
        {"policy": LeasePolicy(duration_seconds=29)},
        {"policy": LeasePolicy(duration_seconds=3601)},
        {"policy": LeasePolicy(max_attempts=0)},
        {"policy": LeasePolicy(future_skew_seconds=301)},
    ],
)
def test_invalid_identity_and_policy_are_rejected(kwargs):
    values = {
        "backend": FakeBackend(),
        "name": "model-release",
        "namespace": "ml",
        "holder": "pipeline-b",
        "clock": lambda: NOW,
    }
    values.update(kwargs)
    with pytest.raises(LeaseEvidenceError):
        LeaseCoordinator(**values)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(apiVersion="v1"),
        lambda x: x["metadata"].pop("resourceVersion"),
        lambda x: x["metadata"].update(name="other"),
        lambda x: x["spec"].update(holderIdentity="bad holder"),
        lambda x: x["spec"].update(leaseDurationSeconds=True),
        lambda x: x["spec"].update(renewTime="not-a-time"),
        lambda x: x["spec"].update(leaseTransitions=-1),
    ],
)
def test_malformed_existing_lease_fails_closed(mutate):
    current = lease(holder=None)
    mutate(current)
    with pytest.raises(LeaseEvidenceError):
        coordinator(FakeBackend(current)).acquire()


def test_naive_clock_is_rejected():
    lock = coordinator(FakeBackend(), now=datetime(2026, 9, 25, 17, 0))  # noqa: DTZ001
    with pytest.raises(LeaseEvidenceError, match="timezone-aware"):
        lock.acquire()


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_kubectl_backend_get_create_and_replace_commands():
    calls = []
    responses = [completed(stdout=json.dumps(lease())), completed(), completed()]

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    backend = KubectlLeaseBackend("ml", runner=runner)
    current = backend.get("model-release")
    backend.create(current)
    backend.replace(current)
    assert calls[0][0] == ["kubectl", "get", "lease", "model-release", "-o", "json", "-n", "ml"]
    assert calls[1][0] == ["kubectl", "create", "-f", "-", "-n", "ml"]
    assert calls[2][0] == ["kubectl", "replace", "-f", "-", "-n", "ml"]
    assert json.loads(calls[1][1]["input"])["kind"] == "Lease"


def test_kubectl_backend_classifies_not_found_conflict_and_other_failure():
    backend = KubectlLeaseBackend(
        "ml", runner=lambda *args, **kwargs: completed(1, stderr="NotFound")
    )
    assert backend.get("model-release") is None

    backend = KubectlLeaseBackend(
        "ml", runner=lambda *args, **kwargs: completed(1, stderr="Conflict")
    )
    with pytest.raises(release_lock._Conflict):
        backend.replace(lease())

    backend = KubectlLeaseBackend(
        "ml", runner=lambda *args, **kwargs: completed(1, stderr="forbidden")
    )
    with pytest.raises(LeaseBackendError):
        backend.get("model-release")


def test_kubectl_backend_rejects_duplicate_and_oversized_json():
    duplicate = '{"kind":"Lease","kind":"Lease"}'
    backend = KubectlLeaseBackend("ml", runner=lambda *args, **kwargs: completed(stdout=duplicate))
    with pytest.raises(LeaseEvidenceError, match="duplicate"):
        backend.get("model-release")

    oversized = "{" + (" " * (64 * 1024)) + "}"
    backend = KubectlLeaseBackend("ml", runner=lambda *args, **kwargs: completed(stdout=oversized))
    with pytest.raises(LeaseEvidenceError, match="byte budget"):
        backend.get("model-release")
