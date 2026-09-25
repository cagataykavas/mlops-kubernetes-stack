from __future__ import annotations

import sys
from typing import ClassVar

import pytest

from scripts import release, release_lock

LeaseHeldError = release_lock.LeaseHeldError


class FakeCoordinator:
    events: ClassVar[list[str]] = []

    def __init__(self, backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def acquire(self):
        self.events.append("acquire")

    def release(self):
        self.events.append("release")


def test_locked_action_acquires_runs_and_releases(monkeypatch):
    events = FakeCoordinator.events = []
    monkeypatch.setattr(release, "KubectlLeaseBackend", lambda namespace: f"backend:{namespace}")
    monkeypatch.setattr(release, "LeaseCoordinator", FakeCoordinator)
    release._locked("ml", "release-lock", "pipeline-1", 600, lambda: events.append("action"))
    assert events == ["acquire", "action", "release"]


def test_locked_action_releases_when_rollout_raises(monkeypatch):
    events = FakeCoordinator.events = []
    monkeypatch.setattr(release, "KubectlLeaseBackend", lambda namespace: object())
    monkeypatch.setattr(release, "LeaseCoordinator", FakeCoordinator)

    def fail():
        events.append("action")
        raise RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed"):
        release._locked("ml", "release-lock", "pipeline-1", 600, fail)
    assert events == ["acquire", "action", "release"]


def test_explicit_and_environment_holder_precedence(monkeypatch):
    monkeypatch.setenv("RELEASE_LOCK_HOLDER", "environment-holder")
    assert release._holder_identity("explicit-holder") == "explicit-holder"
    assert release._holder_identity(None) == "environment-holder"


def test_generated_holder_is_bounded_and_safe(monkeypatch):
    monkeypatch.delenv("RELEASE_LOCK_HOLDER", raising=False)
    monkeypatch.setattr(release.socket, "gethostname", lambda: "Host_Name With Unsafe Characters")
    monkeypatch.setattr(release.os, "getpid", lambda: 123)
    holder = release._holder_identity(None)
    assert holder.startswith("local-")
    assert len(holder) == 22
    assert "Host_Name" not in holder


def test_main_wraps_deploy_in_lock(monkeypatch):
    captured = {}

    def fake_locked(namespace, lock_name, holder, lease_seconds, action):
        captured.update(
            namespace=namespace,
            lock_name=lock_name,
            holder=holder,
            lease_seconds=lease_seconds,
        )
        action()

    monkeypatch.setattr(release, "_locked", fake_locked)
    monkeypatch.setattr(
        release,
        "deploy",
        lambda image, namespace, timeout: captured.update(
            image=image, deploy_namespace=namespace, timeout=timeout
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release.py",
            "deploy",
            "--image",
            "registry/image@sha256:abc",
            "--namespace",
            "ml",
            "--lock-holder",
            "pipeline-9",
            "--lease-seconds",
            "900",
        ],
    )
    assert release.main() == 0
    assert captured == {
        "namespace": "ml",
        "lock_name": "ml-model-service-release",
        "holder": "pipeline-9",
        "lease_seconds": 900,
        "image": "registry/image@sha256:abc",
        "deploy_namespace": "ml",
        "timeout": "180s",
    }


def test_main_returns_stable_lock_exit_without_holder_leak(monkeypatch, capsys):
    def held(*args, **kwargs):
        raise LeaseHeldError("secret-foreign-holder")

    monkeypatch.setattr(release, "_locked", held)
    monkeypatch.setattr(
        sys,
        "argv",
        ["release.py", "rollback", "--namespace", "ml", "--lock-holder", "pipeline-9"],
    )
    assert release.main() == 4
    error = capsys.readouterr().err
    assert "LeaseHeldError" in error
    assert "secret-foreign-holder" not in error
