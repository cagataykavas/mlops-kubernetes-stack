from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys

if __package__:
    from .release_lock import (
        KubectlLeaseBackend,
        LeaseCoordinator,
        LeaseError,
        LeasePolicy,
    )
else:
    from release_lock import (
        KubectlLeaseBackend,
        LeaseCoordinator,
        LeaseError,
        LeasePolicy,
    )


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args))
    return subprocess.run(args, check=check, text=True, capture_output=False)


def deploy(image: str, namespace: str, timeout: str) -> None:
    run("kubectl", "apply", "-n", namespace, "-f", "k8s/deployment.yaml")
    run("kubectl", "apply", "-n", namespace, "-f", "k8s/resilience.yaml")
    run(
        "kubectl",
        "set",
        "image",
        "deployment/ml-model-service",
        f"api={image}",
        "-n",
        namespace,
        "--record=false",
    )
    try:
        run(
            "kubectl",
            "rollout",
            "status",
            "deployment/ml-model-service",
            "-n",
            namespace,
            f"--timeout={timeout}",
        )
    except subprocess.CalledProcessError:
        print("rollout failed; requesting rollback", file=sys.stderr)
        rollback(namespace)
        raise


def rollback(namespace: str) -> None:
    run(
        "kubectl",
        "rollout",
        "undo",
        "deployment/ml-model-service",
        "-n",
        namespace,
    )
    run(
        "kubectl",
        "rollout",
        "status",
        "deployment/ml-model-service",
        "-n",
        namespace,
        "--timeout=120s",
    )


def _holder_identity(explicit: str | None) -> str:
    configured = explicit or os.environ.get("RELEASE_LOCK_HOLDER")
    if configured:
        return configured
    local_identity = f"{socket.gethostname()}:{os.getpid()}"
    return f"local-{hashlib.sha256(local_identity.encode()).hexdigest()[:16]}"


def _add_lock_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock-name", default="ml-model-service-release")
    parser.add_argument("--lock-holder")
    parser.add_argument("--lease-seconds", type=int, default=600)


def _locked(namespace: str, lock_name: str, holder: str | None, lease_seconds: int, action) -> None:
    coordinator = LeaseCoordinator(
        KubectlLeaseBackend(namespace),
        name=lock_name,
        namespace=namespace,
        holder=_holder_identity(holder),
        policy=LeasePolicy(duration_seconds=lease_seconds),
    )
    coordinator.acquire()
    try:
        action()
    finally:
        coordinator.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Release helper for the demo ML service")
    sub = parser.add_subparsers(dest="command", required=True)

    release = sub.add_parser("deploy")
    release.add_argument(
        "--image", required=True, help="Immutable image reference, preferably digest-pinned"
    )
    release.add_argument("--namespace", default="default")
    release.add_argument("--timeout", default="180s")
    _add_lock_arguments(release)

    undo = sub.add_parser("rollback")
    undo.add_argument("--namespace", default="default")
    _add_lock_arguments(undo)

    args = parser.parse_args()
    try:
        if args.command == "deploy":
            _locked(
                args.namespace,
                args.lock_name,
                args.lock_holder,
                args.lease_seconds,
                lambda: deploy(args.image, args.namespace, args.timeout),
            )
        else:
            _locked(
                args.namespace,
                args.lock_name,
                args.lock_holder,
                args.lease_seconds,
                lambda: rollback(args.namespace),
            )
    except LeaseError as exc:
        print(f"release lock rejected: {type(exc).__name__}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
