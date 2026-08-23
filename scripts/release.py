from __future__ import annotations

import argparse
import subprocess
import sys


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
        f"deployment/ml-model-service",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Release helper for the demo ML service")
    sub = parser.add_subparsers(dest="command", required=True)

    release = sub.add_parser("deploy")
    release.add_argument("--image", required=True, help="Immutable image reference, preferably digest-pinned")
    release.add_argument("--namespace", default="default")
    release.add_argument("--timeout", default="180s")

    undo = sub.add_parser("rollback")
    undo.add_argument("--namespace", default="default")

    args = parser.parse_args()
    if args.command == "deploy":
        deploy(args.image, args.namespace, args.timeout)
    else:
        rollback(args.namespace)


if __name__ == "__main__":
    main()
