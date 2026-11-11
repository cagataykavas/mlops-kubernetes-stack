from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

IMAGE_PATTERN = re.compile(
    r"^(?P<repository>[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*)@"
    r"(?P<digest>sha256:[0-9a-f]{64})$"
)
NAMESPACE_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
TIMEOUT_PATTERN = re.compile(r"^(?P<value>[1-9][0-9]*)(?P<unit>s|m)$")
IMAGE_DIGEST_ANNOTATION = "mlops.cagataykavas.dev/image-digest"


class ReleaseVerificationError(RuntimeError):
    """Raised when the observed rollout does not match the requested release."""


def run(
    *args: str,
    check: bool = True,
    input_text: str | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args))
    return subprocess.run(
        args,
        check=check,
        text=True,
        input=input_text,
        capture_output=capture_output,
    )


def validate_image(image: str) -> str:
    match = IMAGE_PATTERN.fullmatch(image)
    if match is None:
        raise ValueError("image must be an immutable OCI reference pinned by sha256 digest")
    return match.group("digest")


def validate_namespace(namespace: str) -> None:
    if NAMESPACE_PATTERN.fullmatch(namespace) is None:
        raise ValueError("namespace must be a valid DNS-1123 label")


def validate_timeout(timeout: str) -> None:
    match = TIMEOUT_PATTERN.fullmatch(timeout)
    if match is None:
        raise ValueError("timeout must be between 1s and 30m")
    seconds = int(match.group("value")) * (60 if match.group("unit") == "m" else 1)
    if not 1 <= seconds <= 1_800:
        raise ValueError("timeout must be between 1s and 30m")


def render_deployment_manifest(image: str) -> str:
    digest = validate_image(image)
    source = Path(__file__).resolve().parents[1] / "k8s" / "deployment.yaml"
    documents = [document for document in yaml.safe_load_all(source.read_text()) if document]
    deployments = [document for document in documents if document.get("kind") == "Deployment"]
    if len(deployments) != 1:
        raise ValueError("deployment manifest must contain exactly one Deployment")

    pod_template = deployments[0]["spec"]["template"]
    containers = pod_template["spec"]["containers"]
    api_containers = [container for container in containers if container.get("name") == "api"]
    if len(api_containers) != 1:
        raise ValueError("deployment must contain exactly one api container")
    api_containers[0]["image"] = image
    pod_template.setdefault("metadata", {}).setdefault("annotations", {})[
        IMAGE_DIGEST_ANNOTATION
    ] = digest
    return yaml.safe_dump_all(documents, sort_keys=False)


def verify_deployment_image(image: str, namespace: str) -> None:
    digest = validate_image(image)
    deployed = run(
        "kubectl",
        "get",
        "deployment/ml-model-service",
        "-n",
        namespace,
        "-o",
        'jsonpath={.spec.template.spec.containers[?(@.name=="api")].image}',
        capture_output=True,
    ).stdout.strip()
    if deployed != image:
        raise ReleaseVerificationError(
            f"deployment image mismatch: requested {image!r}, observed {deployed!r}"
        )

    pod_rows = run(
        "kubectl",
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        "app=ml-model-service",
        "-o",
        'jsonpath={range .items[*]}{.metadata.name}{"\\t"}'
        '{.spec.containers[?(@.name=="api")].image}{"\\t"}'
        '{.status.containerStatuses[?(@.name=="api")].imageID}{"\\n"}{end}',
        capture_output=True,
    ).stdout
    matching_pods = 0
    for row in pod_rows.splitlines():
        fields = row.split("\t")
        if len(fields) != 3 or fields[1] != image:
            continue
        matching_pods += 1
        if f"@{digest}" not in fields[2]:
            raise ReleaseVerificationError(
                f"pod {fields[0]!r} image ID does not match requested digest"
            )
    if matching_pods == 0:
        raise ReleaseVerificationError("no running pod reports the requested image reference")


def deploy(image: str, namespace: str, timeout: str) -> None:
    validate_namespace(namespace)
    validate_timeout(timeout)
    manifest = render_deployment_manifest(image)
    root = Path(__file__).resolve().parents[1]
    run("kubectl", "apply", "-n", namespace, "-f", str(root / "k8s" / "resilience.yaml"))
    run("kubectl", "apply", "-n", namespace, "-f", "-", input_text=manifest)
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
        verify_deployment_image(image, namespace)
    except (subprocess.CalledProcessError, ReleaseVerificationError):
        print("rollout failed verification; requesting rollback", file=sys.stderr)
        rollback(namespace)
        raise


def rollback(namespace: str) -> None:
    validate_namespace(namespace)
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
    release.add_argument(
        "--image",
        required=True,
        help="Immutable OCI image reference pinned by sha256 digest",
    )
    release.add_argument("--namespace", default="default")
    release.add_argument("--timeout", default="180s")

    undo = sub.add_parser("rollback")
    undo.add_argument("--namespace", default="default")

    args = parser.parse_args()
    try:
        if args.command == "deploy":
            deploy(args.image, args.namespace, args.timeout)
        else:
            rollback(args.namespace)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
