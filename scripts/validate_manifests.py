from __future__ import annotations

import re
from pathlib import Path

import yaml

REQUIRED_KINDS = {"Deployment", "Service", "HorizontalPodAutoscaler", "PodDisruptionBudget", "NetworkPolicy"}
IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def load_documents() -> list[dict]:
    documents: list[dict] = []
    for path in sorted(Path("k8s").glob("*.yaml")):
        documents.extend(doc for doc in yaml.safe_load_all(path.read_text()) if doc)
    return documents


def main() -> None:
    documents = load_documents()
    kinds = {document.get("kind") for document in documents}
    missing = REQUIRED_KINDS - kinds
    if missing:
        raise SystemExit(f"missing required Kubernetes resources: {sorted(missing)}")

    deployment = next(doc for doc in documents if doc.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    if IMAGE_DIGEST.fullmatch(container.get("image", "")) is None:
        raise SystemExit("deployment template image must be digest-pinned")
    probe_paths = {
        container["livenessProbe"]["httpGet"]["path"],
        container["readinessProbe"]["httpGet"]["path"],
        container["startupProbe"]["httpGet"]["path"],
    }
    expected = {"/health/live", "/health/ready"}
    if not expected.issubset(probe_paths):
        raise SystemExit(f"probe paths do not include {sorted(expected)}")

    security = container.get("securityContext", {})
    if security.get("allowPrivilegeEscalation") is not False:
        raise SystemExit("container must disable privilege escalation")
    if security.get("readOnlyRootFilesystem") is not True:
        raise SystemExit("container must use a read-only root filesystem")

    print(f"validated {len(documents)} Kubernetes resources: {sorted(kinds)}")


if __name__ == "__main__":
    main()
