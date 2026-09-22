from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


class ManifestInputError(ValueError):
    """Raised when manifest evidence cannot be evaluated safely."""


@dataclass(frozen=True)
class AdmissionPolicy:
    min_replicas: int = 2
    min_termination_grace_seconds: int = 20
    require_startup_probe: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.min_replicas, bool) or self.min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        if (
            isinstance(self.min_termination_grace_seconds, bool)
            or self.min_termination_grace_seconds < 1
        ):
            raise ValueError("min_termination_grace_seconds must be positive")


@dataclass(frozen=True, order=True)
class PolicyViolation:
    code: str
    resource: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AdmissionReport:
    accepted: bool
    document_count: int
    deployment_count: int
    violations: tuple[PolicyViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "document_count": self.document_count,
            "deployment_count": self.deployment_count,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def load_documents(paths: Iterable[Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            loaded = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ManifestInputError(f"cannot load {path}: {exc}") from exc
        for index, document in enumerate(loaded, start=1):
            if document is None:
                continue
            if not isinstance(document, dict):
                raise ManifestInputError(f"{path} document {index} must be a mapping")
            documents.append(document)
    if not documents:
        raise ManifestInputError("no Kubernetes manifest documents found")
    return documents


def _resource_id(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata")
    name = metadata.get("name") if isinstance(metadata, Mapping) else None
    kind = document.get("kind")
    return f"{kind or '<missing-kind>'}/{name or '<missing-name>'}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else ()


def _selector_matches(selector: Mapping[str, Any], labels: Mapping[str, Any]) -> bool:
    return bool(selector) and all(labels.get(key) == value for key, value in selector.items())


def _zero_unavailable(value: Any) -> bool:
    return value == 0 or (isinstance(value, str) and value in {"0", "0%"})


def _has_explicit_non_latest_image(image: Any) -> bool:
    if not isinstance(image, str) or not image.strip():
        return False
    image = image.strip()
    if "@sha256:" in image:
        algorithm, _, digest = image.rpartition("@sha256:")
        return (
            bool(algorithm)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        )
    final_component = image.rsplit("/", 1)[-1]
    return ":" in final_component and not final_component.endswith(":latest")


def evaluate_documents(
    documents: Sequence[Mapping[str, Any]],
    policy: AdmissionPolicy = AdmissionPolicy(),
) -> AdmissionReport:
    if not documents:
        raise ManifestInputError("documents must not be empty")

    violations: list[PolicyViolation] = []
    identities: set[tuple[str, str, str]] = set()
    indexed: dict[str, list[Mapping[str, Any]]] = {}

    def reject(code: str, resource: str, path: str, message: str) -> None:
        violations.append(PolicyViolation(code, resource, path, message))

    for document in documents:
        if not isinstance(document, Mapping):
            raise ManifestInputError("every document must be a mapping")
        resource = _resource_id(document)
        kind = document.get("kind")
        api_version = document.get("apiVersion")
        metadata = _mapping(document.get("metadata"))
        name = metadata.get("name")
        namespace = metadata.get("namespace", "default")
        if not isinstance(api_version, str) or not api_version:
            reject("missing_api_version", resource, "apiVersion", "apiVersion is required")
        if not isinstance(kind, str) or not kind:
            reject("missing_kind", resource, "kind", "kind is required")
            continue
        if not isinstance(name, str) or not name:
            reject("missing_resource_name", resource, "metadata.name", "metadata.name is required")
            continue
        if not isinstance(namespace, str) or not namespace:
            reject(
                "invalid_namespace", resource, "metadata.namespace", "namespace must be non-empty"
            )
            continue
        identity = (kind, namespace, name)
        if identity in identities:
            reject("duplicate_resource", resource, "metadata", "resource identity must be unique")
        identities.add(identity)
        indexed.setdefault(kind, []).append(document)

    deployments = indexed.get("Deployment", [])
    if not deployments:
        reject("missing_deployment", "cluster", "kind", "at least one Deployment is required")

    deployment_labels: dict[str, Mapping[str, Any]] = {}
    deployment_replicas: dict[str, int] = {}
    for deployment in deployments:
        resource = _resource_id(deployment)
        name = str(_mapping(deployment.get("metadata")).get("name", ""))
        spec = _mapping(deployment.get("spec"))
        replicas = spec.get("replicas")
        if (
            not isinstance(replicas, int)
            or isinstance(replicas, bool)
            or replicas < policy.min_replicas
        ):
            reject(
                "insufficient_replicas",
                resource,
                "spec.replicas",
                f"replicas must be an integer >= {policy.min_replicas}",
            )
        else:
            deployment_replicas[name] = replicas

        selector = _mapping(_mapping(spec.get("selector")).get("matchLabels"))
        template = _mapping(spec.get("template"))
        template_labels = _mapping(_mapping(template.get("metadata")).get("labels"))
        if not _selector_matches(selector, template_labels):
            reject(
                "deployment_selector_mismatch",
                resource,
                "spec.selector.matchLabels",
                "selector must be non-empty and match pod-template labels",
            )
        deployment_labels[name] = template_labels

        strategy = _mapping(spec.get("strategy"))
        rolling = _mapping(strategy.get("rollingUpdate"))
        if strategy.get("type") != "RollingUpdate" or not _zero_unavailable(
            rolling.get("maxUnavailable")
        ):
            reject(
                "unsafe_rollout_strategy",
                resource,
                "spec.strategy",
                "RollingUpdate with maxUnavailable=0 is required",
            )

        pod_spec = _mapping(template.get("spec"))
        pod_security = _mapping(pod_spec.get("securityContext"))
        if pod_security.get("runAsNonRoot") is not True:
            reject(
                "pod_may_run_as_root",
                resource,
                "spec.template.spec.securityContext.runAsNonRoot",
                "must be true",
            )
        if _mapping(pod_security.get("seccompProfile")).get("type") != "RuntimeDefault":
            reject(
                "missing_seccomp_profile",
                resource,
                "spec.template.spec.securityContext.seccompProfile.type",
                "RuntimeDefault is required",
            )
        grace = pod_spec.get("terminationGracePeriodSeconds")
        if (
            not isinstance(grace, int)
            or isinstance(grace, bool)
            or grace < policy.min_termination_grace_seconds
        ):
            reject(
                "insufficient_termination_grace",
                resource,
                "spec.template.spec.terminationGracePeriodSeconds",
                f"must be an integer >= {policy.min_termination_grace_seconds}",
            )

        containers = _sequence(pod_spec.get("containers"))
        if not containers:
            reject(
                "missing_containers",
                resource,
                "spec.template.spec.containers",
                "at least one container is required",
            )
        names: set[str] = set()
        for index, value in enumerate(containers):
            container = _mapping(value)
            prefix = f"spec.template.spec.containers[{index}]"
            container_name = container.get("name")
            if not isinstance(container_name, str) or not container_name or container_name in names:
                reject(
                    "invalid_container_name",
                    resource,
                    f"{prefix}.name",
                    "container names must be non-empty and unique",
                )
            else:
                names.add(container_name)
            if not _has_explicit_non_latest_image(container.get("image")):
                reject(
                    "mutable_or_invalid_image",
                    resource,
                    f"{prefix}.image",
                    "use an explicit non-latest tag or a valid sha256 digest",
                )
            security = _mapping(container.get("securityContext"))
            if security.get("allowPrivilegeEscalation") is not False:
                reject(
                    "privilege_escalation_allowed",
                    resource,
                    f"{prefix}.securityContext.allowPrivilegeEscalation",
                    "must be false",
                )
            if security.get("readOnlyRootFilesystem") is not True:
                reject(
                    "writable_root_filesystem",
                    resource,
                    f"{prefix}.securityContext.readOnlyRootFilesystem",
                    "must be true",
                )
            dropped = _sequence(_mapping(security.get("capabilities")).get("drop"))
            if "ALL" not in dropped:
                reject(
                    "linux_capabilities_not_dropped",
                    resource,
                    f"{prefix}.securityContext.capabilities.drop",
                    "must include ALL",
                )
            resources = _mapping(container.get("resources"))
            for budget in ("requests", "limits"):
                values = _mapping(resources.get(budget))
                for dimension in ("cpu", "memory"):
                    if not isinstance(values.get(dimension), str) or not values.get(dimension):
                        reject(
                            "missing_resource_budget",
                            resource,
                            f"{prefix}.resources.{budget}.{dimension}",
                            f"{budget}.{dimension} is required",
                        )
            required_probes = ["readinessProbe", "livenessProbe"]
            if policy.require_startup_probe:
                required_probes.append("startupProbe")
            for probe_name in required_probes:
                probe = _mapping(container.get(probe_name))
                http_get = _mapping(probe.get("httpGet"))
                if not http_get.get("path") or http_get.get("port") is None:
                    reject(
                        "missing_http_probe",
                        resource,
                        f"{prefix}.{probe_name}.httpGet",
                        "HTTP path and port are required",
                    )

    def targets_any_deployment(selector: Mapping[str, Any]) -> bool:
        return any(_selector_matches(selector, labels) for labels in deployment_labels.values())

    service_labels: dict[str, Mapping[str, Any]] = {}
    service_ports: dict[str, set[str]] = {}
    for service in indexed.get("Service", []):
        service_name = str(_mapping(service.get("metadata")).get("name", ""))
        service_labels[service_name] = _mapping(_mapping(service.get("metadata")).get("labels"))
        service_spec = _mapping(service.get("spec"))
        selector = _mapping(service_spec.get("selector"))
        if not targets_any_deployment(selector):
            reject(
                "orphan_service",
                _resource_id(service),
                "spec.selector",
                "selector must target a Deployment",
            )
        service_ports[service_name] = {
            str(port.get("name"))
            for value in _sequence(service_spec.get("ports"))
            if (port := _mapping(value)).get("name")
        }

    for monitor in indexed.get("ServiceMonitor", []):
        resource = _resource_id(monitor)
        spec = _mapping(monitor.get("spec"))
        selector = _mapping(_mapping(spec.get("selector")).get("matchLabels"))
        targets = [
            name for name, labels in service_labels.items() if _selector_matches(selector, labels)
        ]
        if not targets:
            reject(
                "orphan_service_monitor",
                resource,
                "spec.selector.matchLabels",
                "selector must target a Service",
            )
        for index, value in enumerate(_sequence(spec.get("endpoints"))):
            port_name = _mapping(value).get("port")
            if (
                not isinstance(port_name, str)
                or not targets
                or any(port_name not in service_ports[name] for name in targets)
            ):
                reject(
                    "unknown_service_monitor_port",
                    resource,
                    f"spec.endpoints[{index}].port",
                    "endpoint port must name a port exposed by every selected Service",
                )

    pdb_targets: set[str] = set()
    for pdb in indexed.get("PodDisruptionBudget", []):
        resource = _resource_id(pdb)
        spec = _mapping(pdb.get("spec"))
        selector = _mapping(_mapping(spec.get("selector")).get("matchLabels"))
        targets = [
            name
            for name, labels in deployment_labels.items()
            if _selector_matches(selector, labels)
        ]
        if not targets:
            reject(
                "orphan_pdb",
                resource,
                "spec.selector.matchLabels",
                "selector must target a Deployment",
            )
        pdb_targets.update(targets)
        if spec.get("minAvailable") is None and spec.get("maxUnavailable") is None:
            reject(
                "missing_disruption_budget",
                resource,
                "spec",
                "minAvailable or maxUnavailable is required",
            )
        minimum = spec.get("minAvailable")
        if isinstance(minimum, int) and targets:
            if minimum < 1 or any(minimum > deployment_replicas.get(name, 0) for name in targets):
                reject(
                    "invalid_disruption_budget",
                    resource,
                    "spec.minAvailable",
                    "must preserve at least one pod without exceeding replicas",
                )

    network_targets: set[str] = set()
    for network_policy in indexed.get("NetworkPolicy", []):
        resource = _resource_id(network_policy)
        spec = _mapping(network_policy.get("spec"))
        selector = _mapping(_mapping(spec.get("podSelector")).get("matchLabels"))
        targets = [
            name
            for name, labels in deployment_labels.items()
            if _selector_matches(selector, labels)
        ]
        if not targets:
            reject(
                "orphan_network_policy",
                resource,
                "spec.podSelector.matchLabels",
                "selector must target a Deployment",
            )
        network_targets.update(targets)
        policy_types = set(_sequence(spec.get("policyTypes")))
        if not {"Ingress", "Egress"}.issubset(policy_types):
            reject(
                "incomplete_network_policy",
                resource,
                "spec.policyTypes",
                "Ingress and Egress isolation are required",
            )

    hpa_targets: set[str] = set()
    for hpa in indexed.get("HorizontalPodAutoscaler", []):
        resource = _resource_id(hpa)
        spec = _mapping(hpa.get("spec"))
        target = _mapping(spec.get("scaleTargetRef"))
        target_name = target.get("name")
        if target.get("kind") != "Deployment" or target_name not in deployment_labels:
            reject("orphan_hpa", resource, "spec.scaleTargetRef", "must target a known Deployment")
            continue
        hpa_targets.add(str(target_name))
        minimum = spec.get("minReplicas")
        maximum = spec.get("maxReplicas")
        if not isinstance(minimum, int) or minimum < policy.min_replicas:
            reject(
                "unsafe_hpa_minimum",
                resource,
                "spec.minReplicas",
                f"must be >= {policy.min_replicas}",
            )
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not isinstance(minimum, int)
            or maximum <= minimum
        ):
            reject(
                "invalid_hpa_range",
                resource,
                "spec.maxReplicas",
                "maxReplicas must be greater than minReplicas",
            )

    for name in sorted(deployment_labels):
        resource = f"Deployment/{name}"
        for covered, code, path, message in (
            (
                name in pdb_targets,
                "missing_pdb_coverage",
                "crossResource.pdb",
                "Deployment requires PodDisruptionBudget coverage",
            ),
            (
                name in network_targets,
                "missing_network_policy_coverage",
                "crossResource.networkPolicy",
                "Deployment requires ingress and egress isolation",
            ),
            (
                name in hpa_targets,
                "missing_hpa_coverage",
                "crossResource.hpa",
                "Deployment requires HorizontalPodAutoscaler coverage",
            ),
        ):
            if not covered:
                reject(code, resource, path, message)

    ordered = tuple(sorted(violations))
    return AdmissionReport(
        accepted=not ordered,
        document_count=len(documents),
        deployment_count=len(deployments),
        violations=ordered,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Kubernetes workload admission policy")
    parser.add_argument("paths", nargs="*", type=Path, help="YAML files (defaults to k8s/*.yaml)")
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON evidence")
    args = parser.parse_args(argv)

    paths = args.paths or sorted(Path("k8s").glob("*.yaml"))
    try:
        report = evaluate_documents(load_documents(paths))
    except (ManifestInputError, ValueError) as exc:
        payload = {"accepted": False, "error": "invalid_manifest_input", "message": str(exc)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    if args.json or not report.accepted:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"accepted {report.document_count} Kubernetes resources across {report.deployment_count} deployment(s)"
        )
    return 0 if report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
