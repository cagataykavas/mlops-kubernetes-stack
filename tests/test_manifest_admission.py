from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.validate_manifests import (
    AdmissionPolicy,
    ManifestInputError,
    evaluate_documents,
    load_documents,
    main,
)


@pytest.fixture
def documents() -> list[dict]:
    return load_documents(sorted(Path("k8s").glob("*.yaml")))


def resource(documents: list[dict], kind: str) -> dict:
    return next(document for document in documents if document["kind"] == kind)


def codes(report) -> set[str]:
    return {violation.code for violation in report.violations}


def test_repository_manifests_are_admitted_with_deterministic_evidence(documents):
    report = evaluate_documents(documents)

    assert report.accepted is True
    assert report.document_count == 6
    assert report.deployment_count == 1
    assert report.violations == ()
    assert report.to_dict()["accepted"] is True


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda deployment: deployment["spec"].update(replicas=1), "insufficient_replicas"),
        (
            lambda deployment: deployment["spec"]["strategy"]["rollingUpdate"].update(
                maxUnavailable=1
            ),
            "unsafe_rollout_strategy",
        ),
        (
            lambda deployment: deployment["spec"]["template"]["spec"]["containers"][0].update(
                image="example/model:latest"
            ),
            "mutable_or_invalid_image",
        ),
        (
            lambda deployment: deployment["spec"]["template"]["spec"]["containers"][0][
                "securityContext"
            ].update(allowPrivilegeEscalation=True),
            "privilege_escalation_allowed",
        ),
        (
            lambda deployment: deployment["spec"]["template"]["spec"]["containers"][0].pop(
                "readinessProbe"
            ),
            "missing_http_probe",
        ),
        (
            lambda deployment: deployment["spec"]["template"]["spec"]["containers"][0]["resources"][
                "limits"
            ].pop("memory"),
            "missing_resource_budget",
        ),
    ],
)
def test_deployment_policy_failures_are_explicit(documents, mutate, expected_code):
    candidate = deepcopy(documents)
    mutate(resource(candidate, "Deployment"))

    report = evaluate_documents(candidate)

    assert report.accepted is False
    assert expected_code in codes(report)


@pytest.mark.parametrize(
    ("kind", "selector_path", "expected_codes"),
    [
        ("Service", ("spec", "selector"), {"orphan_service"}),
        (
            "PodDisruptionBudget",
            ("spec", "selector", "matchLabels"),
            {"orphan_pdb", "missing_pdb_coverage"},
        ),
        (
            "NetworkPolicy",
            ("spec", "podSelector", "matchLabels"),
            {"orphan_network_policy", "missing_network_policy_coverage"},
        ),
    ],
)
def test_cross_resource_selectors_must_cover_a_deployment(
    documents, kind, selector_path, expected_codes
):
    candidate = deepcopy(documents)
    selected = resource(candidate, kind)
    cursor = selected
    for key in selector_path[:-1]:
        cursor = cursor[key]
    cursor[selector_path[-1]] = {"app": "wrong-service"}

    report = evaluate_documents(candidate)

    assert expected_codes.issubset(codes(report))


def test_service_monitor_must_target_a_service_and_named_port(documents):
    candidate = deepcopy(documents)
    monitor = resource(candidate, "ServiceMonitor")
    monitor["spec"]["selector"]["matchLabels"] = {"app": "wrong-service"}
    monitor["spec"]["endpoints"][0]["port"] = "missing"

    report = evaluate_documents(candidate)

    assert {"orphan_service_monitor", "unknown_service_monitor_port"}.issubset(codes(report))


def test_hpa_must_target_known_deployment_and_keep_safe_range(documents):
    candidate = deepcopy(documents)
    hpa = resource(candidate, "HorizontalPodAutoscaler")
    hpa["spec"]["scaleTargetRef"]["name"] = "missing"
    hpa["spec"]["minReplicas"] = 1
    hpa["spec"]["maxReplicas"] = 1

    report = evaluate_documents(candidate)

    assert {"orphan_hpa", "missing_hpa_coverage"}.issubset(codes(report))


def test_duplicate_resource_identity_is_rejected(documents):
    candidate = deepcopy(documents)
    candidate.append(deepcopy(resource(candidate, "Service")))

    report = evaluate_documents(candidate)

    assert "duplicate_resource" in codes(report)


def test_custom_policy_is_validated(documents):
    assert "insufficient_replicas" in codes(
        evaluate_documents(documents, AdmissionPolicy(min_replicas=3))
    )
    with pytest.raises(ValueError, match="min_replicas"):
        AdmissionPolicy(min_replicas=0)


def test_empty_or_non_mapping_input_fails_closed(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("---\n", encoding="utf-8")
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- not-a-resource\n", encoding="utf-8")

    with pytest.raises(ManifestInputError, match="no Kubernetes"):
        load_documents([empty])
    with pytest.raises(ManifestInputError, match="must be a mapping"):
        load_documents([scalar])


def test_cli_exit_codes_distinguish_policy_failure_from_malformed_input(
    documents, tmp_path, capsys
):
    unsafe = deepcopy(documents)
    resource(unsafe, "Deployment")["spec"]["replicas"] = 0
    unsafe_path = tmp_path / "unsafe.yaml"
    unsafe_path.write_text(yaml.safe_dump_all(unsafe), encoding="utf-8")
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("spec: [", encoding="utf-8")

    assert main([str(unsafe_path), "--json"]) == 2
    assert '"insufficient_replicas"' in capsys.readouterr().out
    assert main([str(malformed_path), "--json"]) == 3
    assert '"invalid_manifest_input"' in capsys.readouterr().out
