from __future__ import annotations

import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml


def load_release_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
    spec = spec_from_file_location("release", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load release module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load_release_module()

IMAGE = (
    "ghcr.io/cagataykavas/mlops-kubernetes-stack@sha256:"
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)
DIGEST = IMAGE.rsplit("@", 1)[1]


def completed(args: tuple[str, ...], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/cagataykavas/mlops-kubernetes-stack:latest",
        "ghcr.io/cagataykavas/mlops-kubernetes-stack:v1",
        "ghcr.io/cagataykavas/mlops-kubernetes-stack@sha256:abc",
        "GHCR.IO/cagataykavas/mlops-kubernetes-stack@sha256:" + "0" * 64,
    ],
)
def test_release_rejects_mutable_or_malformed_images(image: str) -> None:
    with pytest.raises(ValueError, match="immutable OCI reference"):
        release.validate_image(image)


@pytest.mark.parametrize("namespace", ["ML", "-ml", "ml_serve", "a" * 64])
def test_release_rejects_invalid_namespaces(namespace: str) -> None:
    with pytest.raises(ValueError, match="DNS-1123"):
        release.validate_namespace(namespace)


@pytest.mark.parametrize("timeout", ["0s", "31m", "1h", "forever"])
def test_release_bounds_rollout_timeout(timeout: str) -> None:
    with pytest.raises(ValueError, match="between 1s and 30m"):
        release.validate_timeout(timeout)


def test_rendered_manifest_is_pinned_before_apply() -> None:
    documents = list(yaml.safe_load_all(release.render_deployment_manifest(IMAGE)))
    deployment = next(document for document in documents if document["kind"] == "Deployment")
    template = deployment["spec"]["template"]
    container = next(item for item in template["spec"]["containers"] if item["name"] == "api")

    assert container["image"] == IMAGE
    assert template["metadata"]["annotations"][release.IMAGE_DIGEST_ANNOTATION] == DIGEST


def test_deploy_applies_pinned_manifest_and_attests_running_image(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_run(
        *args: str,
        check: bool = True,
        input_text: str | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output
        calls.append((args, input_text))
        if args[1:3] == ("get", "deployment/ml-model-service"):
            return completed(args, IMAGE)
        if args[1:3] == ("get", "pods"):
            return completed(args, f"pod-a\t{IMAGE}\tdocker-pullable://{IMAGE}\n")
        return completed(args)

    monkeypatch.setattr(release, "run", fake_run)
    release.deploy(IMAGE, "ml", "180s")

    deployment_apply = next(call for call in calls if call[0][-2:] == ("-f", "-"))
    apply_args, rendered = deployment_apply
    assert apply_args == ("kubectl", "apply", "-n", "ml", "-f", "-")
    assert rendered is not None and IMAGE in rendered
    assert not any(args[1:3] == ("set", "image") for args, _ in calls)
    assert any(args[1:3] == ("get", "pods") for args, _ in calls)


def test_attestation_failure_requests_rollback(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(
        *args: str,
        check: bool = True,
        input_text: str | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del check, input_text, capture_output
        calls.append(args)
        if args[1:3] == ("get", "deployment/ml-model-service"):
            return completed(args, "ghcr.io/example/other@sha256:" + "f" * 64)
        return completed(args)

    monkeypatch.setattr(release, "run", fake_run)
    with pytest.raises(release.ReleaseVerificationError, match="image mismatch"):
        release.deploy(IMAGE, "ml", "180s")

    assert any(args[1:3] == ("rollout", "undo") for args in calls)


def test_runtime_image_id_must_match_requested_digest(monkeypatch) -> None:
    def fake_run(
        *args: str,
        check: bool = True,
        input_text: str | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del check, input_text, capture_output
        if args[1:3] == ("get", "deployment/ml-model-service"):
            return completed(args, IMAGE)
        return completed(
            args,
            f"pod-a\t{IMAGE}\tdocker-pullable://ghcr.io/example/model@sha256:{'f' * 64}\n",
        )

    monkeypatch.setattr(release, "run", fake_run)
    with pytest.raises(release.ReleaseVerificationError, match="image ID"):
        release.verify_deployment_image(IMAGE, "ml")
