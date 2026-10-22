from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from artifact_contract import (
    ArtifactContractError,
    build_manifest,
    load_verified_model,
    write_manifest,
)


def model(feature_count: int = 4, classes=(0, 1)) -> SimpleNamespace:
    return SimpleNamespace(
        n_features_in_=feature_count,
        classes_=np.asarray(classes),
    )


def artifact_bundle(tmp_path: Path, value=None):
    artifact_model = value or model()
    artifact_path = tmp_path / "model.joblib"
    manifest_path = tmp_path / "model.manifest.json"
    joblib.dump(artifact_model, artifact_path)
    manifest = build_manifest(artifact_model, artifact_path.read_bytes())
    write_manifest(manifest_path, manifest)
    return artifact_path, manifest_path, manifest


def test_verified_bundle_is_loaded_from_admitted_bytes(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)

    loaded, admitted = load_verified_model(artifact_path, manifest_path)

    assert loaded.n_features_in_ == 4
    assert loaded.classes_.tolist() == [0, 1]
    assert admitted == manifest


@pytest.mark.parametrize("field", ["artifact_sha256", "artifact_size_bytes"])
def test_artifact_identity_mismatch_fails_closed(tmp_path, field):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    replacement = (
        "0" * 64 if field == "artifact_sha256" else manifest.artifact_size_bytes + 1
    )
    write_manifest(manifest_path, replace(manifest, **{field: replacement}))

    with pytest.raises(ArtifactContractError, match="does not match manifest"):
        load_verified_model(artifact_path, manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_count", 5, "feature count"),
        ("classes", [1, 2], "classes"),
        ("model_type", "unexpected.Model", "model type"),
    ],
)
def test_manifest_contract_mismatch_fails_closed(tmp_path, field, value, message):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    payload = manifest.to_dict()
    payload[field] = value
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactContractError, match=message):
        load_verified_model(artifact_path, manifest_path)


def test_corrupt_artifact_with_matching_digest_is_rejected(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    corrupt = b"not a joblib artifact"
    artifact_path.write_bytes(corrupt)
    write_manifest(
        manifest_path,
        replace(
            manifest,
            artifact_sha256=hashlib.sha256(corrupt).hexdigest(),
            artifact_size_bytes=len(corrupt),
        ),
    )

    with pytest.raises(ArtifactContractError, match="deserialization failed"):
        load_verified_model(artifact_path, manifest_path)


def test_duplicate_manifest_field_is_rejected(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    valid = json.dumps(manifest.to_dict(), separators=(",", ":"))
    manifest_path.write_text(valid[:-1] + ',"feature_count":4}')

    with pytest.raises(ArtifactContractError, match="duplicate"):
        load_verified_model(artifact_path, manifest_path)


def test_unknown_manifest_field_is_rejected(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    payload = manifest.to_dict()
    payload["unreviewed"] = True
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactContractError, match="fields"):
        load_verified_model(artifact_path, manifest_path)


def test_symlink_artifact_is_rejected(tmp_path):
    artifact_path, manifest_path, _ = artifact_bundle(tmp_path)
    symlink = tmp_path / "linked.joblib"
    symlink.symlink_to(artifact_path)

    with pytest.raises(ArtifactContractError, match="unavailable"):
        load_verified_model(symlink, manifest_path)


def test_artifact_byte_limit_is_enforced_before_deserialization(tmp_path):
    artifact_path, manifest_path, _ = artifact_bundle(tmp_path)

    with pytest.raises(ArtifactContractError, match="byte limit"):
        load_verified_model(
            artifact_path,
            manifest_path,
            max_artifact_bytes=artifact_path.stat().st_size - 1,
        )


def test_manifest_is_deterministic_and_private_to_contract_fields(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    first = manifest_path.read_bytes()
    write_manifest(manifest_path, manifest)

    assert manifest_path.read_bytes() == first
    assert set(json.loads(first)) == {
        "schema_version",
        "artifact_sha256",
        "artifact_size_bytes",
        "feature_count",
        "classes",
        "model_type",
    }
    assert artifact_path.name not in first.decode()
