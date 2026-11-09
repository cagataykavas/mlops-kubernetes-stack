from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from artifact_contract import (
    ArtifactContractError,
    ManifestTrustPolicy,
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


def signed_bundle(tmp_path: Path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)
    private_key = Ed25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    manifest_bytes = manifest_path.read_bytes()
    signature_path = tmp_path / "model.manifest.sig.json"
    signature_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "ed25519",
                "key_id": "build-signer-v1",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "signature_hex": private_key.sign(manifest_bytes).hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    policy = ManifestTrustPolicy(
        key_id="build-signer-v1", public_key_hex=public_key_hex
    )
    return artifact_path, manifest_path, signature_path, manifest, policy


def test_verified_bundle_is_loaded_from_admitted_bytes(tmp_path):
    artifact_path, manifest_path, manifest = artifact_bundle(tmp_path)

    loaded, admitted = load_verified_model(artifact_path, manifest_path)

    assert loaded.n_features_in_ == 4
    assert loaded.classes_.tolist() == [0, 1]
    assert admitted == manifest


def test_signed_bundle_is_verified_before_deserialization(tmp_path):
    artifact_path, manifest_path, signature_path, manifest, policy = signed_bundle(
        tmp_path
    )

    loaded, admitted = load_verified_model(
        artifact_path,
        manifest_path,
        trust_policy=policy,
        signature_path=signature_path,
    )

    assert loaded.n_features_in_ == 4
    assert admitted == manifest


def test_paired_artifact_and_manifest_replacement_needs_a_new_signature(tmp_path):
    artifact_path, manifest_path, signature_path, _, policy = signed_bundle(tmp_path)
    replacement_model = model(feature_count=5)
    joblib.dump(replacement_model, artifact_path)
    replacement_manifest = build_manifest(replacement_model, artifact_path.read_bytes())
    write_manifest(manifest_path, replacement_manifest)

    with pytest.raises(ArtifactContractError, match="signed manifest digest"):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=policy,
            signature_path=signature_path,
        )


@pytest.mark.parametrize(
    ("policy_mutation", "message"),
    [
        ({"key_id": "other-signer"}, "not trusted"),
        ({"public_key_hex": "00" * 32}, "verification failed"),
        ({"public_key_hex": "AA" * 32}, "lowercase hex"),
        ({"key_id": "bad key id"}, "key id"),
    ],
)
def test_untrusted_manifest_signer_fails_closed(tmp_path, policy_mutation, message):
    artifact_path, manifest_path, signature_path, _, policy = signed_bundle(tmp_path)
    untrusted = replace(policy, **policy_mutation)

    with pytest.raises(ArtifactContractError, match=message):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=untrusted,
            signature_path=signature_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "fields"),
        (lambda value: value.update(algorithm="rsa"), "unsupported"),
        (lambda value: value.update(signature_hex="AA" * 64), "lowercase hex"),
        (lambda value: value.update(signature_hex="00" * 64), "verification failed"),
        (lambda value: value.update(manifest_sha256="0" * 64), "digest"),
    ],
)
def test_malformed_signature_envelope_fails_closed(tmp_path, mutate, message):
    artifact_path, manifest_path, signature_path, _, policy = signed_bundle(tmp_path)
    payload = json.loads(signature_path.read_text())
    mutate(payload)
    signature_path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactContractError, match=message):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=policy,
            signature_path=signature_path,
        )


def test_duplicate_signature_envelope_field_is_rejected(tmp_path):
    artifact_path, manifest_path, signature_path, _, policy = signed_bundle(tmp_path)
    encoded = signature_path.read_text()
    signature_path.write_text(encoded[:-1] + ',"key_id":"build-signer-v1"}')

    with pytest.raises(ArtifactContractError, match="duplicate"):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=policy,
            signature_path=signature_path,
        )


def test_trust_policy_and_signature_path_must_be_paired(tmp_path):
    artifact_path, manifest_path, _, _, policy = signed_bundle(tmp_path)

    with pytest.raises(ArtifactContractError, match="configured together"):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=policy,
        )


def test_symlink_signature_envelope_is_rejected(tmp_path):
    artifact_path, manifest_path, signature_path, _, policy = signed_bundle(tmp_path)
    symlink = tmp_path / "linked.sig.json"
    symlink.symlink_to(signature_path)

    with pytest.raises(ArtifactContractError, match="unavailable"):
        load_verified_model(
            artifact_path,
            manifest_path,
            trust_policy=policy,
            signature_path=symlink,
        )


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
