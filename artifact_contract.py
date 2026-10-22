from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MANIFEST_SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 4096
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_sha256",
    "artifact_size_bytes",
    "feature_count",
    "classes",
    "model_type",
}


class ArtifactContractError(ValueError):
    """Raised when an artifact cannot be admitted safely."""


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    artifact_sha256: str
    artifact_size_bytes: int
    feature_count: int
    classes: tuple[int, int]
    model_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "feature_count": self.feature_count,
            "classes": list(self.classes),
            "model_type": self.model_type,
        }


def qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def build_manifest(model: object, artifact: bytes) -> ArtifactManifest:
    feature_count = _positive_int(
        getattr(model, "n_features_in_", None), "feature_count"
    )
    classes = _binary_classes(getattr(model, "classes_", None))
    return ArtifactManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        artifact_size_bytes=len(artifact),
        feature_count=feature_count,
        classes=classes,
        model_type=qualified_type(model),
    )


def write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    encoded = (
        json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ArtifactContractError("manifest exceeds byte limit")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def load_verified_model(
    artifact_path: Path,
    manifest_path: Path,
    *,
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[object, ArtifactManifest]:
    manifest = load_manifest(manifest_path)
    artifact = _read_regular_file(artifact_path, max_artifact_bytes)
    if len(artifact) != manifest.artifact_size_bytes:
        raise ArtifactContractError("artifact size does not match manifest")
    if hashlib.sha256(artifact).hexdigest() != manifest.artifact_sha256:
        raise ArtifactContractError("artifact digest does not match manifest")

    try:
        model = joblib.load(io.BytesIO(artifact))
    except Exception as error:
        raise ArtifactContractError("artifact deserialization failed") from error

    if qualified_type(model) != manifest.model_type:
        raise ArtifactContractError("model type does not match manifest")
    if _positive_int(getattr(model, "n_features_in_", None), "feature_count") != (
        manifest.feature_count
    ):
        raise ArtifactContractError("feature count does not match manifest")
    if _binary_classes(getattr(model, "classes_", None)) != manifest.classes:
        raise ArtifactContractError("class contract does not match manifest")
    return model, manifest


def load_manifest(path: Path) -> ArtifactManifest:
    raw = _read_regular_file(path, MAX_MANIFEST_BYTES)
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError("manifest is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != REQUIRED_MANIFEST_FIELDS:
        raise ArtifactContractError("manifest fields do not match schema")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ArtifactContractError("unsupported manifest schema")

    digest = payload["artifact_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactContractError("artifact digest is invalid")
    model_type = payload["model_type"]
    if (
        not isinstance(model_type, str)
        or not model_type
        or len(model_type) > 200
        or any(character.isspace() for character in model_type)
    ):
        raise ArtifactContractError("model type is invalid")

    return ArtifactManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        artifact_sha256=digest,
        artifact_size_bytes=_positive_int(
            payload["artifact_size_bytes"], "artifact_size_bytes"
        ),
        feature_count=_positive_int(payload["feature_count"], "feature_count"),
        classes=_binary_classes(payload["classes"]),
        model_type=model_type,
    )


def _read_regular_file(path: Path, byte_limit: int) -> bytes:
    if (
        isinstance(byte_limit, bool)
        or not isinstance(byte_limit, int)
        or byte_limit <= 0
    ):
        raise ArtifactContractError("byte limit must be a positive integer")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactContractError("artifact file is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactContractError("artifact path is not a regular file")
        if metadata.st_size <= 0:
            raise ArtifactContractError("artifact file is empty")
        if metadata.st_size > byte_limit:
            raise ArtifactContractError("artifact file exceeds byte limit")
        chunks: list[bytes] = []
        remaining = byte_limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > byte_limit:
            raise ArtifactContractError("artifact file exceeds byte limit")
        after = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ArtifactContractError("artifact changed while being read")
        if len(content) != metadata.st_size:
            raise ArtifactContractError("artifact read was incomplete")
        return content
    finally:
        os.close(descriptor)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactContractError(f"{field} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ArtifactContractError(f"{field} must be positive")
    return normalized


def _binary_classes(value: object) -> tuple[int, int]:
    classes = np.asarray(value)
    if (
        classes.shape != (2,)
        or classes.dtype.kind not in "iu"
        or set(classes.tolist()) != {0, 1}
    ):
        raise ArtifactContractError("classes must be integer labels 0 and 1")
    return tuple(int(label) for label in classes.tolist())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactContractError("manifest contains duplicate fields")
        result[key] = value
    return result
