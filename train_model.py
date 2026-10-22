from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from artifact_contract import build_manifest, write_manifest


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2500, 4))
    logits = 1.4 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] - 0.2 * x[:, 3]
    y = (logits + rng.normal(scale=0.7, size=len(x)) > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(x, y)
    artifact_directory = Path("artifacts")
    artifact_directory.mkdir(exist_ok=True)
    model_path = artifact_directory / "model.joblib"
    manifest_path = artifact_directory / "model.manifest.json"
    joblib.dump(model, model_path)
    manifest = build_manifest(model, model_path.read_bytes())
    write_manifest(manifest_path, manifest)
    print(
        "saved artifacts/model.joblib and artifacts/model.manifest.json "
        f"(sha256={manifest.artifact_sha256})"
    )


if __name__ == "__main__":
    main()
