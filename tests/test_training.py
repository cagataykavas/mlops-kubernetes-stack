from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib


def load_training_main():
    module_path = Path(__file__).resolve().parents[1] / "train_model.py"
    spec = spec_from_file_location("train_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_training_creates_model(tmp_path, monkeypatch):
    main = load_training_main()
    monkeypatch.chdir(tmp_path)
    main()

    model_path = Path("artifacts/model.joblib")
    assert model_path.exists()

    model = joblib.load(model_path)
    probability = model.predict_proba([[0.0, 0.0, 0.0, 0.0]])[0, 1]
    assert 0.0 <= probability <= 1.0
