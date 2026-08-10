from pathlib import Path

import joblib

from train_model import main


def test_training_creates_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main()
    model_path = Path("artifacts/model.joblib")
    assert model_path.exists()
    model = joblib.load(model_path)
    probability = model.predict_proba([[0.0, 0.0, 0.0, 0.0]])[0, 1]
    assert 0.0 <= probability <= 1.0
