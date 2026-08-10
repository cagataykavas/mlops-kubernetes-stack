from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2500, 4))
    logits = 1.4 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] - 0.2 * x[:, 3]
    y = (logits + rng.normal(scale=0.7, size=len(x)) > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(x, y)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")
    print("saved artifacts/model.joblib")


if __name__ == "__main__":
    main()
