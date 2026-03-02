from __future__ import annotations
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def preprocess(run_id: str, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    import numpy as np
    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_test.npy", y_test)

    meta = {
        "run_id": run_id,
        "output_dir": str(out),
        "test_size": 0.2,
        "random_state": 42
    }

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta