from __future__ import annotations
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train(processed_dir: str, model_dir: str, experiment_name: str = "train_pipeline") -> dict:
    """
    Trains model from processed .npy files.
    Logs params/metrics/model to MLflow.
    Writes a local model artifact for idempotency.
    Returns dict with run_id, accuracy, model_uri, model_path.
    """
    processed = Path(processed_dir)
    model_out = Path(model_dir)
    model_out.mkdir(parents=True, exist_ok=True)

    # --- Idempotency: if model already exists, skip re-train (Airflow will decide skip)
    model_path = model_out / "model.joblib"

    X_train = np.load(processed / "X_train.npy")
    X_test  = np.load(processed / "X_test.npy")
    y_train = np.load(processed / "y_train.npy")
    y_test  = np.load(processed / "y_test.npy")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))

        # log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        # log metrics
        mlflow.log_metric("accuracy", acc)

        # log model artifact to MLflow
        mlflow.sklearn.log_model(model, "model")

        # also write local artifact for idempotency proof
        import joblib
        joblib.dump(model, model_path)

        model_uri = f"runs:/{run_id}/model"

        return {
            "run_id": run_id,
            "accuracy": acc,
            "model_uri": model_uri,
            "model_path": str(model_path),
        }