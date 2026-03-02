from __future__ import annotations
from mlflow.tracking import MlflowClient
import mlflow

def register(run_id: str, model_uri: str, model_name: str = "IrisClassifier") -> dict:
    """
    Registers runs:/... model into MLflow Model Registry.
    Idempotent: if same run_id already registered, do nothing.
    """
    client = MlflowClient()

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    # Idempotency: check if this run_id already registered
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if getattr(v, "run_id", None) == run_id:
            return {"registered": True, "model_name": model_name, "version": v.version, "note": "already_registered"}

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    return {"registered": True, "model_name": model_name, "version": mv.version}