from __future__ import annotations
import mlflow

def validate(run_id: str, accuracy: float, threshold: float = 0.90) -> dict:
    """
    Gate: fail if accuracy < threshold.
    Logs validation status to same MLflow run.
    """
    passed = accuracy >= threshold

    # Attach to existing run and log extra metrics/tags
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("validation_threshold", threshold)
        mlflow.log_metric("validation_passed", 1.0 if passed else 0.0)

    if not passed:
        raise ValueError(f"Validation failed: accuracy={accuracy:.4f} < threshold={threshold:.2f}")

    return {"passed": True, "threshold": threshold, "accuracy": accuracy}