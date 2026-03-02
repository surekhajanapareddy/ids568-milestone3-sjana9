# LINEAGE REPORT 

## Multiple Experiment runs:

| Run | Accuracy | Validation | C      | max_iter | random_state |
| --- | -------- | ---------- | ------ | -------- | ------------ |
| 1   | 0.3333   | 0 ❌        | 1.0    | 1        | 1            |
| 2   | 1.0      | 1 ✅        | 150.0  | 500      | 42           |
| 3   | 0.7      | 0 ❌        | 0.0001 | 50       | 100          |
| 4   | 0.9666   | 1 ✅        | 1.0    | 1000     | 42           |
| 5   | 0.9666   | 1 ✅        | 1.0    | 100      | 42           |
| 6   | 0.9666   | 1 ✅        | 0.1    | 100      | 42           |

## Experiment Design

To evaluate model robustness and hyperparameter sensitivity, six experimental runs were conducted using Logistic Regression on the Iris dataset. The experiments varied key hyperparameters including:

    Regularization strength (C)
    Maximum training iterations (max_iter)
    Random seed (random_state)

All experiments were logged through MLflow and orchestrated using Apache Airflow to ensure reproducibility and lineage tracking.

## Hyperparameter Impact Analysis

The experiments demonstrated clear sensitivity to hyperparameter selection:

    Extremely low iteration values (max_iter=1) resulted in severe underfitting and poor accuracy (33.3%).
    Very strong regularization (C=0.0001) reduced model flexibility and lowered accuracy to 70%.
    Moderate to high regularization values (C=1.0 to 150.0) with sufficient iterations (≥100) produced stable high accuracy (~96.6%–100%).
    Random state variation had minimal effect due to dataset stability.

Validation gating (threshold=0.90) successfully filtered out underperforming models.

## Model Selection for Production

The model with:

    C = 150.0
    max_iter = 500
    random_state = 42
    Accuracy = 1.0
    validation_passed = 1

was selected as the production candidate.

This configuration demonstrated optimal performance while satisfying validation constraints. The model was registered in the MLflow Model Registry, ensuring version control and traceability.

## Reproducibility & Lineage

All runs were orchestrated through Airflow DAG tasks, ensuring:

    Deterministic preprocessing
    Logged hyperparameters
    Registered model artifacts
    Experiment traceability
    Versioned model lifecycle

This establishes a complete MLOps workflow from training to deployment readiness.