from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException, AirflowFailException


# Ensure project modules are importable inside container
sys.path.append("/opt/airflow/project")

from preprocess import preprocess
from train import train
from model_validation import validate
from register_model import register


# -------------------------------------------------------
# Failure Callback
# -------------------------------------------------------
def failure_callback(context):
    ti = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = ti.task_id if ti else "unknown_task"
    run_id = context.get("run_id")
    exc = context.get("exception")

    print(
        f"""
        ---------------- FAILURE DETECTED ----------------
        DAG: {dag_id}
        Task: {task_id}
        Run ID: {run_id}
        Exception: {exc}
        --------------------------------------------------
        """
    )


# -------------------------------------------------------
# Utility Path Builder
# -------------------------------------------------------
def _paths(ds_nodash: str):
    base = Path("/opt/airflow/project")
    processed_dir = base / "data" / "processed" / ds_nodash
    model_dir = base / "artifacts" / ds_nodash

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return str(processed_dir), str(model_dir)


# -------------------------------------------------------
# Task 1: Preprocess
# -------------------------------------------------------
def preprocess_task(**context):
    ds_nodash = context["ds_nodash"]
    processed_dir, _ = _paths(ds_nodash)

    print(f"[INFO] Starting preprocessing for run {ds_nodash}")

    return preprocess(
        run_id=ds_nodash,
        output_dir=processed_dir
    )


# -------------------------------------------------------
# Task 2: Train (Parameterized)
# -------------------------------------------------------
def train_task(**context):
    ds_nodash = context["ds_nodash"]
    processed_dir, model_dir = _paths(ds_nodash)

    #  Read hyperparameters from DAG params
    params = context["params"]

    max_iter = int(params.get("max_iter", 200))
    C = float(params.get("C", 1.0))
    solver = params.get("solver", "lbfgs")
    random_state = int(params.get("random_state", 42))

    print(f"[INFO] Training with params:")
    print(f"       max_iter={max_iter}, C={C}, solver={solver}, random_state={random_state}")

    result = train(
        processed_dir=processed_dir,
        model_dir=model_dir,
        experiment_name="train_pipeline",
        max_iter=max_iter,
        C=C,
        solver=solver,
        random_state=random_state,
    )

    if not result or "run_id" not in result:
        raise AirflowFailException("Training did not return expected output.")

    return result


# -------------------------------------------------------
# Task 3: Validate
# -------------------------------------------------------
def validate_task(**context):
    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")

    if not train_result:
        raise AirflowFailException("No training result found in XCom.")

    run_id = train_result["run_id"]
    accuracy = float(train_result["accuracy"])

    print(f"[INFO] Validating run {run_id} with accuracy {accuracy}")

    return validate(
        run_id=run_id,
        accuracy=accuracy,
        threshold=0.90
    )


# -------------------------------------------------------
# Task 4: Register
# -------------------------------------------------------
def register_task(**context):
    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")

    if not train_result:
        raise AirflowFailException("No training result found for registration.")

    run_id = train_result["run_id"]
    model_uri = train_result["model_uri"]

    print(f"[INFO] Registering model for run {run_id}")

    return register(
        run_id=run_id,
        model_uri=model_uri,
        model_name="IrisClassifier"
    )


# -------------------------------------------------------
# DAG Configuration
# -------------------------------------------------------
default_args = {
    "owner": "surekha",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
    "on_failure_callback": failure_callback,
}


with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Milestone 3: preprocess -> train -> validate -> register with MLflow",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "milestone3"],
    params={
        "max_iter": 200,
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": 42,
    },
) as dag:

    t1 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task,
    )

    t2 = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    t3 = PythonOperator(
        task_id="validate_model",
        python_callable=validate_task,
    )

    t4 = PythonOperator(
        task_id="register_model",
        python_callable=register_task,
    )

    t1 >> t2 >> t3 >> t4