from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

# Import your root-level pipeline modules
import sys
sys.path.append("/opt/airflow/project")

from preprocess import preprocess
from train import train
from model_validation import validate
from register_model import register


def failure_callback(context):
    ti = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = ti.task_id if ti else "unknown_task"
    run_id = context.get("run_id")
    exc = context.get("exception")
    print(f"[FAILURE] dag={dag_id} task={task_id} run_id={run_id} exception={exc}")


def _paths(ds_nodash: str):
    base = Path("/opt/airflow/project")
    processed_dir = base / "data" / "processed" / ds_nodash
    model_dir = base / "artifacts" / ds_nodash
    return str(processed_dir), str(model_dir)


def preprocess_task(**context):
    ds_nodash = context["ds_nodash"]
    processed_dir, _ = _paths(ds_nodash)
    return preprocess(run_id=ds_nodash, output_dir=processed_dir)


def train_task(**context):
    ds_nodash = context["ds_nodash"]
    processed_dir, model_dir = _paths(ds_nodash)

    # Idempotency: if model already exists, skip training
    model_path = Path(model_dir) / "model.joblib"
    if model_path.exists():
        raise AirflowSkipException(f"Model already exists at {model_path}. Skipping training.")

    result = train(processed_dir=processed_dir, model_dir=model_dir, experiment_name="train_pipeline")
    return result


def validate_task(**context):
    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")
    run_id = train_result["run_id"]
    accuracy = float(train_result["accuracy"])
    return validate(run_id=run_id, accuracy=accuracy, threshold=0.90)


def register_task(**context):
    ti = context["ti"]
    train_result = ti.xcom_pull(task_ids="train_model")
    run_id = train_result["run_id"]
    model_uri = train_result["model_uri"]
    return register(run_id=run_id, model_uri=model_uri, model_name="IrisClassifier")


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
    description="M3: preprocess -> train -> validate -> register with MLflow",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "milestone3"],
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