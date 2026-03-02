# IDS 568 – Milestone 3  

<p align="center">
  [![CI](https://github.com/surekhajanapareddy/ids568-milestone3-sjana9/actions/workflows/train_and_validate.yml/badge.svg)](https://github.com/surekhajanapareddy/ids568-milestone3-sjana9/actions/workflows/train_and_validate.yml)
  <img src="https://img.shields.io/badge/Docker-Containerized-blue?logo=docker">
  <img src="https://img.shields.io/badge/Airflow-2.8.1-red?logo=apache-airflow">
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow">
  <img src="https://img.shields.io/badge/PostgreSQL-Backend-blue?logo=postgresql">
  <img src="https://img.shields.io/badge/Python-3.10+-green?logo=python">
</p>

## End-to-End MLOps Pipeline with Airflow & MLflow

This project implements a production-style machine learning pipeline using:

- Apache Airflow (workflow orchestration)
- MLflow (experiment tracking & model registry)
- Docker & Docker Compose (containerized deployment)
- PostgreSQL (Airflow metadata database)
- GitHub Actions (CI/CD validation)

The pipeline performs preprocessing, training, validation, and model registration in a fully reproducible and version-controlled manner.

---

# 🚀 Project Architecture

```
Airflow DAG
   │
   ├── preprocess_data
   │
   ├── train_model (MLflow logging)
   │
   ├── validate_model (quality gate)
   │
   └── register_model (MLflow Model Registry)
```

### Components

### 1️⃣ Apache Airflow
- Orchestrates the ML workflow
- Handles retries and failure callbacks
- Supports parameterized hyperparameter experimentation

### 2️⃣ MLflow
- Logs parameters and metrics
- Tracks experiment lineage
- Registers validated models
- Maintains versioned model registry

### 3️⃣ PostgreSQL
- Stores Airflow metadata
- Production-ready replacement for SQLite

### 4️⃣ Docker
- Ensures reproducibility
- Isolates dependencies
- Enables consistent local execution

---

# 📁 Repository Structure

```
ids568-milestone3-sjana9/
│
├── dags/
│   └── train_pipeline.py
│
├── preprocess.py
├── train.py
├── model_validation.py
├── register_model.py
│
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
│
└── .github/
    └── workflows/
        └── train_and_validate.yml
```

---

# 🛠 Setup Instructions

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/surekhajanapareddy/ids568-milestone3-sjana9.git
cd ids568-milestone3-sjana9
```

---

## 2️⃣ Start the Full Stack

```bash
docker compose up -d
```

This launches:

- Airflow Webserver → http://localhost:8080  
- MLflow UI → http://localhost:5000  
- PostgreSQL container  

---

## 3️⃣ Initialize Airflow (if needed)

```bash
docker compose up airflow-init
```

---

## 4️⃣ Access Airflow UI

Open:

```
http://localhost:8080
```

Login (default credentials):

```
username: airflow
password: airflow
```

---

# ▶️ How to Run the Pipeline

## Trigger Default Run

1. Open Airflow UI
2. Select DAG: `train_pipeline`
3. Click **Trigger DAG**

---

## Run with Custom Hyperparameters

Click **Trigger DAG w/ config** and provide JSON:

```json
{
  "max_iter": 500,
  "C": 10.0,
  "solver": "lbfgs",
  "random_state": 42
}
```

---

# 🔬 Experiment Tracking

Open MLflow UI:

```
http://localhost:5000
```

We can:

- Compare multiple runs
- View parameter changes
- Analyze accuracy metrics
- Inspect model artifacts
- View registered model versions

---

# 🧪 Quality Gate (Validation)

The validation task enforces:

```
accuracy >= 0.90
```

If validation fails:

- Model is NOT registered
- Pipeline stops at validation stage

This simulates production model approval controls.

---

# 🔁 Retry & Failure Handling

The DAG includes:

- retries = 3
- retry_delay = 1 minute
- execution_timeout = 10 minutes
- custom failure callback logging

Failed tasks automatically retry before final failure.

---

# 📊 Hyperparameter Experiments

Minimum 5 experiments were executed varying:

- Regularization strength (C)
- max_iter
- random_state

Results were tracked in MLflow to evaluate model performance sensitivity.

---

# 🏷 Model Registration

Only validated models are registered in:

MLflow Model Registry

Registered model name:

```
IrisClassifier
```

Each successful run increments model version automatically.

---

# 🔄 CI/CD Integration

GitHub Actions workflow:

```
.github/workflows/train_and_validate.yml
```

Pipeline validates:

- Python dependencies
- Model training execution
- Validation logic
- Basic quality checks

Automatically runs on push to main branch.

---

# 🧠 Design Decisions

| Component | Reason |
|-----------|--------|
| Airflow | Production-grade orchestration |
| MLflow | Standard experiment tracking |
| PostgreSQL | Reliable metadata backend |
| Docker | Reproducibility |
| Validation Gate | Prevent bad models from registering |

---

# 📌 Key Features Implemented

- End-to-end DAG pipeline
- Parameterized hyperparameter tuning
- MLflow experiment logging
- Model registry with versioning
- Retry & failure handling
- Validation threshold gate
- CI/CD workflow

---

# 🏁 Milestone Completion Status

| Requirement | Status |
|-------------|--------|
| Airflow DAG | ✅ Complete |
| MLflow Tracking | ✅ Complete |
| Model Registry | ✅ Complete |
| Retry Handling | ✅ Complete |
| 5+ Experiments | ✅ Complete |
| Validation Gate | ✅ Complete |
| CI/CD Workflow | ✅ Complete |

---

# 👩‍💻 Author

Surekha Janapareddy  
MS Business Analytics  
University of Illinois Chicago  

---

# 📜 License

For academic use only.