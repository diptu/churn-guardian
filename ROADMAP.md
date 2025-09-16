# 🚀  Churn Prediction Project Roadmap

## 1. Project Setup

[x]Project structure (app/, scripts/, data/, reports/)

[x] Virtual environment + dependencies (FastAPI, PyTorch, scikit-learn, joblib, pydantic, uv)

[x]Config via app/core/config.py and .env

[x] Logging via app/core/logger.py (env-aware)

## 2. Data & Preprocessing

[x] Data loaders (app/services/data_loader.py)

[]Train/validation splits

[] Preprocessing pipeline (app/utils/preprocessing.py)

[] Save processed datasets (.joblib)

3. Model Training

[] PyTorch MLP (app/models/model.py)

[] Training script (app/scripts/train.py) with command-line epochs and batch_size

[] Save trained model (.pt)

## 4. FastAPI Microservice

[] /predict endpoint (app/api/v1/predict.py)

[] Pydantic schema for input validation

[] Preprocessing + model inference

[] Swagger docs + example payload

[] /health endpoint for monitoring

## 5. Testing & CI

[x] Linting (ruff, pylint)

[x] Formatting (black)

[x] Type checking (mypy)

[] Unit tests for preprocessing & model loader

[] Integration tests for /predict (high priority)

## 6. Deployment Prep

[] Dockerize FastAPI service

[] Environment-based config for dev/staging/prod

[] Logging + monitoring setup

[]CI/CD pipeline for lint, tests, typecheck

## 🌱 Future Enhancements (Medium/Low Priority)

## Model & MLOps

[] Hyperparameter tuning

[] Model versioning & MLflow tracking

[] Batch prediction support

## API & Features

[] Edge-case handling (missing/extra features, wrong types)

[] Explainability (SHAP, permutation importance)

[] Continuous training/retraining pipeline

## Observability & Monitoring

[] Structured JSON logging

[] Prometheus/Grafana metrics

[] Alerts on model/API issues


## Next steps:
[] preprocees from [eda findings](./eda_findings.md)
[] AutoML benchmarking - without feature engineering
[]  AutoML benchmarking - with feature engineering
