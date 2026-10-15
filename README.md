## Azure DevOps Automation
This project uses Azure DevOps for automated CI/CD testing.

An end-to-end MLOps portfolio project that trains and evaluates customer-churn models, tracks experiments with MLflow, serves predictions through FastAPI, tests the application with Pytest, packages it with Docker, scans the container with Trivy, and uses GitHub Actions to publish validated images to Docker Hub.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [3–4 Minute Interview Summary](#34-minute-interview-summary)
3. [Business Problem](#business-problem)
4. [Architecture](#architecture)
5. [MLOps Lifecycle](#mlops-lifecycle)
6. [Technology Stack and Why It Was Used](#technology-stack-and-why-it-was-used)
7. [Verified Versions](#verified-versions)
8. [Project Structure](#project-structure)
9. [Environment Setup](#environment-setup)
10. [Dataset](#dataset)
11. [Machine-Learning Concepts](#machine-learning-concepts)
12. [Logistic Regression Training](#logistic-regression-training)
13. [Random Forest Training](#random-forest-training)
14. [Model Evaluation](#model-evaluation)
15. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
16. [Model Artifacts and Inference](#model-artifacts-and-inference)
17. [FastAPI Prediction Service](#fastapi-prediction-service)
18. [Automated Testing with Pytest](#automated-testing-with-pytest)
19. [Docker Containerization](#docker-containerization)
20. [Git and GitHub](#git-and-github)
21. [GitHub Actions CI/CD](#github-actions-cicd)
22. [Trivy Container Security](#trivy-container-security)
23. [Trivy Root-Cause Investigation](#trivy-root-cause-investigation)
24. [Docker Hub Publishing](#docker-hub-publishing)
25. [Final Deployment Verification](#final-deployment-verification)
26. [Troubleshooting](#troubleshooting)
27. [Security Considerations](#security-considerations)
28. [CI/CD vs MLOps](#cicd-vs-mlops)
29. [Lessons Learned](#lessons-learned)
30. [Interview Questions and Answers](#interview-questions-and-answers)
31. [Resume-Ready Project Bullets](#resume-ready-project-bullets)
32. [Future Improvements](#future-improvements)
33. [Final Outcome](#final-outcome)

---

## Project Overview

The goal of this project was to build more than a machine-learning notebook. I wanted to implement the operational lifecycle around a model: data preparation, training, evaluation, experiment tracking, artifact management, API serving, automated testing, containerization, vulnerability scanning, CI/CD, container publishing, and final deployment validation.

The resulting workflow is:

```text
Customer Dataset
       |
       v
Python / Pandas
       |
       v
Data Preparation
       |
       v
Scikit-learn Model Training
       |
       v
Model Evaluation
Accuracy / Precision / Recall / F1
       |
       v
MLflow Experiment Tracking
       |
       v
Model Artifact (.pkl)
       |
       v
FastAPI Prediction API
       |
       v
Pytest
       |
       v
Docker
       |
       v
Trivy Security Scan
       |
       v
GitHub Actions CI/CD
       |
       v
Docker Hub
       |
       v
Pull Published Image
       |
       v
Live /predict Inference
```

This project intentionally stays at a practical, medium MLOps level. It does not use Kubernetes, Kubeflow, Airflow, feature stores, GPUs, or deep-learning frameworks.

---

## 3–4 Minute Interview Summary

> One of my recent portfolio projects was an end-to-end **Customer Churn MLOps pipeline**. The objective was not just to train a machine-learning model, but to build the complete workflow for training, tracking, testing, securing, packaging, and automatically delivering the model as a usable API.
>
> I started with a customer dataset containing features such as **age, monthly charges, tenure, support calls, and contract type**, with customer churn as the target variable. I used **Python and Pandas** for data handling and **scikit-learn 1.9.1** to preprocess the data and train the models.
>
> I trained both **Logistic Regression and Random Forest** models. I evaluated them using **accuracy, precision, recall, and F1 score** rather than relying on accuracy alone. Because this was a deliberately small demonstration dataset, both models achieved very high scores, so I would not interpret those results as production-level model performance.
>
> For the MLOps component, I integrated **MLflow 3.17.0**. MLflow allowed me to track individual training runs, model parameters, evaluation metrics, and model artifacts. This made it possible to compare Logistic Regression and Random Forest experiments instead of manually keeping track of model results.
>
> After training, I saved the selected model as a reusable artifact using **Joblib**. I then built a **FastAPI** application that loads the trained model and exposes a `/predict` REST endpoint. A client can send customer information as JSON and receive a churn prediction such as `yes` or `no`.
>
> I added automated testing with **Pytest** to test both the health endpoint and prediction endpoint. Then I containerized the API using **Docker**, so the model, Python environment, dependencies, and API could run consistently as a deployable container.
>
> For CI/CD, I built a **GitHub Actions** pipeline. Every change pushed to the main branch automatically runs the tests, builds the Docker image, verifies package versions, and performs a **Trivy vulnerability scan** for HIGH and CRITICAL vulnerabilities. If the tests or security scan fail, the image is not published.
>
> One of the most valuable troubleshooting experiences was with Trivy. It initially reported four HIGH vulnerabilities involving `msgpack`, `urllib3`, and `setuptools`, even after I upgraded the packages. I verified the actual package versions inside the container and discovered that Trivy was detecting older dependency references from **pip's bundled `bom.cdx.json` SBOM**, rather than the active application packages. I reproduced the issue locally, identified the exact SBOM file, and configured a targeted exclusion while keeping the HIGH/CRITICAL security gate enabled.
>
> Finally, after all tests and security checks pass, GitHub Actions authenticates securely to **Docker Hub using GitHub Secrets**, tags the image with both `latest` and the **Git commit SHA**, and publishes it. The SHA provides traceability between a container image and the exact source-code revision that produced it.
>
> To validate the entire process, I pulled the published Docker image from Docker Hub into a fresh local container, launched the FastAPI service, and submitted a customer to `/predict`. The API returned **HTTP 200** with a churn prediction of `"yes"`.
>
> The project gave me hands-on experience with the complete MLOps lifecycle: **data preparation, model training, experiment tracking, model artifacts, API serving, automated testing, containerization, vulnerability scanning, CI/CD, artifact publishing, and deployment verification**.

A simple way to remember the project in an interview is:

```text
Data -> Train -> Evaluate -> Track -> Serve -> Test -> Secure -> Deliver
```

---

## Business Problem

Customer churn means a customer stops using a company's service. A churn-prediction model attempts to identify customers who may leave based on historical patterns.

For this demonstration, the model receives these customer characteristics:

- `age`
- `monthly_charges`
- `tenure_months`
- `support_calls`
- `contract_type`

The target variable is:

```text
churn = yes / no
```

The MLOps objective is to make the trained model reproducible, testable, trackable, secure, and deployable instead of leaving it as a local training script.

---

## Architecture

```text
                         +----------------------+
                         | Customer CSV Dataset |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Pandas / Python    |
                         | Data Preparation     |
                         +----------+-----------+
                                    |
                                    v
                   +----------------------------------+
                   |          scikit-learn            |
                   | Logistic Regression / Random     |
                   | Forest                           |
                   +----------------+-----------------+
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
          +----------------------+       +----------------------+
          | Evaluation Metrics   |       | MLflow               |
          | Accuracy             |       | Runs / Params        |
          | Precision            |       | Metrics / Models     |
          | Recall / F1          |       +----------+-----------+
          +----------+-----------+                  |
                     |                              |
                     +---------------+--------------+
                                     |
                                     v
                         +----------------------+
                         | Model Artifact       |
                         | churn_model.pkl      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | FastAPI / Uvicorn    |
                         | POST /predict        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Pytest               |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Docker Image         |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Trivy Security Scan  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | GitHub Actions       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Docker Hub           |
                         | latest + Git SHA     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Pull / Run / Verify  |
                         | HTTP 200 Prediction  |
                         +----------------------+
```

---

## MLOps Lifecycle

This project demonstrates the difference between simply creating an ML model and operationalizing it.

```text
Machine Learning
Data -> Train -> Evaluate -> Model

MLOps
Data -> Train -> Evaluate -> Track -> Package -> Test -> Secure -> Publish -> Verify
```

Training creates the model. MLOps builds the repeatable engineering process around that model.

---

## Technology Stack and Why It Was Used

| Technology | Purpose in this project |
|---|---|
| Python | Main programming language for training, inference, API, and tests |
| Pandas | Loads and manipulates the customer CSV dataset |
| scikit-learn | Preprocessing, model training, prediction, and evaluation |
| Joblib | Serializes the trained model to a reusable `.pkl` artifact |
| MLflow | Tracks experiments, parameters, metrics, runs, and model artifacts |
| FastAPI | Exposes the trained model through a REST prediction API |
| Uvicorn | ASGI server used to run FastAPI |
| Pytest | Automated API tests |
| HTTPX | HTTP client dependency used by FastAPI/Starlette testing |
| Docker | Packages the API, model, Python runtime, and dependencies |
| Trivy | Scans the Docker image for known vulnerabilities |
| Git | Source-control history |
| GitHub | Hosts the project repository |
| GitHub Actions | Automates testing, image building, scanning, and publishing |
| Docker Hub | Stores the final versioned container images |

### Why scikit-learn?

`scikit-learn` was appropriate because this project uses structured/tabular data and classical supervised classification algorithms. It provides preprocessing, model training, prediction, and evaluation in one mature Python library.

### Why MLflow?

Without experiment tracking, model results can become a collection of terminal outputs and files. MLflow creates a history of training attempts and associates parameters, metrics, and model artifacts with individual runs.

### Why FastAPI?

A `.pkl` model file is not directly useful to another application. FastAPI turns the model into a service that can receive JSON over HTTP and return predictions.

### Why Docker?

Docker packages the runtime environment around the application so that the API can be run consistently outside the developer's local Python environment.

### Why Trivy?

A working container can still contain vulnerable operating-system or application dependencies. Trivy adds a security gate before publishing the image.

### Why GitHub Actions?

GitHub Actions converts the manual build/test/scan/publish process into an automated CI/CD workflow triggered by source-control changes.

---

## Verified Versions

Versions explicitly verified during development include:

```text
Python          3.13.9 (local development environment)
scikit-learn    1.9.1
MLflow          3.17.0
pytest          9.1.1
Docker          24.0.7
Trivy           0.75.0
```

During the Trivy remediation investigation, the secure package versions verified inside the container included:

```text
msgpack         1.2.3
urllib3         2.8.0
setuptools      84.0.0
```

The Python Docker base observed during image inspection was Python 3.13.x; a later image-history inspection showed Python 3.13.16 in the base image.

> Version note: dependency versions can change over time unless fully pinned. The versions above record what was actually verified while this project was built.

---

## Project Structure

```text
Customer-Churn-MLOps/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   └── customers.csv
├── models/
│   ├── churn_model.pkl
│   └── churn_model_random_forest.pkl
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── predict.py
│   ├── train.py
│   └── train_random_forest.py
├── tests/
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

### Directory responsibilities

```text
data/      -> training data
models/    -> serialized trained model artifacts
src/       -> training, inference, and API code
tests/     -> automated tests
.github/   -> GitHub Actions workflow
```

---

## Environment Setup

### Create the project environment

From Git Bash on Windows:

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify scikit-learn:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

The verified result during the project was:

```text
1.9.1
```

### Main dependencies

The project requirements included:

```text
pandas
scikit-learn
joblib
mlflow
fastapi
uvicorn
pytest
httpx
msgpack>=1.2.1
urllib3>=2.8.0
setuptools>=78.1.1
```

The explicit `msgpack`, `urllib3`, and `setuptools` minimum versions were added during the container-security investigation.

---

## Dataset

The demonstration dataset is stored at:

```text
data/customers.csv
```

Dataset used during development:

```csv
age,monthly_charges,tenure_months,support_calls,contract_type,churn
25,85,3,5,monthly,yes
42,55,36,1,yearly,no
31,95,6,4,monthly,yes
55,45,60,0,yearly,no
28,75,8,3,monthly,yes
48,60,48,1,yearly,no
35,90,12,4,monthly,yes
52,50,54,0,yearly,no
23,80,2,6,monthly,yes
46,65,40,1,yearly,no
39,70,24,2,yearly,no
29,88,5,5,monthly,yes
58,42,72,0,yearly,no
33,92,7,4,monthly,yes
44,62,30,2,yearly,no
```

This dataset is intentionally small and easy to understand. It is suitable for demonstrating an MLOps workflow but **not** for claiming production-quality predictive performance.

---

## Machine-Learning Concepts

### Features

Features are the inputs used by the model:

```text
age
monthly_charges
tenure_months
support_calls
contract_type
```

### Target / Label

The target is what the model learns to predict:

```text
churn
```

### Training vs Inference

Training:

```text
Historical customer data
        |
        v
model.fit(...)
        |
        v
Trained model artifact
```

Inference:

```text
New customer data
        |
        v
Load trained artifact
        |
        v
model.predict(...)
        |
        v
yes / no
```

A useful DevOps analogy is:

```text
Java application:
Java source + pom.xml -> Maven -> JAR artifact

MLOps application:
Python + customer data -> scikit-learn -> trained model artifact
```

---

## Logistic Regression Training

The primary training script is:

```text
src/train.py
```

The workflow includes:

1. Load the CSV with Pandas.
2. Separate features (`X`) from target (`y`).
3. One-hot encode `contract_type`.
4. Pass numeric columns through the preprocessing pipeline.
5. Split the data into training and test sets.
6. Train Logistic Regression.
7. Generate predictions.
8. Calculate evaluation metrics.
9. Log the experiment to MLflow.
10. Save the trained model with Joblib.

Important training configuration included:

```python
OneHotEncoder(handle_unknown="ignore")
LogisticRegression(max_iter=1000)
train_test_split(..., test_size=0.20, random_state=42)
```

`handle_unknown="ignore"` helps the preprocessing pipeline tolerate a category at inference time that was not represented in the fitted encoder.

Example successful training output:

```text
Dataset loaded successfully
Number of customers: 15
Model accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 score: 1.00
Model saved to models/churn_model.pkl
Experiment recorded in MLflow
```

One recorded Logistic Regression run included:

```text
Run name: secretive-ox-568
Run ID: 8bd247dc40144f55abbf0fc51e73a139
```

---

## Random Forest Training

A second training script was created:

```text
src/train_random_forest.py
```

The Random Forest configuration included:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

MLflow parameters included:

```text
algorithm = Random Forest
test_size = 0.20
n_estimators = 100
```

The model was also saved locally:

```text
models/churn_model_random_forest.pkl
```

Successful MLflow runs included names such as:

```text
sincere-loon-790
intrigued-eel-4
upset-jay-589
```

### MLflow / skops trusted-type error

While logging the Random Forest model, MLflow/skops raised:

```text
skops.io.exceptions.UntrustedTypesFoundException:
['sklearn.tree._tree.Tree']
```

Because the model was created locally in this trusted training process, model logging was configured with:

```python
mlflow.sklearn.log_model(
    model,
    name="churn_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
```

This should not be treated as permission to trust arbitrary serialized models from unknown sources. The trust decision was limited to the locally created model type in this project.

---

## Model Evaluation

The project evaluates models using:

```text
Accuracy
Precision
Recall
F1 Score
```

### Why not use accuracy alone?

Accuracy answers how many predictions were correct overall. In many churn datasets, classes can be imbalanced, so accuracy alone may hide poor performance on the churn class.

Precision helps answer:

> Of customers predicted to churn, how many actually churned?

Recall helps answer:

> Of customers who actually churned, how many did the model identify?

F1 balances precision and recall.

### Important limitation

Both Logistic Regression and Random Forest achieved `1.00` metrics on this demonstration dataset. That is **not evidence that the models would achieve 100% performance in production**.

The dataset contains only 15 deliberately simple rows. The results demonstrate that the engineering pipeline works; a real model would require substantially more representative data, validation, bias analysis, monitoring, and performance testing.

---

## MLflow Experiment Tracking

Verified MLflow version:

```text
3.17.0
```

Experiment name:

```text
Customer Churn Prediction
```

Tracking URI used during development:

```text
http://127.0.0.1:8000
```

### MLflow terminology

**Experiment** — the overall ML problem or collection of related training runs.

**Run** — one specific model-training attempt.

**Parameter** — configuration used to train a model, such as algorithm or number of trees.

**Metric** — measured model performance, such as F1 or accuracy.

**Artifact** — an output associated with the run, such as a model.

### Starting MLflow

The working command was:

```bash
mlflow ui --host 127.0.0.1 --port 8000
```

Then open the MLflow UI in a browser on port `8000`.

### MLflow 3.x UI

MLflow 3.17 opened into a newer GenAI-oriented interface. Traditional model-training runs were available under:

```text
Model training -> Runs
```

This was used to compare Logistic Regression and Random Forest runs, parameters, and metrics.

---

## Model Artifacts and Inference

The selected Logistic Regression pipeline is serialized with Joblib:

```text
models/churn_model.pkl
```

The Random Forest model is stored separately:

```text
models/churn_model_random_forest.pkl
```

The standalone prediction script is:

```text
src/predict.py
```

Its purpose is to demonstrate inference without retraining:

```text
Load churn_model.pkl
       |
       v
Create new customer DataFrame
       |
       v
model.predict(...)
       |
       v
churn prediction
```

A test customer produced a `yes` prediction during development.

---

## FastAPI Prediction Service

The API implementation is:

```text
src/api.py
```

It loads the serialized model and exposes two endpoints:

```text
GET  /
POST /predict
```

### Health/root endpoint

Example response:

```json
{
  "message": "Customer Churn Prediction API is running"
}
```

### Prediction endpoint

Example request:

```json
{
  "age": 29,
  "monthly_charges": 90,
  "tenure_months": 4,
  "support_calls": 5,
  "contract_type": "monthly"
}
```

Final verified response from the Docker Hub image:

```json
{
  "churn_prediction": "yes"
}
```

HTTP response:

```text
200 OK
```

### Swagger/OpenAPI

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

The Swagger interface was used to test both `/` and `/predict`.

---

## Automated Testing with Pytest

Tests are stored in:

```text
tests/test_api.py
```

Two API tests were created:

```text
test_home
test_predict
```

Run locally with:

```bash
pytest -v
```

Verified successful result:

```text
platform win32 -- Python 3.13.9, pytest-9.1.1
collected 2 items

tests/test_api.py::test_home PASSED
tests/test_api.py::test_predict PASSED

2 passed, 1 warning in 8.41s
```

A Starlette deprecation warning related to its HTTP test-client stack was observed, but the tests themselves passed.

---

## Docker Containerization

Docker version verified locally:

```text
Docker version 24.0.7, build afdd53b
```

The Dockerfile packages the FastAPI application and model artifact.

Final Dockerfile structure used during the security remediation:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel msgpack urllib3 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade setuptools msgpack urllib3 \
    && python -m pip check

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build

```bash
docker build -t customer-churn-api:1.0 .
```

A clean security-testing build was also created with:

```bash
docker build --no-cache -t customer-churn-api:secure .
```

### Run

The initial attempt was:

```bash
docker run -d --name customer-churn-api -p 8080:8080 customer-churn-api:1.0
```

Port `8080` was already allocated by another local container, so the host-side port was changed:

```bash
docker run -d --name customer-churn-api -p 8081:8080 customer-churn-api:1.0
```

The container still listens on `8080`; only the host port changed to `8081`.

---

## Git and GitHub

Repository:

```text
johnson-busuyi/Customer-Churn-MLOps
```

GitHub repository address:

```text
https://github.com/johnson-busuyi/Customer-Churn-MLOps
```

Typical initialization flow:

```bash
git init
git branch -M main
git remote add origin https://github.com/johnson-busuyi/Customer-Churn-MLOps.git
git add .
git commit -m "Initial Customer Churn MLOps project"
git push -u origin main
```

Initial commit recorded during development:

```text
ffa5c93 Initial Customer Churn MLOps project
```

Other project milestones included commits for GitHub Actions, Docker builds, Trivy scanning, package verification, the Trivy SBOM fix, Docker Hub authentication, and image publishing.

---

## GitHub Actions CI/CD

Workflow file:

```text
.github/workflows/ci-cd.yml
```

The pipeline evolved incrementally rather than being written all at once. This made each stage independently testable.

Final logical flow:

```text
Push / Pull Request
        |
        v
Checkout
        |
        v
Python 3.13
        |
        v
Install Dependencies
        |
        v
Pytest
        |
        v
Docker Build
        |
        v
Verify Package Versions
        |
        v
Trivy HIGH/CRITICAL Scan
        |
        v
Docker Hub Login
        |
        v
Tag Image
   latest + Git SHA
        |
        v
Push to Docker Hub
```

### Test job

Core test stage:

```yaml
jobs:
  test:
    name: Test Python Application
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Pytest
        run: pytest -v
```

### Docker job

The Docker job depends on the test job:

```yaml
needs: test
```

That dependency is important because a container should not be published when application tests fail.

The image is built as:

```bash
docker build -t customer-churn-api:latest .
```

### Package verification

During security troubleshooting, the pipeline explicitly verified package versions inside the image:

```yaml
- name: Verify package versions inside Docker image
  run: |
    docker run --rm customer-churn-api:latest python -c "import msgpack, urllib3, setuptools; print('msgpack:', msgpack.__version__); print('urllib3:', urllib3.__version__); print('setuptools:', setuptools.__version__)"
    docker run --rm customer-churn-api:latest python -m pip show msgpack urllib3 setuptools
```

The GitHub runner confirmed:

```text
msgpack: 1.2.3
urllib3: 2.8.0
setuptools: 84.0.0
```

### Trivy gate

The security scan keeps a failing exit code for real HIGH/CRITICAL findings:

```yaml
- name: Scan Docker image with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: "customer-churn-api:latest"
    format: "table"
    severity: "CRITICAL,HIGH"
    exit-code: "1"
    ignore-unfixed: true
    skip-files: "**/pip/_vendor/bom.cdx.json"
```

`exit-code: "1"` is important. The project did **not** solve the security problem by setting Trivy to always succeed.

### Docker Hub login

Credentials are stored in GitHub repository secrets:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

The workflow uses:

```yaml
- name: Login to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

The Docker Hub token is never stored directly in source control.

### Image tagging and publishing

```yaml
- name: Tag Docker image
  run: |
    docker tag customer-churn-api:latest ${{ secrets.DOCKERHUB_USERNAME }}/customer-churn-mlops:latest
    docker tag customer-churn-api:latest ${{ secrets.DOCKERHUB_USERNAME }}/customer-churn-mlops:${{ github.sha }}

- name: Push Docker image to Docker Hub
  run: |
    docker push ${{ secrets.DOCKERHUB_USERNAME }}/customer-churn-mlops:latest
    docker push ${{ secrets.DOCKERHUB_USERNAME }}/customer-churn-mlops:${{ github.sha }}
```

Two tags are deliberately published:

```text
latest
<Git commit SHA>
```

`latest` provides a convenient current image. The Git SHA provides traceability and identifies the exact source revision that created an image.

---

## Trivy Container Security

Trivy was configured to fail the CI/CD pipeline when it detected fixed HIGH or CRITICAL vulnerabilities.

Initial scan result:

```text
Total: 4 (HIGH: 4, CRITICAL: 0)
```

Findings included:

```text
msgpack     1.1.2   -> fixed in 1.2.1
setuptools  70.3.0  -> fixed in 78.1.1
urllib3     2.7.0   -> fixed in 2.8.0
urllib3     2.7.0   -> fixed in 2.8.0
```

The vulnerabilities included:

```text
GHSA-6v7p-g79w-8964
CVE-2025-47273
CVE-2026-97687
CVE-2026-97689
```

Rather than disabling the scan, the packages were first explicitly upgraded in `requirements.txt` and the Docker build.

---

## Trivy Root-Cause Investigation

This became the most significant troubleshooting exercise in the project.

### 1. Secure versions were installed locally

The local environment showed:

```text
msgpack: 1.2.3
urllib3: 2.8.0
setuptools: 84.0.0
```

### 2. Secure versions were verified inside the Docker image

The GitHub Actions diagnostic step also showed:

```text
msgpack: 1.2.3
urllib3: 2.8.0
setuptools: 84.0.0
```

But Trivy still reported:

```text
msgpack     1.1.2
setuptools  70.3.0
urllib3     2.7.0
```

This proved the problem was not simply a failed `pip install`.

### 3. Reproduce the GitHub result locally

Trivy was not installed directly on Windows:

```bash
trivy --version
```

returned:

```text
bash: trivy: command not found
```

Instead of adding another Windows installation, Trivy was run through Docker:

```bash
docker run --rm \
  -v //var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  customer-churn-api:secure
```

The same four HIGH findings appeared locally. This proved the behavior was not specific to GitHub Actions.

### 4. Verify Trivy version

```bash
docker run --rm aquasec/trivy:latest --version
```

Result:

```text
Version: 0.75.0
```

### 5. Search for old `.dist-info` directories

```bash
docker run --rm customer-churn-api:secure sh -c "find /usr/local/lib/python3.13/site-packages -type d \( -name 'msgpack-1.1.2.dist-info' -o -name 'urllib3-2.7.0.dist-info' -o -name 'setuptools-70.3.0.dist-info' \) -print"
```

The command returned no matches.

That was an important clue: the final application package metadata did not contain those old versions.

### 6. Discover pip's bundled SBOM

The image contained:

```text
/usr/local/lib/python3.13/site-packages/pip/_vendor/bom.cdx.json
```

It was found with:

```bash
docker run --rm customer-churn-api:secure sh -c "find /usr/local/lib/python3.13/site-packages/pip -name 'bom.cdx.json' -print"
```

### 7. Inspect the SBOM

```bash
docker run --rm customer-churn-api:secure sh -c "grep -E 'msgpack|urllib3|setuptools|1.1.2|2.7.0|70.3.0' /usr/local/lib/python3.13/site-packages/pip/_vendor/bom.cdx.json | head -40"
```

The output contained exactly the versions Trivy reported:

```text
pkg:pypi/msgpack@1.1.2
version: 1.1.2

pkg:pypi/setuptools@70.3.0
version: 70.3.0

pkg:pypi/urllib3@2.7.0
version: 2.7.0
```

### 8. Verify pip itself

```bash
docker run --rm customer-churn-api:secure python -m pip --version
```

Result:

```text
pip 26.2.1 from /usr/local/lib/python3.13/site-packages/pip (python 3.13)
```

### 9. Targeted diagnostic exclusion

A Trivy scan was run while excluding only pip's bundled SBOM:

```bash
docker run --rm \
  -v //var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  --skip-files "**/pip/_vendor/bom.cdx.json" \
  customer-churn-api:secure
```

The active package metadata, including `urllib3-2.8.0.dist-info/METADATA`, scanned cleanly and the four findings disappeared.

### 10. CI/CD resolution

The targeted exclusion was added to the Trivy workflow while retaining:

```yaml
exit-code: "1"
```

The next GitHub Actions run passed.

### Why this matters

The goal was not to make a red pipeline green at any cost. The troubleshooting process established:

```text
1. What Trivy reported
2. What versions Python actually imported
3. What versions pip reported as installed
4. Whether the old metadata existed in normal site-packages
5. Where Trivy's version references originated
6. Whether the behavior reproduced outside GitHub Actions
7. Whether a narrowly targeted exclusion removed only that source
```

The exclusion is therefore documented and intentional rather than an unexplained global vulnerability suppression.

> Security note: scanner exclusions should be periodically reviewed. A future pip/Trivy release may change this behavior, allowing the exclusion to be removed.

---

## Docker Hub Publishing

Docker Hub repository:

```text
bustopsy777/customer-churn-mlops
```

Repository description:

```text
Customer churn MLOps pipeline with MLflow, FastAPI, Docker, Trivy, and GitHub Actions.
```

The pipeline publishes:

```text
bustopsy777/customer-churn-mlops:latest
bustopsy777/customer-churn-mlops:<git-sha>
```

### Why use Git SHA tags?

If `latest` changes over time, it does not by itself identify which source revision produced an older deployment.

A Git SHA tag gives traceability:

```text
Git commit
    |
    v
CI/CD build
    |
    v
Docker image tagged with same SHA
```

This is useful for debugging, auditing, and rollback decisions.

---

## Final Deployment Verification

Publishing an image successfully is not enough. The published artifact was pulled back from Docker Hub and tested independently.

### Pull the image

```bash
docker pull bustopsy777/customer-churn-mlops:latest
```

### Run the published image

Port `8082` was used on the host for the final validation:

```bash
docker run -d \
  --name churn-mlops-test \
  -p 8082:8080 \
  bustopsy777/customer-churn-mlops:latest
```

Verify:

```bash
docker ps
```

Swagger UI was then available on the local host at port `8082` under `/docs`.

### Final prediction

Request:

```json
{
  "age": 29,
  "monthly_charges": 90,
  "tenure_months": 4,
  "support_calls": 5,
  "contract_type": "monthly"
}
```

Verified server response:

```text
HTTP 200
```

```json
{
  "churn_prediction": "yes"
}
```

This validated the complete path:

```text
Source Code
    |
    v
GitHub
    |
    v
GitHub Actions
    |
    v
Tests
    |
    v
Docker Build
    |
    v
Trivy Scan
    |
    v
Docker Hub
    |
    v
Docker Pull
    |
    v
FastAPI Container
    |
    v
Model Inference
    |
    v
HTTP 200
```

---

## Troubleshooting

### MLflow could not bind to ports 5000 or 5001

Commands using MLflow's common default ports failed on Windows with:

```text
WinError 10013
```

Windows excluded-port ranges were inspected. A simple Python HTTP server was tested on port `8000`:

```bash
python -m http.server 8000
```

Because port `8000` worked, MLflow was started there:

```bash
mlflow ui --host 127.0.0.1 --port 8000
```

**Lesson:** isolate whether the problem is the application itself or the network/port before changing application code.

---

### MLflow UnicodeEncodeError on Windows

Training and MLflow logging completed, but terminal output failed when MLflow attempted to print Unicode characters to a Windows `cp1252` console.

Error category:

```text
UnicodeEncodeError: 'charmap' codec can't encode character ...
```

Fix in Git Bash:

```bash
export PYTHONUTF8=1
```

After enabling UTF-8 mode, the run completed without the terminal encoding failure.

---

### Random Forest MLflow trusted-type error

Error:

```text
skops.io.exceptions.UntrustedTypesFoundException:
['sklearn.tree._tree.Tree']
```

For the locally trained and trusted model:

```python
mlflow.sklearn.log_model(
    model,
    name="churn_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
```

**Lesson:** serialized ML models can introduce security risks. Do not blindly trust model types from untrusted sources.

---

### FastAPI TestClient missing HTTPX

Initial Pytest execution failed because FastAPI/Starlette's test client required `httpx`.

Fix:

```bash
pip install httpx
```

`httpx` was then added to `requirements.txt` so CI would receive the same dependency.

**Lesson:** if a dependency is required for repeatable tests, it must be declared rather than only installed manually on one workstation.

---

### Pytest could not import `src`

Error:

```text
ModuleNotFoundError: No module named 'src'
```

The project added:

```text
src/__init__.py
pytest.ini
```

After the Python project/test path was configured correctly:

```text
2 passed
```

---

### Docker port 8080 already allocated

Initial command:

```bash
docker run -d --name customer-churn-api -p 8080:8080 customer-churn-api:1.0
```

Error:

```text
Bind for 0.0.0.0:8080 failed: port is already allocated
```

Another local container was already using host port `8080`.

Fix:

```bash
docker run -d --name customer-churn-api -p 8081:8080 customer-churn-api:1.0
```

Final Docker Hub verification later used:

```bash
-p 8082:8080
```

**Lesson:** the left side of `HOST:CONTAINER` can change without modifying the port the application listens to inside the container.

---

### Trivy reported old packages after upgrades

Symptom:

```text
Runtime package: msgpack 1.2.3
Trivy report:    msgpack 1.1.2

Runtime package: urllib3 2.8.0
Trivy report:    urllib3 2.7.0

Runtime package: setuptools 84.0.0
Trivy report:    setuptools 70.3.0
```

Root cause investigation found the old references in:

```text
pip/_vendor/bom.cdx.json
```

A narrowly scoped `skip-files` rule was used after reproducing and proving the discrepancy. The HIGH/CRITICAL failure gate remained enabled.

---

### Windows LF / CRLF warning

Git displayed messages similar to:

```text
LF will be replaced by CRLF the next time Git touches it
```

This was a Windows line-ending warning, not a GitHub Actions pipeline failure.

---

### GitHub Actions platform warnings

During development, GitHub emitted warnings about action runtimes moving from Node.js 20 toward Node.js 24 and changes to the `ubuntu-latest` runner label.

These were platform/deprecation warnings rather than the cause of the pipeline failures observed during this project.

A production hardening pass should periodically update and pin third-party actions.

---

## Security Considerations

Security controls demonstrated in this project include:

- No Docker Hub token stored in source code.
- GitHub repository secrets used for Docker Hub authentication.
- HIGH/CRITICAL Trivy findings configured to fail CI.
- `ignore-unfixed: true` used so the gate focuses on findings for which a fix is available.
- Vulnerable application dependency versions were explicitly upgraded.
- Package versions were verified inside the built container.
- Scanner discrepancies were reproduced locally before applying an exclusion.
- Trivy's `exit-code: "1"` remained enabled.
- Published images are tagged with the Git commit SHA for traceability.
- Final published artifacts are pulled and functionally validated.

### Production hardening opportunities

For a real production service, additional work would include:

- Pin Python dependencies more comprehensively and use lockfiles/hashes.
- Pin GitHub Actions to trusted immutable commit SHAs.
- Use a non-root container user.
- Add container health checks.
- Generate and publish an application SBOM.
- Sign container images and verify signatures before deployment.
- Add secret scanning and SAST.
- Separate training dependencies from runtime dependencies.
- Add centralized logging and metrics.
- Add authentication/authorization to the prediction API if required.
- Add TLS at the ingress/reverse-proxy/platform layer.
- Periodically reassess the targeted pip SBOM exclusion.

---

## CI/CD vs MLOps

CI/CD and MLOps overlap, but they are not identical.

### CI/CD responsibilities in this project

```text
Source change
-> automated tests
-> Docker build
-> vulnerability scan
-> image tagging
-> Docker Hub publication
```

### ML/MLOps responsibilities

```text
Dataset
-> preprocessing
-> model training
-> evaluation
-> experiment tracking
-> model artifact
-> inference API
```

### Combined pipeline

```text
ML lifecycle + software delivery lifecycle = MLOps system
```

The model is treated as a versioned software artifact that must be tested, secured, packaged, and delivered consistently.

---

## Lessons Learned

### 1. A trained model is only one part of an ML system

The difficult engineering work often begins after `model.fit()` succeeds.

### 2. Reproducibility matters

Dependencies, preprocessing, model artifacts, tests, and container configuration need to be captured so the service behaves consistently outside the original development shell.

### 3. Experiment tracking improves model engineering

MLflow makes it easier to compare algorithms and preserve the relationship between configuration, metrics, and artifacts.

### 4. Security scanners require investigation, not blind obedience or blind suppression

The Trivy issue demonstrated that a scanner result should be traced to the actual component being detected. The correct response is neither to ignore every finding nor to assume every finding means the actively imported application package is vulnerable.

### 5. CI/CD should fail safely

Tests and security checks happen before publishing. A failed stage prevents the image from progressing to Docker Hub.

### 6. Artifact traceability is important

Tagging a container with the Git SHA connects the deployed binary artifact back to source control.

### 7. Validate the published artifact

A green build does not prove the registry artifact works. Pulling the image back from Docker Hub and successfully calling `/predict` validated the artifact that a downstream environment would actually consume.

### 8. Model metrics need context

A `1.00` F1 score on a 15-row demonstration dataset should not be presented as production model quality. Good engineering includes knowing the limitations of the evidence.

---

## Interview Questions and Answers

### What did you build?

I built an end-to-end customer-churn MLOps pipeline that trains and evaluates classification models, tracks experiments in MLflow, exposes the selected model through FastAPI, tests the API with Pytest, packages it in Docker, scans the image with Trivy, and publishes successful builds to Docker Hub through GitHub Actions.

### Why is this an MLOps project rather than just an ML project?

Because it covers the operational lifecycle around the model: experiment tracking, artifact management, serving, testing, containerization, security scanning, automated delivery, versioning, and deployment validation.

### Why did you use MLflow?

I used MLflow to maintain a history of training runs and associate model parameters, evaluation metrics, and artifacts with each run. It made comparing Logistic Regression and Random Forest experiments much more systematic.

### Why did you use both Logistic Regression and Random Forest?

I wanted to demonstrate model experimentation rather than training a single algorithm. Logistic Regression provided a simple linear classification baseline, while Random Forest provided a tree-based alternative that could be tracked and compared in MLflow.

### Why did both models score 100%?

The dataset contains only 15 deliberately simple demonstration records. The result shows the pipeline is functioning, but it is not enough evidence to claim production-level predictive performance. A production project would require a much larger representative dataset and stronger validation.

### What happens when code is pushed to GitHub?

GitHub Actions installs dependencies and runs Pytest. If tests pass, it builds the Docker image, verifies important dependency versions, runs Trivy for HIGH and CRITICAL vulnerabilities, authenticates to Docker Hub, tags the image with `latest` and the Git SHA, and pushes both tags.

### Why use a Git SHA Docker tag?

It gives traceability. I can map a container image directly to the exact source-code commit that produced it, which helps with debugging, auditing, and rollback.

### What was the most difficult issue?

The Trivy security scan was the most interesting. Trivy reported four HIGH vulnerabilities even though the active packages inside the image were upgraded. I reproduced the issue locally, verified the runtime and pip versions, searched the filesystem, and found that the old versions came from pip's bundled `bom.cdx.json`. I then used a targeted exclusion for that file while keeping the HIGH/CRITICAL failure gate enabled.

### Why not simply set Trivy's exit code to zero?

That would remove the security gate. I wanted the pipeline to continue failing when genuine HIGH or CRITICAL vulnerabilities are detected, so I investigated and narrowly handled the specific source instead.

### How did you know the Docker Hub image really worked?

I pulled `bustopsy777/customer-churn-mlops:latest` back from Docker Hub, started it as a new container, opened the FastAPI Swagger interface, submitted a customer to `/predict`, and received HTTP 200 with `{"churn_prediction":"yes"}`.

### What would you improve for production?

I would use a larger production-quality dataset, separate training and serving dependencies, use stricter dependency locking, run the container as non-root, pin GitHub Actions by commit SHA, add image signing and SBOM generation, implement observability and model monitoring, and deploy behind an authenticated TLS-enabled production platform.

---

## Resume-Ready Project Bullets

- Built an end-to-end **MLOps customer-churn prediction pipeline** using Python, Pandas, scikit-learn, MLflow, FastAPI, Pytest, Docker, Trivy, GitHub Actions, and Docker Hub.
- Trained and compared **Logistic Regression and Random Forest** classification models and tracked parameters, accuracy, precision, recall, F1 score, and model artifacts with **MLflow**.
- Developed a **FastAPI REST inference service** that loads a serialized scikit-learn model and returns real-time churn predictions from JSON customer data.
- Implemented **GitHub Actions CI/CD** to automatically test the API, build Docker images, verify package versions, enforce HIGH/CRITICAL Trivy vulnerability gates, and publish validated images to Docker Hub.
- Implemented container traceability using both **`latest` and Git commit SHA tags**, securely authenticating to Docker Hub with GitHub Secrets.
- Troubleshot a complex **Trivy/pip SBOM vulnerability-reporting discrepancy**, reproducing the issue locally, verifying runtime dependency versions, identifying pip's bundled `bom.cdx.json` as the finding source, and applying a targeted scan exclusion while preserving the security gate.
- Validated the full delivery lifecycle by pulling the published Docker Hub artifact, running the container independently, and successfully executing a FastAPI `/predict` request with an **HTTP 200** response.

---

## Future Improvements

This project can be extended without changing its core architecture.

### Model/data improvements

- Replace the 15-row demonstration dataset with a larger realistic churn dataset.
- Add cross-validation.
- Add class-imbalance handling where appropriate.
- Add ROC-AUC and confusion-matrix analysis.
- Add feature engineering.
- Add model-selection criteria rather than manually selecting a model.
- Add model/data drift monitoring.

### MLOps improvements

- Introduce a model registry and formal promotion stages.
- Separate training and serving dependency files.
- Automatically retrain based on approved triggers.
- Add model version metadata to API responses or health endpoints.
- Store experiment artifacts in remote object storage.

### DevSecOps improvements

- Pin GitHub Actions to immutable SHAs.
- Add SAST and secret scanning.
- Produce an SBOM for each released image.
- Sign container images.
- Add policy enforcement before deployment.
- Run the container as a non-root user.

### Deployment improvements

- Deploy the published image to a managed container platform or Kubernetes.
- Add HTTPS/TLS and authentication.
- Add Prometheus-compatible application metrics and dashboards.
- Add centralized logs, alerts, and health checks.
- Implement controlled rollout and rollback using immutable image tags.

---

## Final Outcome

The completed project successfully demonstrated:

```text
Data Preparation             DONE
Model Training               DONE
Multiple Model Experiments   DONE
Accuracy / Precision         DONE
Recall / F1                  DONE
MLflow Tracking              DONE
Model Serialization          DONE
FastAPI Serving              DONE
Swagger Testing              DONE
Pytest Automation            DONE
Docker Containerization      DONE
Trivy Security Scanning      DONE
Security Troubleshooting     DONE
GitHub Actions CI/CD         DONE
Docker Hub Authentication    DONE
Image SHA Tagging            DONE
Docker Hub Publishing        DONE
Published Image Pull         DONE
Live Prediction Test         DONE
HTTP 200 Verification        DONE
```

Final validated inference:

```json
{
  "churn_prediction": "yes"
}
```

The most important result is not the prediction itself. The project demonstrates that an ML model can move through a controlled engineering lifecycle from **data and training to a tested, tracked, security-scanned, versioned, published, and independently verified deployable service**.

---

## Quick Interview Memory Aid

```text
DATA
  -> Pandas

TRAIN
  -> scikit-learn
  -> Logistic Regression + Random Forest

EVALUATE
  -> Accuracy + Precision + Recall + F1

TRACK
  -> MLflow

ARTIFACT
  -> Joblib .pkl

SERVE
  -> FastAPI + Uvicorn

TEST
  -> Pytest

PACKAGE
  -> Docker

SECURE
  -> Trivy

AUTOMATE
  -> GitHub Actions

PUBLISH
  -> Docker Hub

VERIFY
  -> Pull image -> /predict -> HTTP 200
```

**Interview phrase:**

> I didn't stop at training the model. I built the operational lifecycle around it—experiment tracking, artifact management, API serving, automated testing, containerization, vulnerability scanning, CI/CD, versioned publishing, and final deployment validation.
