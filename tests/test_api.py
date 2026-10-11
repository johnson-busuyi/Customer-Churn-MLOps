from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Customer Churn Prediction API is running"
    }


def test_predict():
    customer = {
        "age": 27,
        "monthly_charges": 89,
        "tenure_months": 4,
        "support_calls": 5,
        "contract_type": "monthly",
    }

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 200
    assert "churn_prediction" in response.json()
