import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# Create the FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting whether a customer will churn.",
    version="1.0.0",
)


# Load the trained model
model = joblib.load("models/churn_model.pkl")


# Define the customer data expected by the API
class Customer(BaseModel):
    age: int
    monthly_charges: float
    tenure_months: int
    support_calls: int
    contract_type: str


# Health/root endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: Customer):

    # Convert incoming customer data into a DataFrame
    customer_data = pd.DataFrame(
        [customer.model_dump()]
    )

    # Make prediction
    prediction = model.predict(customer_data)

    # Return prediction
    return {
        "churn_prediction": prediction[0]
    }
