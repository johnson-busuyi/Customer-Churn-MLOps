import pandas as pd
import joblib


# Load the trained model artifact
model = joblib.load("models/churn_model.pkl")


# Create a new customer
new_customer = pd.DataFrame(
    [
        {
            "age": 27,
            "monthly_charges": 89,
            "tenure_months": 4,
            "support_calls": 5,
            "contract_type": "monthly",
        }
    ]
)


# Ask the trained model to make a prediction
prediction = model.predict(new_customer)


print("Customer information:")
print(new_customer)

print("\nPrediction:")
print(f"Customer churn: {prediction[0]}")
