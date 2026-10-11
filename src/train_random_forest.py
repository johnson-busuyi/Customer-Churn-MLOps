import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# 1. Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8000")
mlflow.set_experiment("Customer Churn Prediction")


# 2. Load customer data
data = pd.read_csv("data/customers.csv")

print("Dataset loaded successfully")
print(f"Number of customers: {len(data)}")


# 3. Separate features from target
X = data.drop("churn", axis=1)
y = data["churn"]


# 4. Identify categorical columns
categorical_features = ["contract_type"]


# 5. Convert categorical data into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)


# 6. Create Random Forest model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)


# 7. Split data into training and testing sets
test_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
)


# 8. Start an MLflow experiment run
with mlflow.start_run():

    # Train the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions using the test data
    predictions = model.predict(X_test)

    # 9. Calculate model evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="yes",
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="yes",
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="yes",
    )

    # 10. Record training parameters in MLflow
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("n_estimators", 100)

    # 11. Record model performance metrics in MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 12. Save the trained model in MLflow
    # The sklearn Tree type is explicitly trusted because
    # this Random Forest model was created and trained locally.
    mlflow.sklearn.log_model(
        model,
        name="churn_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # 13. Save the Random Forest model locally
    joblib.dump(
        model,
        "models/churn_model_random_forest.pkl",
    )

    # 14. Display results
    print(f"Model accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    print("Model saved to models/churn_model_random_forest.pkl")
    print("Experiment recorded in MLflow")
