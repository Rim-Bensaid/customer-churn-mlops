import pytest
from fastapi.testclient import TestClient

import api.main as api_module


# Faux modèle utilisé uniquement pendant les tests
class FakeModel:

    classes_ = [0, 1]

    def predict(self, data):
        return [0] * len(data)

    def predict_proba(self, data):
        return [[0.8, 0.2] for _ in range(len(data))]


# Exemple de données client valides
CUSTOMER_DATA = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 850.0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

@pytest.fixture
def client(monkeypatch):

    # Remplacer le chargement MLflow par le faux modèle
    monkeypatch.setattr(
        api_module.mlflow.sklearn,
        "load_model",
        lambda model_uri: FakeModel()
    )

    with TestClient(api_module.app) as test_client:
        yield test_client

def test_home(client):

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["api_status"] == "running"
    assert response.json()["model_status"] == "loaded"

def test_predict_valid_customer(client):

    response = client.post("/predict", json=CUSTOMER_DATA)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert "message" in result

def test_predict_invalid_customer(client):

    invalid_data = CUSTOMER_DATA.copy()
    invalid_data["tenure"] = -5

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422

def test_predict_invalid_category(client):

    invalid_data = CUSTOMER_DATA.copy()
    invalid_data["Contract"] = "Annual"

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
