import time
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.monitoring.monitor import log_api_request, log_prediction

# Configuration MLflow
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/customer-churn-model@champion"

mlflow.set_tracking_uri(TRACKING_URI)

# Modèle chargé au démarrage de l'API
model = None

# Structure des données reçues
class CustomerData(BaseModel):

    # Variables numériques
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    # Variables catégorielles
    gender: Literal["Female", "Male"]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]



# Chargement du modèle au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Chargement du modèle depuis MLflow...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    print("Modèle chargé avec succès.")

    yield

    model = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API de prédiction du départ des clients.",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code

    except Exception:
        status_code = 500

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        log_api_request(
            request.method,
            request.url.path,
            status_code,
            latency_ms
        )

        raise

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_api_request(
        request.method,
        request.url.path,
        status_code,
        latency_ms
    )

    return response

# Vérification de l'API
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "model": MODEL_URI
    }


# Vérification de l'état du modèle
@app.get("/health")
def health():
    return {
        "api_status": "running",
        "model_status": "loaded" if model is not None else "not loaded"
    }


# Prédiction du churn

@app.post("/predict")
def predict_churn(customer: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé."
        )

    try:
        customer_df = pd.DataFrame([customer.model_dump()])

        prediction = int(model.predict(customer_df)[0])

        probabilities = model.predict_proba(customer_df)[0]
        class_index = list(model.classes_).index(1)
        churn_probability = float(probabilities[class_index])

        churn_label = "Yes" if prediction == 1 else "No"

        message = (
            "Le client présente un risque de départ."
            if prediction == 1
            else "Le client ne présente pas de risque élevé de départ selon le modèle."
        )

        # Enregistrement du résultat de la prédiction
        log_prediction(
            prediction=churn_label,
            probability=round(churn_probability, 4),
            status="success"
        )

        return {
            "prediction": churn_label,
            "churn_probability": round(churn_probability, 4),
            "message": message
        }

    except Exception as e:  # noqa: BLE001

        # Enregistrement de l'erreur
        log_prediction(
            status="error",
            error=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {e!s}"
        )
