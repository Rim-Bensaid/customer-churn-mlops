import csv
from datetime import UTC, datetime
from pathlib import Path

# Dossier racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossier contenant les rapports de monitoring
MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"

# Fichiers de suivi
REQUESTS_FILE = MONITORING_DIR / "api_requests.csv"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.csv"

def write_csv_row(file_path, fieldnames, row):
    """Ajoute une ligne dans un fichier CSV."""

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = file_path.exists() and file_path.stat().st_size > 0

    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def log_api_request(method, endpoint, status_code, latency_ms):
    """Enregistre les informations d'une requête API."""

    fields = [
        "timestamp",
        "method",
        "endpoint",
        "status_code",
        "latency_ms"
    ]

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2)
    }

    write_csv_row(REQUESTS_FILE, fields, row)

def log_prediction(prediction=None, probability=None,
                   status="success", error=None):
    """Enregistre le résultat d'une prédiction."""

    fields = [
        "timestamp",
        "prediction",
        "churn_probability",
        "status",
        "error"
    ]

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "prediction": prediction,
        "churn_probability": probability,
        "status": status,
        "error": error
    }

    write_csv_row(PREDICTIONS_FILE, fields, row)
