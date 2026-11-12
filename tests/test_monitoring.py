import csv

from src.monitoring import monitor


def test_log_api_request(tmp_path, monkeypatch):
    # Rediriger les fichiers vers un dossier temporaire
    requests_file = tmp_path / "api_requests.csv"

    monkeypatch.setattr(
        monitor,
        "MONITORING_DIR",
        tmp_path
    )

    monkeypatch.setattr(
        monitor,
        "REQUESTS_FILE",
        requests_file
    )

    monitor.log_api_request(
        method="GET",
        endpoint="/health",
        status_code=200,
        latency_ms=15.5
    )

    assert requests_file.exists()

    with open(requests_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["method"] == "GET"
    assert rows[0]["endpoint"] == "/health"
    assert rows[0]["status_code"] == "200"

def test_log_prediction(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"

    monkeypatch.setattr(
        monitor,
        "MONITORING_DIR",
        tmp_path
    )

    monkeypatch.setattr(
        monitor,
        "PREDICTIONS_FILE",
        predictions_file
    )

    monitor.log_prediction(
        prediction="Yes",
        probability=0.85,
        status="success"
    )

    assert predictions_file.exists()

    with open(predictions_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["prediction"] == "Yes"
    assert rows[0]["churn_probability"] == "0.85"
    assert rows[0]["status"] == "success"

def test_log_prediction_error(tmp_path, monkeypatch):
    predictions_file = tmp_path / "predictions.csv"

    monkeypatch.setattr(
        monitor,
        "MONITORING_DIR",
        tmp_path
    )

    monkeypatch.setattr(
        monitor,
        "PREDICTIONS_FILE",
        predictions_file
    )

    monitor.log_prediction(
        status="error",
        error="Test error"
    )

    with open(predictions_file, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "Test error"
