import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REPORT_PATH = REPORTS_DIR / "preprocessing_report.json"

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]


def load_data():
    """Charge et vérifie le dataset brut."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {missing_columns}"
        )

    if df.empty:
        raise ValueError("Le dataset est vide.")

    print("Dataset chargé avec succès.")
    print("Dimensions initiales :", df.shape)

    return df


def clean_data(df):
    """Nettoie les données avant leur séparation."""

    df = df.copy()

    # Vérification des identifiants
    if df[ID_COLUMN].isna().any():
        raise ValueError("Des identifiants clients sont manquants.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Des identifiants clients sont dupliqués.")

    # Conversion de TotalCharges en numérique
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    missing_total_charges = int(df["TotalCharges"].isna().sum())

    # Vérification du contexte des valeurs manquantes
    missing_rows = df["TotalCharges"].isna()

    if missing_rows.any():
        valid_missing_rows = (
            (df.loc[missing_rows, "tenure"] == 0)
            & (df.loc[missing_rows, "Churn"] == "No")
        )

        if not valid_missing_rows.all():
            raise ValueError(
                "Certaines valeurs manquantes de TotalCharges "
                "ne correspondent pas au cas identifié."
            )

        df.loc[missing_rows, "TotalCharges"] = 0

    # Vérification de la variable cible
    valid_target_values = {"Yes", "No"}

    if not set(df[TARGET_COLUMN].dropna().unique()).issubset(
        valid_target_values
    ):
        raise ValueError("Valeurs inattendues dans la variable Churn.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("La variable cible contient des valeurs manquantes.")

    # Encodage de la cible
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # Suppression de l'identifiant
    df = df.drop(columns=[ID_COLUMN])

    # Vérification finale
    if df.isna().any().any():
        missing_columns = df.columns[df.isna().any()].tolist()

        raise ValueError(
            f"Valeurs manquantes restantes : {missing_columns}"
        )

    print("Nettoyage terminé.")
    print("Valeurs TotalCharges remplacées :", missing_total_charges)
    print("Dimensions après nettoyage :", df.shape)

    return df, missing_total_charges

def split_data(df):
    """Sépare les données en entraînement et test."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Reconstruction des jeux de données
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    # Conservation du même ordre de colonnes
    train_df = train_df[df.columns]
    test_df = test_df[df.columns]

    print("\nSéparation terminée.")
    print("Taille entraînement :", len(train_df))
    print("Taille test :", len(test_df))

    return train_df, test_df

def save_data(train_df, test_df, missing_count):
    """Sauvegarde les données préparées et le rapport."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    report = {
        "dataset": "IBM Telco Customer Churn",
        "original_rows": len(train_df) + len(test_df),
        "original_features": len(train_df.columns),
        "removed_identifier": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "missing_total_charges_replaced": missing_count,
        "replacement_value": 0,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_churn_distribution": {
            str(k): int(v)
            for k, v in train_df[TARGET_COLUMN].value_counts().items()
        },
        "test_churn_distribution": {
            str(k): int(v)
            for k, v in test_df[TARGET_COLUMN].value_counts().items()
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nFichiers sauvegardés :")
    print("Entraînement :", TRAIN_PATH)
    print("Test :", TEST_PATH)
    print("Rapport :", REPORT_PATH)

def main():
    print("PRÉPARATION DES DONNÉES CUSTOMER CHURN")

    df = load_data()

    df, missing_count = clean_data(df)

    train_df, test_df = split_data(df)

    save_data(train_df, test_df, missing_count)

    print("\nPRÉTRAITEMENT TERMINÉ AVEC SUCCÈS.")

if __name__ == "__main__":
    main()