
from pathlib import Path

import pandas as pd


# Chemin racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chemin du dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_data():
    """Charge le dataset brut."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def explore_data(df):
    """Effectue une première exploration du dataset."""

    print("1. DIMENSIONS DU DATASET")

    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    print("2. APERÇU DES DONNÉES")

    print(df.head())

    print("3. INFORMATIONS SUR LES VARIABLES")

    df.info()

    print("4. VALEURS MANQUANTES")

    missing = df.isnull().sum()

    print(missing[missing > 0])

    print("5. DOUBLONS")

    print("Nombre de lignes dupliquées :", df.duplicated().sum())

    print(
        "Nombre d'identifiants clients dupliqués :",
        df["customerID"].duplicated().sum()
    )

    print("6. DISTRIBUTION DE LA VARIABLE CIBLE")

    print(df["Churn"].value_counts())

    print("\nPourcentages :")

    print(
        df["Churn"].value_counts(normalize=True) * 100
    )


    print("7. STATISTIQUES DESCRIPTIVES")

    print(df.describe(include="all").T)

    print("8. VERIFICATION DE TOTALCHARGES")
  
    total_charges = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print(
        "Valeurs non convertibles en numérique :",
        total_charges.isna().sum()
    )

    print(
        "Clients concernés avec TotalCharges vide :",
        df["TotalCharges"].astype(str).str.strip().eq("").sum()
    )


def main():

    print("CUSTOMER CHURN - EXPLORATION DES DONNEES")

    df = load_data()

    explore_data(df)


if __name__ == "__main__":
    main()
