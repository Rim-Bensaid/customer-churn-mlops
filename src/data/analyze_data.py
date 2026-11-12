from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Chemins
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

def load_data():
    """Charge les données."""

    df = pd.read_csv(DATA_PATH)

    # Conversion numérique pour l'analyse uniquement
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df

def analyze_total_charges(df):
    """Analyse les valeurs manquantes de TotalCharges."""

    print("\n ANALYSE TOTALCHARGES ")

    missing = df[df["TotalCharges"].isna()]

    print("Nombre de valeurs manquantes :", len(missing))

    print("\nObservations concernées :")

    print(
        missing[
            ["customerID", "tenure", "MonthlyCharges",
             "TotalCharges", "Churn"]
        ].to_string(index=False)
    )

    print("\nAncienneté des clients concernés :")

    print(missing["tenure"].value_counts())

    print("\nRépartition Churn des clients concernés :")

    print(missing["Churn"].value_counts())

def analyze_numerical_variables(df):
    """Analyse les variables numériques."""

    print("\n VARIABLES NUMERIQUES ")

    numerical_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    print(df[numerical_columns].describe().T)


def analyze_categorical_variables(df):
    """Analyse les variables catégorielles."""

    print("\n VARIABLES CATEGORIELLES ")

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:

        if column in ["customerID", "Churn"]:
            continue

        print(f"\nVariable : {column}")

        print(df[column].value_counts(dropna=False))

def analyze_churn_by_contract(df):
    """Analyse le taux de churn selon le contrat."""

    print("\n CHURN PAR CONTRAT ")

    churn_rate = (
        df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print(churn_rate)


def create_visualizations(df):
    """Génère les graphiques exploratoires."""

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Distribution de la cible
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Churn"
    )

    plt.title("Distribution de la variable Churn")
    plt.xlabel("Churn")
    plt.ylabel("Nombre de clients")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "churn_distribution.png"
    )

    plt.close()

    # 2. Ancienneté selon le churn
    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="tenure"
    )

    plt.title("Ancienneté selon le statut Churn")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "tenure_by_churn.png"
    )

    plt.close()

    # 3. Taux de churn selon le contrat
    churn_contract = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=churn_contract,
        x="Contract",
        y="ChurnRate"
    )

    plt.title("Taux de churn selon le type de contrat")
    plt.ylabel("Taux de churn (%)")
    plt.xlabel("Type de contrat")

    plt.tight_layout()

    plt.savefig(
        REPORTS_PATH / "churn_by_contract.png"
    )

    plt.close()

    print("\nGraphiques enregistrés dans reports/.")

def main():

    print("ANALYSE EXPLORATOIRE APPROFONDIE")

    df = load_data()

    analyze_total_charges(df)

    analyze_numerical_variables(df)

    analyze_categorical_variables(df)

    analyze_churn_by_contract(df)

    create_visualizations(df)

if __name__ == "__main__":
    main()