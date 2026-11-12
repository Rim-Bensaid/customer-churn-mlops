
import streamlit as st
import pandas as pd
import requests

# Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="⨂",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# Style
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f5f7fa;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] h1 {
        color: #1e293b;
        font-size: 1.35rem;
        font-weight: 700;
    }

    [data-testid="stSidebar"] p {
        color: #475569;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 10px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        border-color: #64748b;
        background-color: #eef2f7;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.title("⨂ Customer Churn Prediction")
st.caption("Plateforme de prédiction et de visualisation du désabonnement des clients.")

# Connexion API
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        st.success("API FastAPI connectée")
    else:
        st.warning("L'état de l'API est à vérifier.")
except requests.exceptions.RequestException:
    st.error("API indisponible. Lancez FastAPI pour effectuer des prédictions.")

# Navigation
st.sidebar.title("Navigation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Prédiction", "Visualisation des modèles", "Modèle champion"],
    label_visibility="collapsed"
)

# Prédiction
if page == "Prédiction":

    st.header("Prédiction du désabonnement")
    st.write("Saisissez les informations du client.")

    with st.form("customer_form"):

        st.subheader("Informations générales")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Genre", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partenaire", ["Yes", "No"])
            dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
            tenure = st.number_input(
                "Ancienneté (mois)",
                min_value=0,
                max_value=100,
                value=12
            )

        with col2:
            monthly = st.number_input(
                "Charges mensuelles",
                min_value=0.0,
                value=70.0
            )
            total = st.number_input(
                "Charges totales",
                min_value=0.0,
                value=840.0
            )
            contract = st.selectbox(
                "Type de contrat",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Facturation électronique",
                ["Yes", "No"]
            )
            payment = st.selectbox(
                "Mode de paiement",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.subheader("Services souscrits")

        col3, col4 = st.columns(2)

        with col3:
            phone = st.selectbox("Service téléphonique", ["Yes", "No"])
            multiple = st.selectbox(
                "Lignes multiples",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Service Internet",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Sécurité Internet",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Sauvegarde en ligne",
                ["Yes", "No", "No internet service"]
            )

        with col4:
            protection = st.selectbox(
                "Protection des appareils",
                ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Support technique",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        submitted = st.form_submit_button(
            "Prédire le désabonnement",
            use_container_width=True
        )

    if submitted:

        customer = {
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment
        }

        try:
            with st.spinner("Prédiction en cours..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer,
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                prediction = result.get(
                    "prediction",
                    result.get("churn", result.get("predicted_churn"))
                )

                probability = result.get(
                    "probability",
                    result.get("churn_probability")
                )

                st.subheader("Résultat de la prédiction")

                if prediction is not None:
                    if str(prediction).lower() == "yes":
                        st.error("Risque de désabonnement : OUI")
                    else:
                        st.success("Risque de désabonnement : NON")

                if probability is not None:
                    probability = float(probability)

                    if 0 <= probability <= 1:
                        st.metric(
                            "Probabilité de désabonnement",
                            f"{probability:.1%}"
                        )
                        st.progress(probability)

                with st.expander("Détails de la réponse API"):
                    st.json(result)

            else:
                st.error(f"Erreur API : {response.status_code}")
                st.json(response.json())

        except requests.exceptions.RequestException as error:
            st.error(f"Impossible de contacter l'API : {error}")

        except (TypeError, ValueError):
            st.error("Le format de la réponse de l'API est invalide.")

# Visualisation
elif page == "Visualisation des modèles":

    st.header("Comparaison des modèles")

    data = pd.DataFrame({
        "Modèle": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Accuracy": [0.7381, 0.7700, 0.8027],
        "Precision": [0.5043, 0.5590, 0.6655],
        "Recall": [0.7834, 0.6337, 0.5160],
        "F1-score": [0.6136, 0.5940, 0.5813],
        "ROC-AUC": [0.8416, 0.8225, 0.8433]
    })

    st.dataframe(
        data.set_index("Modèle").style.format("{:.2%}"),
        use_container_width=True
    )

    st.subheader("Comparaison des performances")

    metric = st.selectbox(
        "Choisir une métrique",
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    chart_data = data[["Modèle", metric]].set_index("Modèle")
    st.bar_chart(chart_data)

    st.info(
        "Logistic Regression a été retenu comme modèle champion "
        "grâce à son Recall de 78,34 % et son F1-score de 61,36 %."
    )

# Modèle champion
elif page == "Modèle champion":

    st.header("Modèle enregistré dans MLflow")

    col1, col2, col3 = st.columns(3)

    col1.metric("Modèle", "Logistic Regression")
    col2.metric("Version", "5")
    col3.metric("Alias MLflow", "champion")

    st.subheader("Métriques du modèle")

    metrics = {
        "Accuracy": 0.7381,
        "Precision": 0.5043,
        "Recall": 0.7834,
        "F1-score": 0.6136,
        "ROC-AUC": 0.8416
    }

    for name, value in metrics.items():
        st.write(f"**{name} : {value:.2%}**")
        st.progress(value)

    st.success("Le modèle champion est chargé par l'API FastAPI.")
