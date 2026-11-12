# Customer Churn Prediction — Projet MLOps

## 1. Présentation du projet

Ce projet consiste à développer et industrialiser un système de Machine Learning capable de prédire le désabonnement des clients d'un opérateur téléphonique (*Customer Churn*).

L'objectif est d'identifier les clients susceptibles de résilier leur abonnement afin d'aider l'entreprise à mettre en place des actions de fidélisation.

Le projet couvre les principales étapes d'un pipeline MLOps : préparation des données, entraînement, évaluation, suivi des expériences, gestion des modèles, déploiement d'une API et visualisation des résultats.

## 2. Dataset

Le projet utilise le dataset **IBM Telco Customer Churn**.

- 7 043 clients
- 21 colonnes initiales
- Variable cible : `Churn`
- 5 174 clients non désabonnés
- 1 869 clients désabonnés

Les données ont été nettoyées, puis divisées en deux ensembles :
- Entraînement : 5 634 observations
- Test : 1 409 observations

## 3. Technologies utilisées

- Python
- Pandas et NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Streamlit
- Pytest
- Ruff
- Git et GitHub Actions

## 4. Architecture du projet

```text
customer-churn-mlops/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── monitoring/
│
├── src/
│   ├── data/
│   │   ├── explore_data.py
│   │   ├── analyze_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── register_model.py
│   │   └── promote_model.py
│   │
│   └── monitoring/
│       └── monitor.py
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

## 5. Installation

Cloner le dépôt :

```bash
git clone https://github.com/Rim-Bensaid/customer-churn-mlops.git
cd customer-churn-mlops
```

Créer et activer l'environnement virtuel sous Windows :

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

Le fichier `requirements.txt` doit notamment inclure `streamlit` et `requests` pour l'interface graphique.

## 6. Pipeline MLOps

Le pipeline comprend trois étapes principales :

1. Prétraitement des données.
2. Entraînement des modèles.
3. Évaluation et comparaison des performances.

Les étapes sont orchestrées avec DVC :

```powershell
dvc repro
```

Le suivi des expériences et des métriques est réalisé avec MLflow.

## 7. Comparaison des modèles

Trois algorithmes ont été entraînés et évalués :

- Logistic Regression
- Random Forest
- Gradient Boosting

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.81 % | 50.43 % | 78.34 % | 61.36 % | 84.16 % |
| Random Forest | 77.00 % | 55.90 % | 63.37 % | 59.40 % | 82.25 % |
| Gradient Boosting | 80.27 % | 66.55 % | 51.60 % | 58.13 % | 84.33 % |

La **Logistic Regression** a été retenue comme modèle champion grâce à son Recall de 78.34 % et son F1-score de 61.36 %.

Ce choix privilégie l'identification des clients susceptibles de se désabonner.

## 8. Gestion des modèles avec MLflow

MLflow permet de :

- Suivre les expériences d'entraînement.
- Enregistrer les paramètres et les métriques.
- Comparer les modèles.
- Enregistrer les modèles dans le Model Registry.
- Gérer les versions et les alias.

Le modèle retenu est enregistré sous le nom :

`customer-churn-model`

Son alias de production est :

`champion`

## 9. API FastAPI

Une API REST développée avec FastAPI permet d'utiliser le modèle champion pour effectuer des prédictions.

Les endpoints disponibles sont :

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Page d'accueil |
| `/health` | GET | Vérification de l'état de l'API et du modèle |
| `/predict` | POST | Prédiction du désabonnement |

L'API charge le modèle enregistré dans MLflow avec l'alias `champion`.

Pour démarrer l'API :

```powershell
uvicorn api.main:app --reload
```

Documentation interactive :

`http://127.0.0.1:8000/docs`

## 10. Dashboard Streamlit

Une interface graphique développée avec Streamlit permet d'interagir avec le système de prédiction et de visualiser les résultats.

Elle comprend trois espaces :

- **Prédiction :** saisie des informations d'un client et affichage du résultat de prédiction ainsi que de la probabilité de désabonnement.
- **Visualisation des modèles :** comparaison des trois modèles selon les métriques Accuracy, Precision, Recall, F1-score et ROC-AUC.
- **Modèle champion :** présentation du modèle retenu, de sa version MLflow et de ses performances.

Le dashboard communique avec l'API FastAPI pour effectuer les prédictions.

Pour démarrer Streamlit, ouvrir un deuxième terminal et exécuter :

```powershell
streamlit run app/streamlit_app.py
```

Interface accessible à l'adresse :

`http://localhost:8501`

## 11. Monitoring

Un système de monitoring permet de suivre l'activité de l'API et les prédictions effectuées.

Les informations enregistrées comprennent notamment :

- Les requêtes reçues.
- Les codes de réponse HTTP.
- Le temps de réponse.
- Les résultats des prédictions.
- Les éventuelles erreurs.

Les fichiers de suivi sont enregistrés dans `reports/monitoring/`.

## 12. Tests et intégration continue

Les tests automatisés sont réalisés avec Pytest afin de vérifier le fonctionnement de l'API et du monitoring.

Ruff est utilisé pour contrôler la qualité du code.

GitHub Actions automatise l'exécution des contrôles :

1. Installation des dépendances.
2. Vérification du code avec Ruff.
3. Exécution des tests avec Pytest.

## 13. Conclusion

Ce projet met en œuvre un pipeline MLOps complet pour la prédiction du désabonnement des clients, depuis la préparation des données jusqu'à la mise à disposition d'une interface de prédiction.

L'association de DVC, MLflow, FastAPI et Streamlit permet de structurer le cycle de vie du modèle, de suivre ses performances et de rendre ses prédictions accessibles à travers une interface graphique.