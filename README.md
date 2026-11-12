# Customer Churn Prediction — Projet MLOps

## 1. Présentation du projet

Ce projet consiste à développer et industrialiser un système de Machine Learning capable de prédire le désabonnement des clients d'un opérateur téléphonique (*Customer Churn*).

L'objectif est d'identifier les clients susceptibles de résilier leur abonnement afin d'aider l'entreprise à mettre en place des actions de fidélisation.

Le projet couvre les principales étapes d'un pipeline MLOps classique :

- Exploration et préparation des données.
- Entraînement et évaluation de plusieurs modèles.
- Suivi des expériences avec MLflow.
- Versionnement des données avec DVC.
- Enregistrement et promotion des modèles.
- Déploiement d'une API avec FastAPI.
- Monitoring des requêtes et des prédictions.
- Tests automatisés et intégration continue avec GitHub Actions.

## 2. Dataset

Le projet utilise le dataset **Telco Customer Churn**, fourni par IBM.

- Nombre de lignes : 7 043.
- Nombre de colonnes initiales : 21.
- Variable cible : `Churn`.
- Classes : `Yes` (désabonnement) et `No` (absence de désabonnement).

Le fichier original est conservé dans `data/raw/`.

La préparation est réalisée avec `src/data/preprocess.py`.

Les 11 valeurs manquantes de `TotalCharges` sont remplacées par 0, car elles correspondent à des clients ayant une ancienneté (`tenure`) égale à 0 et une valeur `Churn` égale à `No`.

La colonne `customerID` est supprimée.

Les données sont ensuite divisées en deux ensembles :

- Entraînement : 5 634 lignes.
- Test : 1 409 lignes.

La séparation est stratifiée afin de conserver la proportion des classes.

Le prétraitement des variables est intégré dans un pipeline Scikit-learn et ajusté sur les données d'entraînement.

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement |
| Pandas et NumPy | Manipulation des données |
| Scikit-learn | Prétraitement et entraînement |
| MLflow | Suivi des expériences et registre des modèles |
| DVC | Versionnement des données et pipeline |
| FastAPI | API de prédiction |
| Pytest | Tests automatisés |
| Ruff | Contrôle de la qualité du code |
| Git et GitHub | Versionnement du code |
| GitHub Actions | Intégration continue |

## 4. Architecture du projet

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
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
├── api/
│   └── main.py
│
├── tests/
│   ├── test_api.py
│   └── test_monitoring.py
│
├── models/
├── reports/
├── configs/
├── dashboard/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

### Cloner le dépôt

```powershell
git clone https://github.com/Rim-Bensaid/customer-churn-mlops.git
cd customer-churn-mlops
```

### Créer un environnement virtuel

```powershell
python -m venv .venv
```

### Activer l'environnement

```powershell
.\.venv\Scripts\Activate.ps1
```

### Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

**Important concernant DVC :**

Le fichier de données original doit être disponible localement dans `data/raw/` pour exécuter le pipeline. Le projet utilise actuellement un stockage DVC local.

## 6. Exécution du pipeline

Toutes les commandes suivantes doivent être exécutées depuis la racine du projet.

### Préparation des données

```powershell
python src/data/preprocess.py
```

### Entraînement des modèles

```powershell
python src/models/train.py
```

Trois modèles sont entraînés :

- Logistic Regression.
- Random Forest.
- Gradient Boosting.

### Évaluation des modèles

```powershell
python src/models/evaluate.py
```

Les métriques et les matrices de confusion sont enregistrées dans le dossier `reports/`.

### Reproduire le pipeline avec DVC

```powershell
dvc repro
```

Pour vérifier l'état du pipeline :

```powershell
dvc status
```

### Enregistrement du modèle

```powershell
python src/models/register_model.py
```

Le modèle sélectionné est enregistré dans MLflow sous le nom :

`customer-churn-model`

L'enregistrement crée une version candidate associée à l'alias `candidate`.

### Promotion du modèle

```powershell
python src/models/promote_model.py
```

Le script vérifie les seuils de qualité définis :

- Recall supérieur ou égal à 0,75.
- F1-score supérieur ou égal à 0,60.
- ROC-AUC supérieur ou égal à 0,80.

Lorsque les conditions sont satisfaites, le modèle est promu avec l'alias `champion`.

## 7. Résultats de l'évaluation

Trois algorithmes ont été comparés sur l'ensemble de test :

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73,81 % | 50,43 % | 78,34 % | 61,36 % | 84,16 % |
| Random Forest | 77,00 % | 55,90 % | 63,37 % | 59,40 % | 82,25 % |
| Gradient Boosting | 80,27 % | 66,55 % | 51,60 % | 58,13 % | 84,33 % |

La régression logistique a été retenue comme modèle final grâce à son Recall de 78,34 % et à son F1-score de 61,36 %.

Le Recall est particulièrement important dans ce projet, car il mesure la proportion de clients qui résilient effectivement leur abonnement et que le modèle parvient à identifier.

La précision obtenue est de 50,43 %.

## 8. Suivi des expériences avec MLflow

MLflow est utilisé pour :

- Suivre les expériences d'entraînement.
- Enregistrer les paramètres et les métriques.
- Conserver les informations relatives aux modèles.
- Gérer les versions dans le registre des modèles.
- Identifier les modèles candidats et le modèle champion.

L'expérience utilisée est :

`customer-churn-prediction`

Le modèle enregistré dans le registre est :

`customer-churn-model`

Pour démarrer l'interface MLflow :

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir :

http://127.0.0.1:5000

## 9. API de prédiction avec FastAPI

Une API REST a été développée avec FastAPI afin d'exposer le modèle sélectionné.

L'API charge le modèle champion depuis MLflow à l'aide de l'URI :

`models:/customer-churn-model@champion`

### Démarrer l'API

```powershell
uvicorn api.main:app --reload
```

Documentation interactive :

http://127.0.0.1:8000/docs

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Présentation de l'API |
| GET | `/health` | Vérification de l'état de l'API et du modèle |
| POST | `/predict` | Prédiction du désabonnement |

L'endpoint `/predict` reçoit les caractéristiques d'un client et retourne une prédiction de désabonnement ainsi que la probabilité associée.

## 10. Monitoring

Un module de monitoring enregistre les informations relatives aux requêtes et aux prédictions.

Les informations suivies comprennent :

- Méthode HTTP et endpoint.
- Code de réponse HTTP.
- Temps de réponse.
- Classe prédite (`Yes` ou `No`).
- Probabilité de prédiction.
- Statut de l'opération et erreurs éventuelles.

Les fichiers de suivi sont générés dans :

```text
reports/monitoring/
├── api_requests.csv
└── predictions.csv
```

Le monitoring permet de suivre le fonctionnement de l'API et de conserver un historique des requêtes et des prédictions.

## 11. Tests et intégration continue

Les tests automatisés sont réalisés avec Pytest.

Pour exécuter tous les tests :

```powershell
python -m pytest tests -v
```

Les tests couvrent notamment :

- Le fonctionnement des endpoints.
- La vérification de l'état de l'API.
- Les prédictions valides.
- Le rejet des données invalides.
- L'enregistrement des requêtes.
- L'enregistrement des prédictions et des erreurs.

La qualité du code est vérifiée avec Ruff :

```powershell
ruff check src api tests
```

GitHub Actions exécute automatiquement les vérifications de qualité du code et les tests selon les événements configurés dans le workflow CI.

## 12. Conclusion

Ce projet met en œuvre un pipeline MLOps classique allant de la préparation des données jusqu'à l'exposition d'un modèle de prédiction via une API.

Il combine plusieurs outils complémentaires pour assurer le suivi des expériences, le versionnement des données, la gestion des versions du modèle, les tests automatisés et le monitoring technique.

La régression logistique a été retenue et enregistrée dans MLflow comme modèle champion, puis intégrée à l'API FastAPI pour réaliser des prédictions.