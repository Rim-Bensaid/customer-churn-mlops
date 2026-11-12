# Customer Churn Prediction — Projet MLOps

## 1. Présentation du projet

Ce projet consiste à développer et industrialiser un système de Machine Learning capable de prédire le désabonnement des clients d'un opérateur téléphonique (*Customer Churn*).

L'objectif est d'identifier les clients susceptibles de résilier leur abonnement afin d'aider l'entreprise à mettre en place des actions de fidélisation.

Le projet couvre les principales étapes d'un pipeline MLOps classique : préparation des données, entraînement, évaluation, suivi des expériences, versionnement, déploiement d'une API et monitoring.

## 2. Dataset

Le projet utilise le dataset **Telco Customer Churn** fourni par IBM.

- Nombre de lignes : 7 043
- Nombre de colonnes initiales : 21
- Variable cible : `Churn`
- Classes : `Yes` (désabonnement) et `No` (fidélisation)

Le fichier original est conservé dans `data/raw/`.

Les données sont préparées avec le script `src/data/preprocess.py`. Les 11 valeurs manquantes de `TotalCharges` sont remplacées par 0, car elles correspondent à des clients ayant une ancienneté (`tenure`) égale à 0 et une valeur `Churn` égale à `No`.

Les données sont ensuite divisées en deux ensembles :
- Entraînement : 5 634 lignes.
- Test : 1 409 lignes.

La séparation est stratifiée afin de conserver la proportion des classes.

## 3. Technologies utilisées

- **Python** : développement.
- **Pandas et NumPy** : manipulation des données.
- **Scikit-learn** : prétraitement et entraînement.
- **MLflow** : suivi des expériences et registre des modèles.
- **DVC** : versionnement des données.
- **FastAPI** : création de l'API de prédiction.
- **Pytest** : tests automatisés.
- **Ruff** : contrôle de la qualité du code.
- **Git et GitHub** : versionnement du code et hébergement du projet.
- **GitHub Actions** : intégration continue.

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
│   │   └── register_model.py
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
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

Cloner le dépôt :

```powershell
git clone https://github.com/Rim-Bensaid/customer-churn-mlops.git
cd customer-churn-mlops
```

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

L'activer :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

**Important :** le dépôt DVC ne possède pas de stockage distant configuré. Le fichier de données original doit donc être disponible localement dans `data/raw/` pour reproduire le pipeline. Le fichier `.dvc` permet de conserver ses informations de versionnement.

## 6. Exécution du pipeline

Depuis la racine du projet :

### Préparation des données

```powershell
python src/data/preprocess.py
```

### Entraînement des modèles

```powershell
python src/models/train.py
```

### Évaluation des modèles

```powershell
python src/models/evaluate.py
```

### Enregistrement du modèle

```powershell
python src/models/register_model.py
```

Le modèle retenu est enregistré dans MLflow sous le nom `customer-churn-model`, avec l'alias `candidate`.

## 7. Résultats de l'évaluation

Trois algorithmes ont été comparés sur l'ensemble de test :

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73,81 % | 50,43 % | 78,34 % | 61,36 % | 84,16 % |
| Random Forest | 77,00 % | 55,90 % | 63,37 % | 59,40 % | 82,25 % |
| Gradient Boosting | 80,27 % | 66,55 % | 51,60 % | 58,13 % | 84,33 % |

La régression logistique a été retenue comme modèle candidat, principalement grâce à son Recall de 78,34 %.

Cette métrique est importante car l'objectif est de détecter les clients susceptibles de se désabonner. Le choix implique toutefois un compromis : la précision est de 50,43 %, ce qui signifie qu'une partie des clients signalés ne se désabonneront pas nécessairement.

La comparaison a été réalisée sur l'ensemble de test ; cette réutilisation pour le choix du modèle constitue une limite méthodologique à prendre en compte.

## 8. Suivi des expériences avec MLflow

MLflow est utilisé pour :
- suivre les expériences d'entraînement ;
- enregistrer les paramètres et les métriques ;
- conserver les informations sur les modèles ;
- gérer le registre des modèles.

Pour ouvrir l'interface MLflow :

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir :

http://127.0.0.1:5000

## 9. API de prédiction

Une API REST a été développée avec FastAPI.

Pour démarrer l'API :

```powershell
uvicorn api.main:app --reload
```

Documentation interactive :

http://127.0.0.1:8000/docs

Endpoints disponibles :

- `GET /` : présentation de l'API.
- `GET /health` : vérification de son état.
- `POST /predict` : prédiction du risque de désabonnement.

L'API charge le modèle enregistré dans MLflow via l'alias `candidate`.

## 10. Monitoring

Un module de monitoring enregistre les informations relatives aux requêtes et aux prédictions :

- méthode HTTP et endpoint ;
- code de réponse ;
- temps de réponse ;
- classe prédite ;
- probabilité de désabonnement ;
- statut et erreurs éventuelles.

Les fichiers de suivi sont générés dans :

```text
reports/monitoring/
├── api_requests.csv
└── predictions.csv
```

Ce monitoring couvre principalement le fonctionnement technique de l'API. La détection de dérive des données et l'évaluation de la qualité réelle des prédictions après déploiement ne sont pas encore implémentées.

## 11. Tests et intégration continue

Les tests automatisés sont réalisés avec Pytest :

```powershell
python -m pytest tests -v
```

La qualité du code est vérifiée avec Ruff :

```powershell
ruff check src api tests
```

GitHub Actions exécute automatiquement ces vérifications lors des événements configurés dans le workflow CI.

## 12. Conclusion

Ce projet met en œuvre un pipeline MLOps classique allant de la préparation des données jusqu'à l'exposition d'un modèle de prédiction via une API.

Il utilise plusieurs outils complémentaires pour assurer le suivi des expériences, le versionnement, les tests automatisés et le monitoring technique.

Le projet constitue une première mise en œuvre d'un processus d'industrialisation d'un modèle de Machine Learning.