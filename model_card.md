# Model Card : Sentiment Analysis Model

## 1. Modèle & Intention
* **Nom du modèle :** Sentiment Classifier
* **Type :** Logistic Regression (scikit-learn)
* **Objectif :** Classifier des avis textuels en sentiments positifs ou négatifs.
* **Auteur / Mainteneur :** Maroua El Mrabet

## 2. Données d'entraînement & Évaluation
* **Jeu de données :** `data/reviews.csv`
* **Prétraitement :** TfidfVectorizer (extraction de n-grammes de mots)
* **Split :** Entraînement / Test avec état aléatoire fixé (`random_state=42`)

## 3. Métriques & Quality Gates
* **Accuracy minimale requise :** 0.40 (définie dans `params.yaml`)
* **Résultats actuels :**
  * Accuracy : 0.400
  * F1 Score : 0.571
* **Statut de validation :** `passed`

## 4. Traçabilité & MLOps
* **Registre MLflow :** Modèle enregistré sous `sentiment-model`
* **Version actuelle Champion :** Version 1
* **Pipeline CI/CD :** Validation automatique via GitHub Actions (`pytest` + couverture)
* **Déploiement :** Promotion sous condition d'approbation humaine (Environnement `production`)

## 5. Limitations & Biais
* Modèle entraîné sur un échantillon réduit pour les besoins du TP.
* Performances dépendantes de la représentativité du vocabulaire présent dans `reviews.csv`.