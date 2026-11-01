import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.data import load_params
from src.train import train

# Charger les paramètres depuis le fichier params.yaml
params = load_params()

# Configurer l'URI de suivi MLflow et le nom de l'expérience
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(params["training"]["experiment_name"])

model_name = params["training"]["model_name"]

# Démarrer une session d'exécution (run) MLflow
with mlflow.start_run(run_name="sentiment_candidate") as run:
    # Entraîner le modèle et récupérer les métriques
    model, metrics = train()

    # Enregistrer les hyperparamètres dans MLflow
    mlflow.log_params({
        "C": params["model"]["C"],
        "max_features": params["model"]["max_features"],
        "ngram_range": str(params["model"]["ngram_range"]),
    })

    # Enregistrer les métriques de performance
    mlflow.log_metrics(metrics)

    # Sauvegarder le modèle et l'enregistrer dans le Model Registry
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
    )

    client = MlflowClient()

    # Récupérer la dernière version enregistrée du modèle
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))

    # Ajouter des tags de traçabilité et de validation
    client.set_model_version_tag(
        model_name, latest.version, "validation_status", "passed"
    )
    client.set_model_version_tag(
        model_name, latest.version, "use_case", "sentiment"
    )

    client.set_registered_model_tag(
        model_name, "task", "text-classification"
    )

    # Assigner l'alias "candidate" à cette version de modèle
    client.set_registered_model_alias(
        model_name, "candidate", latest.version
    )

    print(f"Version enregistree : {latest.version}")
    print(f"Alias candidate -> version {latest.version}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")