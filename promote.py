import argparse

from mlflow import MlflowClient

from src.data import load_params

# Configuration des arguments de la ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument(
    "--approve",
    action="store_true",
    help="Simule l'approbation humaine avant de promouvoir en champion",
)
args = parser.parse_args()

# Charger les paramètres depuis params.yaml
params = load_params()
model_name = params["training"]["model_name"]

# Initialiser le client MLflow pointant vers le serveur local
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Récupérer la version du modèle ayant l'alias "candidate"
candidate = client.get_model_version_by_alias(model_name, "candidate")
status = candidate.tags.get("validation_status")

# Vérifier le Quality Gate / statut de validation enregistré dans les tags
if status != "passed":
    raise RuntimeError(
        "La version candidate n'a pas passe les Quality Gates."
    )

print(f"Candidate version : {candidate.version}")
print(f"validation_status : {status}")

# Si l'argument --approve n'est pas fourni, le script s'arrête
if not args.approve:
    print("Aucune approbation fournie : le modele reste au statut candidate.")
    raise SystemExit(0)

# Promouvoir le modèle candidate au statut de champion
client.set_registered_model_alias(
    model_name, "champion", candidate.version
)
client.set_model_version_tag(
    model_name, candidate.version, "approval_status", "approved"
)

print(f"Promotion reussie : l'alias 'champion' pointe desormais vers la version {candidate.version}")