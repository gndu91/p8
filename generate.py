import pandas as pd
import numpy as np
import joblib
import shap
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from dataset import get_dataset

ARTIFACTS_DIR = Path('artifacts')
OUTPUT_PATH = Path('feature_importance.json')
SAMPLE_PATH = Path('application_sample.csv')
SAMPLE_SIZE = 1000

def generate_global_importance():
    """
    Calcule l'importance globale des features sur un échantillon de données
    et sauvegarde le résultat dans un fichier JSON.
    """
    print("Chargement du modèle depuis les artefacts...")
    try:
        model_pipeline = joblib.load(ARTIFACTS_DIR / "model.pkl")
    except FileNotFoundError:
        print(f"ERREUR: Le fichier modèle 'artifacts/model.pkl' est introuvable.")
        return

    print("Préparation du pipeline et de l'explainer SHAP...")
    preprocessing_pipeline = Pipeline(model_pipeline.steps[:-1])
    model = model_pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)

    print(f"Chargement et échantillonnage des données ({SAMPLE_SIZE} clients)...")
    try:
        data_sample = get_dataset('application_train').sample(SAMPLE_SIZE, random_state=42)
        data_sample.to_csv(SAMPLE_PATH, index=False)
        data_sample = data_sample.drop(columns=['TARGET'], errors='ignore')
    except Exception as e:
        print(f"ERREUR: Impossible de charger les données. Détail : {e}")
        return

    print("Prétraitement des données...")
    preprocessed_sample = preprocessing_pipeline.transform(data_sample)

    print("Calcul des valeurs SHAP (cela peut prendre quelques instants)...")
    shap_values = explainer.shap_values(preprocessed_sample)

    # Pour la classification binaire, shap_values est une liste de 2 arrays.
    # On prend celui de la classe positive (classe 1 : défaut).
    shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

    print("Calcul de l'importance moyenne...")
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    feature_names = preprocessed_sample.columns

    global_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    print(f"Sauvegarde des 20 features les plus importantes dans '{OUTPUT_PATH}'...")
    output_data = global_importance_df.head(20).to_dict('records')

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Opération terminée avec succès.")

if __name__ == "__main__":
    generate_global_importance()