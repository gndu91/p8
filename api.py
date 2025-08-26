import sys

from fastapi import FastAPI, HTTPException
import json
import yaml
import joblib
from pydantic import BaseModel, create_model
from typing import Dict, Any, Union, List
import pandas as pd
from pathlib import Path
from shap import TreeExplainer
from sklearn.pipeline import Pipeline

# Chemin vers le dossier contenant les artefacts du modèle
ARTIFACTS_DIR = Path('artifacts')

# TODO: Store this as an artifact
BUSINESS_THRESHOLD = 0.52

# Chargement de la signature des inputs
inputs_schema = json.loads(yaml.safe_load((ARTIFACTS_DIR / "MLmodel").open('r'))["signature"]["inputs"])
EXPECTED_FEATURES: List[str] = [col['name'] for col in inputs_schema]

def create_forgiving_pydantic_model(model_name: str = "ClientData") -> type[BaseModel]:
    fields: Dict[str, Any] = {}
    for col in inputs_schema:
        mlflow_type = col["type"]

        if mlflow_type == "string":
            python_type = str | None
        elif mlflow_type in ["long", "double"]:
            python_type = Union[int, float, None]
        else:
            python_type = Any

        fields[col["name"]] = (python_type, None)

    print(f"Modèle Pydantic '{model_name}' créé avec {len(fields)} champs.")
    return create_model(model_name, **fields)

model_pipeline = joblib.load(ARTIFACTS_DIR / "model.pkl")

# Added the explainer to get more informations from the model
model_step = model_pipeline.named_steps['model']
if type(model_step).__name__ == 'LGBMClassifier':
    explainer = TreeExplainer(model_step)
else:
    raise NotImplementedError(
        f"Unsupported model, please add the explainer associated with models of type {type(model_step)}")

# TODO: Should I deepcopy the steps? Should I regenerate the preprocessing pipeline every time?
# In order to have more verbosity, the first step is to fetch the preprocessing steps
preprocessing_pipeline = Pipeline(model_pipeline.steps[:-1])

ClientDataModel: type[BaseModel] = create_forgiving_pydantic_model()

app = FastAPI(
    title="Credit Scoring API",
    description="Accept or decline a credit based on probabilities of non payment.",
    version="1.2.1" # Version bump to reflect dtype fix and column order fix
)

@app.get("/status", tags=["Health Check"])
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "up" if model_pipeline else "down"}

@app.post("/predict", tags=["Prediction"])
async def predict(client_data: ClientDataModel):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Unavailable model")

    client_dict = client_data.model_dump()
    client_df = pd.DataFrame([client_dict])
    client_df = client_df.reindex(columns=EXPECTED_FEATURES)

    # **CORE FIX:** Ensure numeric columns have numeric dtypes.
    # The presence of `None` can cause pandas to infer the dtype as 'object'.
    # Scikit-learn imputers require numeric dtypes to function correctly.
    # We use the schema to identify and convert these columns.
    for col in inputs_schema:
        if col['type'] in ['long', 'double']:
            # pd.to_numeric will convert numbers, and also turn None into NaN
            client_df[col['name']] = pd.to_numeric(client_df[col['name']])

    shap_values = explainer.shap_values(post_df := preprocessing_pipeline.transform(client_df))

    try:
        probability = model_pipeline.predict_proba(client_df)[0][1]
    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        raise HTTPException(status_code=400, detail=f"Error occurred during prediction: {e}")

    # TODO: Should I expect this and throw an error if not found?
    client_id = client_dict.get("SK_ID_CURR")

    return {
        "client_id": client_id,
        "probability_default": round(float(probability), 4),
        "decision": "yes" if probability < BUSINESS_THRESHOLD else "no",
        "threshold": BUSINESS_THRESHOLD,
        # TODO: Only pick the top 10
        "feature_importance":dict(zip(post_df.columns, shap_values.flatten())),
    }