from fastapi import FastAPI, HTTPException
import json
import yaml
import joblib
from pydantic import BaseModel, create_model
from typing import Dict, Any, Union
import pandas as pd
from pathlib import Path

# Chemin vers le dossier contenant les artefacts du modèle
ARTIFACTS_DIR = Path('artifacts')

# TODO: Store this as an artifact
BUSINESS_THRESHOLD = 0.52

# Chargement de la signature des inputs
inputs_schema = json.loads(yaml.safe_load((ARTIFACTS_DIR / "MLmodel").open('r'))["signature"]["inputs"])

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

try:
    model_pipeline = joblib.load(ARTIFACTS_DIR / "model.pkl")
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install the requirements defined in artifacts/requirements.txt")

# Create the new, forgiving model
ClientDataModel: type[BaseModel] = create_forgiving_pydantic_model()

app = FastAPI(
    title="Credit Scoring API",
    description="Accept or decline a credit based on probabilities of non payment.",
    version="1.2.0" # Version bump to reflect robustness improvement
)

@app.get("/status", tags=["Health Check"])
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "up" if model_pipeline else "down"}

@app.post("/predict", tags=["Prediction"])
async def predict(client_data: ClientDataModel):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Unavailable model")
        
    # The .model_dump() will contain None for missing values. Pandas converts these
    # to NaN, which is exactly what the scikit-learn pipeline expects.
    client_df = pd.DataFrame([client_data.model_dump()])

    try:
        #  The pipeline handles all imputation and preprocessing from here.
        probability = model_pipeline.predict_proba(client_df)[0][1]
    except Exception as e:
        # This error is now more likely to be a real problem inside the model,
        # not a simple data type issue.
        raise HTTPException(status_code=400, detail=f"Error occurred during prediction: {e}")

    return {
        "client_id": client_data.SK_ID_CURR,
        "probability_default": round(float(probability), 4),
        "decision": "yes" if probability < BUSINESS_THRESHOLD else "no",
        "threshold": BUSINESS_THRESHOLD
    }
