from random import randrange

import pytest, json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Make sure to run pytest from the root directory where `api.py` and `dataset.py` are located.
from api import app, BUSINESS_THRESHOLD
from dataset import get_dataset


# Instantiate the test client for the FastAPI application
client = TestClient(app)


def sanitize_for_json(data_dict: dict) -> dict:
    """
    Sanitizes a dictionary by converting numpy types to native Python types
    and replacing pandas/numpy NaN with None. This makes the dictionary
    JSON-serializable for the test client.
    """
    sanitized = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            # Convert NaN to None for JSON compatibility
            sanitized[key] = float(value) if not pd.isna(value) else None
        elif pd.isna(value):
            sanitized[key] = None
        else:
            sanitized[key] = value
    return sanitized


application_df = get_dataset("application_train")
client_row = application_df.iloc[randrange(len(application_df))].to_dict()
target = client_row.pop('TARGET')

payload = sanitize_for_json(client_row)
response = client.post("/predict", json=payload)
assert response.status_code == 200, f"API call failed for client SK_ID_CURR {payload.get('SK_ID_CURR')}"
print(json.dumps(response.json(), indent=2))