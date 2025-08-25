import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Make sure to run pytest from the root directory where `api.py` and `dataset.py` are located.
from api import app, BUSINESS_THRESHOLD
from dataset import get_dataset


# Instantiate the test client for the FastAPI application
client = TestClient(app)

# Load the dataset for testing. If it fails, skip all tests in this module.
try:
    # We use a small sample to speed up tests
    application_df = get_dataset("application_train").sample(n=10, random_state=42)
    # The 'TARGET' column is not a feature, so we drop it.
    features_df = application_df.drop(columns=['TARGET'], errors='ignore')
except Exception as e:
    pytest.skip(f"Could not load dataset, skipping API tests. Error: {e}", allow_module_level=True)


# --- Helper Function ---

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


# --- Test Cases ---

def test_status_endpoint():
    """
    Tests the /status endpoint to ensure the API is running and the model is loaded.
    """
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "up"}


def test_root_endpoint():
    """
    Tests the root endpoint (/) to ensure it also serves as a health check.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "up"}


def test_predict_success_valid_client():
    """
    Tests a successful prediction with a valid client record from the dataset.
    Checks the response structure, types, and decision logic.
    """
    # Select the first sample client from our test dataframe
    sample_row = features_df.iloc[0].to_dict()
    payload = sanitize_for_json(sample_row)

    # Send the request to the /predict endpoint
    response = client.post("/predict", json=payload)

    # Assert a successful response
    assert response.status_code == 200

    # Parse the JSON response and validate its structure
    data = response.json()
    expected_keys = ["client_id", "probability_default", "decision", "threshold"]
    assert all(key in data for key in expected_keys)

    # Validate the content and types
    assert data["client_id"] == payload["SK_ID_CURR"]
    assert isinstance(data["probability_default"], float)
    assert 0.0 <= data["probability_default"] <= 1.0
    assert data["decision"] in ["yes", "no"]
    assert data["threshold"] == BUSINESS_THRESHOLD

    # Validate the decision logic based on the returned probability
    expected_decision = "yes" if data["probability_default"] < BUSINESS_THRESHOLD else "no"
    assert data["decision"] == expected_decision


def test_predict_with_missing_fields():
    """
    Tests the API's robustness to missing data.
    The Pydantic model is "forgiving" and should accept requests even if
    fields (including those marked 'required' in MLmodel) are missing.
    The backend pipeline is expected to handle the resulting NaNs.
    """
    sample_row = features_df.iloc[1].to_dict()

    # Remove some fields to test robustness
    # 'AMT_CREDIT' is "required" in MLmodel, 'OWN_CAR_AGE' is optional.
    del sample_row['AMT_CREDIT']
    if 'OWN_CAR_AGE' in sample_row:
        del sample_row['OWN_CAR_AGE']

    payload = sanitize_for_json(sample_row)

    response = client.post("/predict", json=payload)

    # The request should still succeed (200 OK)
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == payload["SK_ID_CURR"]
    assert "probability_default" in data


def test_predict_missing_client_id():
    """
    Tests the specific case where SK_ID_CURR is missing from the payload.
    The response should contain `client_id: null`.
    """
    sample_row = features_df.iloc[2].to_dict()
    # Remove the client ID from the payload
    del sample_row['SK_ID_CURR']

    payload = sanitize_for_json(sample_row)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    # The client_id in the response should be None
    assert data["client_id"] is None
    assert "probability_default" in data



def test_predict_invalid_data_type():
    """
    Tests the API's response to a payload with an incorrect data type.
    Pydantic validation should fail, resulting in a 422 Unprocessable Entity error.
    """
    sample_row = features_df.iloc[3].to_dict()
    payload = sanitize_for_json(sample_row)

    # Intentionally provide an invalid type for a numeric field
    payload['AMT_INCOME_TOTAL'] = 'not-a-valid-number'

    response = client.post("/predict", json=payload)

    # Assert that the request is rejected as unprocessable
    assert response.status_code == 422

    # Check that the error message is helpful
    data = response.json()
    assert "detail" in data
    # FastAPI provides a list of validation errors
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0

    error_location = data["detail"][0]["loc"]
    assert error_location[:2] == ["body", "AMT_INCOME_TOTAL"]

    # **FIX:** Make the assertion more general to match different Pydantic error messages.
    # We just check for the key phrase "Input should be a valid".
    assert "Input should be a valid" in data["detail"][0]["msg"]



def test_model_performance_on_test_set():
    """
    Tests the model's performance on a larger sample of the test set.
    It checks if the model's accuracy is above a minimum threshold (50%),
    acting as a basic check to ensure it's better than random guessing.
    It also prints the full results for manual inspection.
    """
    N_SAMPLES = 1024
    MIN_ACCURACY = 0.50

    test_df = get_dataset("application_train").sample(N_SAMPLES)

    # 2. Pop the "TARGET" to make sure there is no leak.
    real_targets = test_df.pop("TARGET")
    features_only_df = test_df

    results = []
    correct_predictions = 0

    # 3. Iterate over the sample, get predictions, and compare
    for index, row in features_only_df.iterrows():
        payload = sanitize_for_json(row.to_dict())
        response = client.post("/predict", json=payload)

        assert response.status_code == 200, f"API call failed for client SK_ID_CURR {payload.get('SK_ID_CURR')}"

        data = response.json()

        real_target = real_targets.loc[index]
        decision = data["decision"]
        probability = data["probability_default"]

        # A decision is correct if:
        # - We grant a loan ('yes') to a non-defaulter (TARGET=0)
        # - We deny a loan ('no') to a defaulter (TARGET=1)
        is_correct = (decision == 'yes' and real_target == 0) or \
                     (decision == 'no' and real_target == 1)

        if is_correct:
            correct_predictions += 1

        results.append({
            "SK_ID_CURR": payload.get("SK_ID_CURR"),
            "Real Value (TARGET)": real_target,
            "Returned Decision": decision,
            "Returned Probability": probability,
            "Correct?": "Yes" if is_correct else "No"
        })

    # 4. Show the decisions for all items as requested
    results_df = pd.DataFrame(results)
    print("\n--- Model Performance Test Results (on 1024 samples) ---")
    # Use to_string() to ensure all rows are printed to the console
    print(results_df.to_string())
    print("----------------------------------------------------------\n")

    # 5. Make sure at least 50% of the returned decisions are correct
    accuracy = correct_predictions / len(features_only_df)

    print(f"Total items tested: {len(features_only_df)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Calculated Accuracy: {accuracy:.2%}")
    print(f"Minimum Required Accuracy: {MIN_ACCURACY:.2%}")

    assert accuracy >= MIN_ACCURACY, f"Model accuracy ({accuracy:.2%}) is below the required threshold of {MIN_ACCURACY:.2%}"