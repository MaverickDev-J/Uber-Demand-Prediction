# import mlflow
# import dagshub
# import json

# import dagshub
# dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)

# # set the mlflow tracking uri
# mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")


# def load_model_information(file_path):
#     with open(file_path) as f:
#         run_info = json.load(f)
        
#     return run_info

# # set model name
# model_path = load_model_information("run_information.json")["model_uri"]

# # load the latest model from model registry
# model = mlflow.sklearn.load_model(model_path)


# def test_load_model_from_registry():
#     assert model is not None, "Failed to load model from registry"
    


import json
import mlflow
import pytest
from pathlib import Path
import dagshub
import tempfile
import joblib
import os

# Initialize Dagshub
dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")


def load_model_information(file_path: str):
    with open(file_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model():
    # Load model information from run_information.json
    file_path = Path(__file__).resolve().parents[1] / "run_information.json"
    run_info = load_model_information(file_path)
    run_id = run_info["run_id"]
    artifact_path = run_info["artifact_path"]
    
    # Download the model artifact directly (non-model specific)
    client = mlflow.MlflowClient()
    
    # Create a temporary directory to download the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download all artifacts from the run (avoiding model-specific endpoints)
        artifacts_path = client.download_artifacts(
            run_id=run_id,
            path="",  # Download all artifacts
            dst_path=temp_dir
        )
        
        # Load the model using joblib directly (since it's a scikit-learn model)
        model_file = os.path.join(artifacts_path, artifact_path, "model.pkl")
        if not os.path.exists(model_file):
            # Try alternative path structure
            model_file = os.path.join(artifacts_path, artifact_path, "model.joblib")
        
        model = joblib.load(model_file)
        return model


def test_load_model_from_run_uri(model):
    assert model is not None, "Failed to load model from run URI"
