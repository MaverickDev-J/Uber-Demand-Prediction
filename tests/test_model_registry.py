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
import mlflow.sklearn
import pytest
from pathlib import Path
import dagshub
import tempfile
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
    
    # Download the model artifact directly
    client = mlflow.MlflowClient()
    
    # Create a temporary directory to download the model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the model artifact
        model_path = client.download_artifacts(
            run_id=run_id,
            path=artifact_path,
            dst_path=temp_dir
        )
        
        # Load the model from the downloaded path
        model = mlflow.sklearn.load_model(model_path)
        
        return model


def test_load_model_from_run_uri(model):
    assert model is not None, "Failed to load model from run URI"
