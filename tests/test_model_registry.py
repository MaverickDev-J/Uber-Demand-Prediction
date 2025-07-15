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
import mlflow.sklearn
import pytest
from pathlib import Path
import dagshub

# Initialize Dagshub
dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")


def load_model_information(file_path: str):
    with open(file_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model():
    # Load model URI from run_information.json
    file_path = Path(__file__).resolve().parents[1] / "run_information.json"
    model_uri = load_model_information(file_path)["model_uri"]
    
    # Load model using URI
    return mlflow.sklearn.load_model(model_uri)


def test_load_model_from_run_uri(model):
    assert model is not None, "Failed to load model from run URI"
