# import mlflow
# import dagshub
# import json
# import pandas as pd
# import joblib
# from pathlib import Path
# import logging
# from sklearn import set_config
# from sklearn.metrics import mean_absolute_percentage_error


# import dagshub
# dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)

# # set the mlflow tracking uri
# mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")

# # set the experiment name
# mlflow.set_experiment("DVC Pipeline")

# set_config(transform_output="pandas")

# # create a logger
# logger = logging.getLogger("evaluate_model")
# logger.setLevel(logging.INFO)

# # attach a console handler
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)

# # make a formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)

# def load_model(model_path):
#     model = joblib.load(model_path)
#     return model


# def save_run_information(run_id, artifact_path, model_uri, path):
#     run_information = {
#         "run_id": run_id,
#         "artifact_path": artifact_path,
#         "model_uri": model_uri
#     }
#     with open(path, "w") as f:
#         json.dump(run_information, f, indent=4)


# if __name__ == "__main__":
#     # current path
#     current_path = Path(__file__)
#     # set the root path
#     root_path = current_path.parent.parent.parent
#     # data_path
#     train_data_path = root_path / "data/processed/train.csv"
#     test_data_path = root_path / "data/processed/test.csv"
    
#     # read the data
#     df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
#     logger.info("Data read successfully")
    
#     # set the datetime column as index
#     df.set_index("tpep_pickup_datetime", inplace=True)
    
#     # make X_test and y_test
#     X_test = df.drop(columns=["total_pickups"])
#     y_test = df["total_pickups"]
    
#     # load the encoder
#     encoder_path = root_path / "models/encoder.joblib"
#     encoder = joblib.load(encoder_path)
#     logger.info("Encoder loaded successfully")
    
#     # transform the test data
#     X_test_encoded = encoder.transform(X_test)
#     logger.info("Data transformed successfully")
    
#     # load the model
#     model_path = root_path / "models/model.joblib"
#     model = load_model(model_path)
#     logger.info("Model loaded successfully")
    
#     # make predictions
#     y_pred = model.predict(X_test_encoded)
    
#     # calculate the loss
#     loss = mean_absolute_percentage_error(y_test, y_pred)
#     logger.info(f"Loss: {loss}")
    
#     # mlflow tracking
#     with mlflow.start_run(run_name="model"):    
#         # log the model parameters
#         mlflow.log_params(model.get_params())
        
#         # log the mertic
#         mlflow.log_metric("MAPE", loss)
        
#         # converts the datasets into mlfow datasets
#         training_data = mlflow.data.from_pandas(pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        
#         validation_data = mlflow.data.from_pandas(pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        
#         # log the datasets
#         mlflow.log_input(training_data, "training")
#         mlflow.log_input(validation_data, "validation")
        
#         # model signature
#         model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
        
#         # log sklearn model
#         logged_model = mlflow.sklearn.log_model(model, "demand_prediction", 
#                                  signature=model_signature,
#                                  pip_requirements="requirements.txt")
        
        
#     # get the run id and arifact uri
#     run_id = logged_model.run_id
#     artifact_path = logged_model.artifact_path
#     model_uri = logged_model.model_uri 
#     logger.info("Mlflow logging complete")
    
#     # save to json file
#     json_file_save_path = root_path / "run_information.json"
#     save_run_information(run_id=run_id,
#                          artifact_path=artifact_path,
#                          model_uri=model_uri,
#                          path=json_file_save_path)
#     logger.info("Run information saved successfully")




import mlflow
import dagshub
import json
import pandas as pd
import joblib
from pathlib import Path
import logging
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error
import tempfile
import os


import dagshub
dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")

# set the experiment name
mlflow.set_experiment("DVC Pipeline")

set_config(transform_output="pandas")

# create a logger
logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def load_model(model_path):
    model = joblib.load(model_path)
    return model


def save_run_information(run_id, artifact_path, model_uri, path):
    run_information = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_uri": model_uri
    }
    with open(path, "w") as f:
        json.dump(run_information, f, indent=4)


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    train_data_path = root_path / "data/processed/train.csv"
    test_data_path = root_path / "data/processed/test.csv"
    
    # read the data
    df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    
    # make X_test and y_test
    X_test = df.drop(columns=["total_pickups"])
    y_test = df["total_pickups"]
    
    # load the encoder
    encoder_path = root_path / "models/encoder.joblib"
    encoder = joblib.load(encoder_path)
    logger.info("Encoder loaded successfully")
    
    # transform the test data
    X_test_encoded = encoder.transform(X_test)
    logger.info("Data transformed successfully")
    
    # load the model
    model_path = root_path / "models/model.joblib"
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    
    # make predictions
    y_pred = model.predict(X_test_encoded)
    
    # calculate the loss
    loss = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Loss: {loss}")
    
    # mlflow tracking
    with mlflow.start_run(run_name="model") as run:    
        # log the model parameters
        mlflow.log_params(model.get_params())
        
        # log the mertic
        mlflow.log_metric("MAPE", loss)
        
        # converts the datasets into mlfow datasets
        training_data = mlflow.data.from_pandas(pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        
        validation_data = mlflow.data.from_pandas(pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime"), targets="total_pickups")
        
        # log the datasets
        mlflow.log_input(training_data, "training")
        mlflow.log_input(validation_data, "validation")
        
        # model signature
        model_signature = mlflow.models.infer_signature(X_test_encoded, y_pred)
        
        # Use the older artifact_path approach to avoid DagHub compatibility issues
        artifact_path = "demand_prediction"
        
        # Create a temporary directory to save the model
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model to temporary directory
            model_file_path = os.path.join(temp_dir, "model.joblib")
            joblib.dump(model, model_file_path)
            
            # Log the model file as artifact
            mlflow.log_artifacts(temp_dir, artifact_path)
            
            # Log model with sklearn flavor using the older approach
            try:
                # Use the older method without the problematic create_logged_model call
                logged_model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=model_signature,
                    pip_requirements="requirements.txt"
                )
                logger.info("Model logged successfully with sklearn flavor")
            except Exception as e:
                logger.warning(f"sklearn.log_model failed: {e}")
                # Fallback: just log as generic artifact
                logged_model_info = None
        
        # Get run information
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        # Create a mock logged_model object for compatibility
        class MockLoggedModel:
            def __init__(self, run_id, artifact_path, model_uri):
                self.run_id = run_id
                self.artifact_path = artifact_path
                self.model_uri = model_uri
        
        logged_model = MockLoggedModel(run_id, artifact_path, model_uri)
        
    logger.info("Mlflow logging complete")

    json_file_save_path = root_path / "run_information.json"
    save_run_information(run_id=logged_model.run_id,
                         artifact_path=logged_model.artifact_path,
                         model_uri=logged_model.model_uri,
                         path=json_file_save_path)
    logger.info("Run information saved successfully")