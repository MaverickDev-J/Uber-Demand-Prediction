# import json
# import mlflow
# import dagshub
# import logging
# from pathlib import Path
# from mlflow.client import MlflowClient

# import dagshub
# dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)

# # set the mlflow tracking uri
# mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")

# # create a logger
# logger = logging.getLogger("register_model")
# logger.setLevel(logging.INFO)

# # attach a console handler
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)

# # make a formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)


# if __name__ == "__main__":
#     # current path
#     current_path = Path(__file__)
#     # set the root path
#     root_path = current_path.parent.parent.parent
#     # run info file name
#     file_name = "run_information.json"
#     # load the json file
#     try:
#         with open(root_path / file_name, "r") as f:
#             run_info = json.load(f)
#             logger.info("Information loaded successfully")
#     except FileNotFoundError:
#         logger.error(f"File {file_name} not found")
#     except json.JSONDecodeError:
#         logger.error(f"Error decoding JSON from the file {file_name}")
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {e}")
    
#     # register the model
#     model_name = "uber_demand_prediction_model"
#     model_uri = run_info["model_uri"]
#     model_version = mlflow.register_model(model_uri, model_name)
    
#     model_name = model_version.name
#     model_version = model_version.version
    
#     logger.info(f"Model registered successfully with version: {model_version} and name: {model_name}")
    
#     # move the registered model to staging stage
#     model_stage = "Staging"
    
#     client = MlflowClient()
    
#     stage_version = client.transition_model_version_stage(name=model_name,
#                                           version=model_version,
#                                           stage=model_stage,
#                                           archive_existing_versions=False)
    
#     staged_model_name = stage_version.name
#     staged_model_version = stage_version.version
#     staged_model_stage = stage_version.current_stage
    
#     logger.info(f"Model moved to stage: {staged_model_stage} with version: {staged_model_version} and name: {staged_model_name}")








import json
import mlflow
import dagshub
import logging
from pathlib import Path
from mlflow.client import MlflowClient
import sys

import dagshub
dagshub.init(repo_owner='maverick011', repo_name='Uber-Demand-Prediction', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/maverick011/Uber-Demand-Prediction.mlflow")

# create a logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # run info file name
    file_name = "run_information.json"
    
    # load the json file
    try:
        with open(root_path / file_name, "r") as f:
            run_info = json.load(f)
            logger.info("Information loaded successfully")
    except FileNotFoundError:
        logger.error(f"File {file_name} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file {file_name}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    
    # register the model
    model_name = "uber_demand_prediction_model"
    model_uri = run_info["model_uri"]
    
    try:
        # Check if model already exists and get existing versions
        client = MlflowClient()
        try:
            existing_model = client.get_registered_model(model_name)
            logger.info(f"Model {model_name} already exists")
        except mlflow.exceptions.RestException:
            logger.info(f"Model {model_name} does not exist, will create new one")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        registered_model_name = model_version.name
        registered_model_version = model_version.version
        
        logger.info(f"Model registered successfully with version: {registered_model_version} and name: {registered_model_name}")
        
        # Move the registered model to staging stage
        model_stage = "Staging"
        
        try:
            # Check if DagHub supports model stage transitions
            stage_version = client.transition_model_version_stage(
                name=registered_model_name,
                version=registered_model_version,
                stage=model_stage,
                archive_existing_versions=False
            )
            
            staged_model_name = stage_version.name
            staged_model_version = stage_version.version
            staged_model_stage = stage_version.current_stage
            
            logger.info(f"Model moved to stage: {staged_model_stage} with version: {staged_model_version} and name: {staged_model_name}")
            
        except mlflow.exceptions.RestException as stage_error:
            logger.warning(f"Model staging failed (this might not be supported by DagHub): {stage_error}")
            logger.info("Model registration completed successfully, but staging was skipped")
            
        except Exception as stage_error:
            logger.warning(f"Unexpected error during model staging: {stage_error}")
            logger.info("Model registration completed successfully, but staging was skipped")
            
    except mlflow.exceptions.RestException as reg_error:
        logger.error(f"Model registration failed: {reg_error}")
        sys.exit(1)
    except Exception as reg_error:
        logger.error(f"Unexpected error during model registration: {reg_error}")
        sys.exit(1)
    
    logger.info("Model registration process completed successfully")