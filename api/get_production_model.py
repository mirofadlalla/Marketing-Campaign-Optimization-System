import mlflow
import logging
# from zenml import step
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def return_model_run_id( model_name: str):
    try:
        mlflow.set_tracking_uri("http://localhost:8000")
        client = MlflowClient()
        alias_name = "Production"
        
        logging.info(f"Evaluating model {model_name}")
        try:
            # Get the current production version
            prod_version = client.get_model_version_by_alias(name=model_name, alias=alias_name)
            run_id = prod_version.run_id

            return run_id 
        except mlflow.exceptions.RestException:
            # No production version exists yet (first deployment)
            logging.info("No Production version found. Promoting this version as the first Production version.")
            return True
    
    except Exception as e:
        logging.error(f"Error in evaluate_and_promote: {e}")
        raise

def return_model(model_name):
    ''' 
    Get the run ID of the current production model 
    and load it using MLflow's pyfunc interface.
    '''
    run_id = return_model_run_id(model_name)
    model_uri = f"runs:/{run_id}/model" 
    return mlflow.pyfunc.load_model(model_uri)