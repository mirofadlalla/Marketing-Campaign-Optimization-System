import mlflow
import logging
from zenml import step
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DeploymentTriggerConfig(BaseModel):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.1


@step(enable_cache=False)
def evaluate_and_promote(new_score: float, model_name: str, new_version: str):
    """
    Compare the new model against the current production model and promote if better.
    
    Args:
        new_score: Performance metric (R2 score) of the new model
        model_name: Name of the registered model in MLFlow
        new_version: Version number of the new model
    
    Returns:
        True if the new model was promoted, False otherwise
    """
    try:
        mlflow.set_tracking_uri("http://localhost:8000")
        client = MlflowClient()
        alias_name = "Production"
        
        logging.info(f"Evaluating model {model_name} v{new_version} with score {new_score}")

        try:
            # Get the current production version
            prod_version = client.get_model_version_by_alias(name=model_name, alias=alias_name)
            run_id = prod_version.run_id
        
            prod_run = client.get_run(run_id)
            prod_score = prod_run.data.metrics.get("r2_score", 0)
            
            logging.info(f"🏆 Current Champion (v{prod_version.version}) R2: {prod_score}")
            logging.info(f"🚀 New Challenger (v{new_version}) R2: {new_score}")

            if new_score > prod_score:
                logging.info(f"✅ Success! Version {new_version} is better. Promoting...")
                client.set_registered_model_alias(model_name, alias_name, new_version)
                return True
            else:
                logging.info(f"Version {new_version} didn't beat the Production version. Keeping the old one.")
                return False

        except mlflow.exceptions.RestException:
            # No production version exists yet (first deployment)
            logging.info("No Production version found. Promoting this version as the first Production version.")
            client.set_registered_model_alias(model_name, alias_name, new_version)
            return True
    
    except Exception as e:
        logging.error(f"Error in evaluate_and_promote: {e}")
        raise
