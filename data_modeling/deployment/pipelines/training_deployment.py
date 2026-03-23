import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.config import DockerSettings

# Add parent directory to path so we can import steps
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deployment.utils.deployment_trigger import evaluate_and_promote
from steps.load_df import data_load
from steps.data_preprossing import data_preprosser
from steps.feature_engineering import build_pipeline
from steps.training import advanced_training_step
from steps.apply_fetaure_eng_on_data_preprossing import return_featured_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

docker_settings = DockerSettings(required_integrations=["mlflow", "sklearn"])

@pipeline(
    enable_cache=False,
    settings={
        "docker": docker_settings
    }
)
def training_deployment_pipeline(
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    Training and deployment pipeline for the conversion rate prediction model.
    
    This pipeline:
    1. Loads marketing campaign data
    2. Preprocesses the data
    3. Applies feature engineering
    4. Trains the model
    5. Evaluates and promotes the best model
    6. Deploys it with MLFlow
    """
    logging.info("Starting training and deployment pipeline...")
    
    # Load data
    df = data_load(data_path="/home/omar/Python/Marketing_Campaign_Optimization_System/data/marketing_campaign_dataset.xlsx")
    # logging.info(f"Data loaded. Shape: {df.shape}")

    # Preprocess
    X_train, X_test, y_train, y_test, df = data_preprosser(df)
    logging.info("Data preprocessing completed")
    
    # Build feature engineering pipeline
    pipeline = build_pipeline(df)
    logging.info("Feature engineering pipeline built")

    # Apply feature engineering
    X_train_processed, X_test_processed, _, _ = return_featured_data(X_train, X_test, y_train, y_test, pipeline)
    logging.info("Feature engineering applied")

    # Train model
    model, rmse, r2 = advanced_training_step(X_train_processed, X_test_processed, y_train, y_test)
    # logging.info(f"Model trained. R2 Score: {r2}, RMSE: {rmse}")

    # Evaluate and decide on deployment
    deployment_decision = evaluate_and_promote(
        new_score=r2,
        model_name="Conversion_Rate_Prediction_Model",
        new_version="2"
    )
    
    # Deploy the model
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
    
    logging.info("Training and deployment pipeline completed")


if __name__ == "__main__":
    training_deployment_pipeline()
