import logging
import sys
from pathlib import Path
import pandas as pd
from zenml import pipeline

# Add parent directory to path so we can import steps
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from steps.load_df import data_load
from steps.feature_engineering import build_pipeline
from deployment.steps.inference_data_loader import inference_data_preprocessor
from deployment.steps.prediction_service_loader import prediction_service_loader
from deployment.steps.predictor import predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str = "training_deployment_pipeline",
    pipeline_step_name: str = "mlflow_model_deployer_step",
    data_path: str = "/home/omar/Python/Marketing_Campaign_Optimization_System/data/marketing_campaign_dataset.xlsx"
):
    """
    Inference pipeline for making predictions on new data.
    
    This pipeline:
    1. Loads new data from the specified path
    2. Applies the same preprocessing as training
    3. Loads the deployed model service
    4. Makes predictions
    
    Args:
        pipeline_name: Name of the training pipeline that deployed the model
        pipeline_step_name: Name of the deployment step
        data_path: Path to inference data
    
    Returns:
        Predictions from the model
    """
    logging.info("Starting inference pipeline...")
    
    # Load raw inference data
    logging.info(f"Loading inference data from {data_path}...")
    raw_data = data_load(data_path=data_path)
    # logging.info(f"Data loaded. Shape: {raw_data.shape}")
    
    # Load the preprocessing pipeline (same as training)
    logging.info("Building preprocessing pipeline...")
    preprocessing_pipeline = build_pipeline(raw_data)
    
    # Preprocess the inference data
    logging.info("Preprocessing inference data...")
    preprocessed_data = inference_data_preprocessor(
        df=raw_data,
        pipeline=preprocessing_pipeline
    )
    # logging.info(f"Data preprocessed. Shape: {preprocessed_data.shape}")
    
    # Load the prediction service
    logging.info(f"Loading prediction service from pipeline: {pipeline_name}...")
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    
    # Make predictions
    logging.info("Making predictions...")
    predictions = predictor(
        service=model_deployment_service,
        data=preprocessed_data
    )
    
    logging.info("Inference pipeline completed")
    return predictions


if __name__ == "__main__":
    # Example usage
    predictions = inference_pipeline()
    logging.info(f"Predictions: {predictions}")
