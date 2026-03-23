import json
import logging
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import Union


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Run an inference request against a prediction service.
    
    Args:
        service: MLFlow deployment service for making predictions
        data: Preprocessed data (DataFrame or numpy array)
    
    Returns:
        Model predictions as numpy array
    """
    try:
        logging.info("Starting prediction step...")
        
        # Start the service
        service.start(timeout=10)
        logging.info("Prediction service started successfully")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            logging.info(f"Converting DataFrame to array. Shape: {data.shape}")
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Make prediction
        logging.info("Making prediction...")
        prediction = service.predict(data_array)
        
        logging.info(f"Prediction completed. Output shape: {np.array(prediction).shape}")
        print(f"🔥 Prediction: {prediction}")
        
        return np.array(prediction)
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise