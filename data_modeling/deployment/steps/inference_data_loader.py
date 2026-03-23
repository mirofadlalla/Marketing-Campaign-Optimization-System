import logging
import pandas as pd
import numpy as np
from zenml import step
from typing import Annotated, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@step(enable_cache=False)
def inference_data_preprocessor(
    df: pd.DataFrame,
    pipeline
) -> Annotated[pd.DataFrame, "X_inference_processed"]:
    """
    Preprocess inference data using the same pipeline as training data.
    
    Args:
        df: Raw inference data
        pipeline: Fitted preprocessing pipeline from training
    
    Returns:
        Preprocessed inference data ready for prediction
    """
    try:
        logging.info("Starting inference data preprocessing step")
        
        # Apply the same preprocessing steps as training data
        logging.info("Applying data preprocessing...")
        
        # Drop unnecessary columns (same as in training)
        droped_cols = ['Date', 'Campaign_ID']
        df = df.drop([col for col in droped_cols if col in df.columns], axis=1)
        
        logging.info(f"Remaining columns: {df.columns.tolist()}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        logging.info(f"Shape after dropping duplicates: {df.shape}")
        
        # Remove columns with >80% nulls
        null_features = df.columns[(df.isna().mean() > 0.8)]
        df = df.drop(null_features, axis=1)
        
        logging.info(f"Shape after dropping nulls: {df.shape}")
        
        # Handle leakage (remove target if present for inference data)
        if 'Conversion_Rate' in df.columns:
            X_inference = df.drop('Conversion_Rate', axis=1)
        else:
            X_inference = df
        
        logging.info("Applying feature engineering pipeline...")
        
        # Apply the fitted pipeline (transform only, no fit)
        X_inference_processed = pipeline.transform(X_inference)
        
        # Convert to DataFrame for consistency
        try:
            feature_names = pipeline.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_inference_processed.shape[1])]
        
        X_inference_processed = pd.DataFrame(
            X_inference_processed, 
            columns=feature_names, 
            index=X_inference.index
        )
        
        logging.info(f"Inference data processed successfully. Shape: {X_inference_processed.shape}")
        return X_inference_processed
    
    except Exception as e:
        logging.error(f"Error in inference data preprocessing: {e}")
        raise
