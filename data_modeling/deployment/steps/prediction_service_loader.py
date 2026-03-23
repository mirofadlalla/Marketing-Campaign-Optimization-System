import json
import logging
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.config import DockerSettings


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, 
    pipeline_step_name: str, 
    running: bool = False,
    model_name: str = "model"  
) -> MLFlowDeploymentService: 
    """
    Get the prediction service from the deployment pipeline.
    
    Args:
        pipeline_name: Name of the ZenML pipeline that deployed the model
        pipeline_step_name: Name of the deployment step
        running: Whether to look for running services
        model_name: Name of the registered model
    
    Returns:
        MLFlowDeploymentService instance for making predictions
    """
    try:
        logging.info(f"Loading prediction service for pipeline: {pipeline_name}, step: {pipeline_step_name}")
        
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        
        existing_services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name, 
            running=running,
        )
        
        if not existing_services:
            raise RuntimeError(
                f"No MLflow prediction service found for pipeline '{pipeline_name}' "
                f"and step '{pipeline_step_name}'"
            )
        
        logging.info(f"Found {len(existing_services)} service(s). Using the first one.")
        return existing_services[0]
    
    except Exception as e:
        logging.error(f"Error loading prediction service: {e}")
        raise