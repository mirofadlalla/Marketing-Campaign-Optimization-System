"""Deployment module for ZenML model serving."""

from deployment.pipelines.training_deployment import training_deployment_pipeline
from deployment.pipelines.inference import inference_pipeline

__all__ = [
    "training_deployment_pipeline",
    "inference_pipeline",
]
