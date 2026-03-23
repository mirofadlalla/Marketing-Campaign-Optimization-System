import mlflow
from zenml import step
from zenml.integrations.mlflow.model_registries.mlflow_model_registry import MLFlowModelRegistry
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

class DeploymentTriggerConfig(BaseModel):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.1

@step(enable_cache=False)
def evaluate_and_promote(new_score: float, model_name: str, new_version: str):
    mlflow.set_tracking_uri("http://localhost:8000") 
    client = MlflowClient()
    alias_name = "Production"

    try:
        prod_version = client.get_model_version_by_alias(name=model_name, alias=alias_name)
        run_id = prod_version.run_id
    
        prod_run = client.get_run(run_id)
        prod_score = prod_run.data.metrics.get("r2_score", 0)
        
        print(f"🏆 Current Champion (v{prod_version.version}) R2: {prod_score}")
        print(f"🚀 New Challenger (v{new_version}) R2: {new_score}")

        if new_score > prod_score:
            print(f"✅ Success! Version {new_version} is better. Promoting...")
            client.set_registered_model_alias(model_name, alias_name, new_version)
            return True
        else:
            print(f"Version {new_version} didn't beat the Production version. Keeping the old one.")
            return False

    except mlflow.exceptions.RestException:
        # لو مفيش champion أصلاً (أول مرة تشغل السيستم)
        print("No Production version found. Promoting this version as the first Production version.")
        client.set_registered_model_alias(model_name, alias_name, new_version)
        return True

# مثال لتشغيل الكود:
# هنفترض إنك لسه مخلص تدريب وطلع معاك Version '3' والـ Score كان 0.92
# print(evaluate_and_promote(0.92, "Conversion_Rate_Prediction_Model", "3"))