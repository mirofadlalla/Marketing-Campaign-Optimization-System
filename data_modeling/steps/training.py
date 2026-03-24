from zenml import step
from typing import Tuple, Annotated

import numpy as np
import pandas as pd
import mlflow
import logging

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline

@step(experiment_tracker="experiment-tracker_Conversion")
def advanced_training_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pipeline
) -> Tuple[
    Annotated[Pipeline, "model"],
    Annotated[float, "rmse"],
    Annotated[float, "r2"]
]:

    try:
        logging.info("Starting advanced training step (XGBoost)")

        mlflow.set_experiment("conversion_rate_pipeline")

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name="XGBoost_Advanced_Training"):

            # 1. fit preprocessing
            pipeline.fit(X_train, y_train)

            X_train_processed = pipeline.transform(X_train)
            X_test_processed = pipeline.transform(X_test)


            # Hyperparameter tuning with RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [35, 55, 75],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            model = XGBRegressor(random_state=42)

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=10,
                scoring='neg_mean_squared_error',
                cv=3,
                verbose=1,
                random_state=42
            )

            random_search.fit(X_train_processed, y_train)

            best_model = random_search.best_estimator_

            # 3. build full pipeline
            full_pipeline = Pipeline([
                ("preprocessing", pipeline),
                ("model", best_model)
            ])

            # Predictions and evaluation
            y_pred = full_pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            from mlflow.models import infer_signature
            signature = infer_signature(X_train, y_pred)

            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric("train_size", X_train.shape[0])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(
                full_pipeline,
                artifact_path="model",
                registered_model_name="Conversion_Rate_Prediction_Model",
                signature=signature,
                input_example=X_train.head(3)
            )

            return full_pipeline, rmse, r2
            
    except Exception as e:
        logging.error(f"Error in advanced training step: {e}")
        raise e