from zenml import step
from typing import Tuple, Annotated

import numpy as np
import pandas as pd
import mlflow
import logging

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

@step(experiment_tracker="experiment-tracker_Conversion")
def advanced_training_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[
    Annotated[XGBRegressor, "model"],
    Annotated[float, "rmse"],
    Annotated[float, "r2"]
]:

    try:
        logging.info("Starting advanced training step (XGBoost)")

        mlflow.set_experiment("Conversion_Rate_Prediction")

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name="XGBoost_Advanced_Training"):

            # Hyperparameter tuning with RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
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

            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            # Predictions and evaluation
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.xgboost.log_model(best_model, "model")
            mlflow.xgboost.log_model(best_model, "model")
            
            return best_model, rmse, r2
            
    except Exception as e:
        logging.error(f"Error in advanced training step: {e}")
        raise e