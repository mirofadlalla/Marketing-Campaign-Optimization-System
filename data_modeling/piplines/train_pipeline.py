import logging

from steps.load_df import data_load
from steps.data_preprossing import data_preprosser
from steps.feature_engineering import build_pipeline
from steps.training import advanced_training_step

from zenml import pipeline

@pipeline(enable_cache=False, experiment_tracker="exp-tracker_conversion")
def conversion_rate_pipeline(data_path: str = "E:\pyDS\Marketing_Campaign_Optimization_System\data\marketing_campaign_dataset.xlsx"):
    df = data_load(data_path)

    X_train, X_test, y_train, y_test = data_preprosser(df)
    
    pipeline = build_pipeline(df)

    # Fit ONLY on train
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # logging.info("Feature engineering completed")

    model, rmse, r2 = advanced_training_step(X_train_processed, X_test_processed, y_train, y_test)