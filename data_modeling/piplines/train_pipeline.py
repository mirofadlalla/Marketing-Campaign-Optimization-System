import logging
from steps.load_df import data_load
from steps.data_preprossing import data_preprosser
from steps.feature_engineering import build_pipeline
from steps.training import advanced_training_step

from steps.apply_fetaure_eng_on_data_preprossing import return_featured_data

from zenml import pipeline

@pipeline(enable_cache=False)
def conversion_rate_pipeline(data_path: str = "/home/omar/Python/Marketing_Campaign_Optimization_System/data/marketing_campaign_dataset.xlsx"):
#     df = data_load(data_path)

#     X_train, X_test, y_train, y_test, df = data_preprosser(df)
    
#     pipeline = build_pipeline(df)

#     X_train_processed, X_test_processed, _, _ = return_featured_data(X_train, X_test, y_train, y_test, pipeline)

#     # # logging.info("Feature engineering completed")

#     model, rmse, r2 = advanced_training_step(X_train_processed, X_test_processed, y_train, y_test, pipeline)

    df = data_load(data_path)

    X_train, X_test, y_train, y_test, df = data_preprosser(df)

    pipeline = build_pipeline(df)

    model, rmse, r2 = advanced_training_step(
        X_train, X_test, y_train, y_test, pipeline
    )