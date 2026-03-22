import pandas as pd
import logging

from zenml import step
from typing import Annotated , Tuple

# from steps.feature_engineering import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False, 
      experiment_tracker="exp-tracker_conversion"
    )
@step(enable_cache=False, experiment_tracker="exp-tracker_conversion")
def data_preprosser(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test'],
]:
    try:
        logging.info("Starting preprocessing step")

        # Drop unnecessary columns (NOT target)
        droped_cols = ['Date', 'Campaign_ID']
        df = df.drop([col for col in droped_cols if col in df.columns], axis=1)

        # Remove duplicates
        df = df.drop_duplicates()

        # Remove columns with >80% nulls
        null_features = df.columns[(df.isna().mean() > 0.8)]
        df = df.drop(null_features, axis=1)

        # Handle leakage
        corr = df.corr(numeric_only=True)['Conversion_Rate']
        lek_features = corr[abs(corr) > 0.95].index.tolist()
        lek_features = [col for col in lek_features if col != 'Conversion_Rate']

        # Split data
        from sklearn.model_selection import train_test_split

        X = df.drop(['Conversion_Rate'] + lek_features, axis=1)
        y = df['Conversion_Rate']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # pipeline = build_pipeline(df)

        # # Fit ONLY on train
        # X_train_processed = pipeline.fit_transform(X_train)
        # X_test_processed = pipeline.transform(X_test)

        # logging.info("Feature engineering completed")

        # return X_train_processed, X_test_processed, y_train, y_test

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise