import logging
import pandas as pd
from zenml import step

from typing import Annotated, Tuple

@step(enable_cache=False)
def return_featured_data(X_train : pd.DataFrame, 
                         X_test : pd.DataFrame,
                         y_train : pd.Series,
                         y_test : pd.Series,
                         pipeline
                        ) -> Tuple[
                            Annotated[pd.DataFrame, "X_train_processed"],
                            Annotated[pd.DataFrame, "X_test_processed"],
                            Annotated[pd.Series, "y_train"],
                            Annotated[pd.Series, "y_test"],
                        ]:
    try:
        logging.info("Starting applying feature engineering step on data")

        # Fit ONLY on train (need to pass y_train for feature selection)
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        X_test_processed = pipeline.transform(X_test)

        # Convert numpy arrays back to DataFrames to match expected output types
        # Get feature names from the pipeline if available
        try:
            feature_names = pipeline.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

        return X_train_processed, X_test_processed, y_train, y_test 
    
    except Exception as e:
        logging.error(f"Couldn't process the feature engineering step on data: {e}")
        raise

    