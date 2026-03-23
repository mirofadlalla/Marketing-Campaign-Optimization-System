import pandas as pd
import logging
from zenml import step

import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest, f_regression

@step(enable_cache=True)
def build_pipeline(df : pd.DataFrame):
    logging.info("Building preprocessing pipeline...")

    logging.info("Identifying numeric and categorical columns...")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    logging.info(f"Numeric columns: {num_cols}")
    logging.info(f"Categorical columns: {cat_cols}")

    logging.info("Setting up preprocessing pipelines...")
    num_pipline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore")),
    ])

    logging.info("Combining pipelines into a ColumnTransformer...")
    preprocessor = ColumnTransformer([
        ('nums', num_pipline, num_cols),
        ('cats', cat_pipline, cat_cols),
    ])

    logging.info("Building final pipeline with feature selection...")
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k=10))
    ])

    logging.info("Pipeline built successfully.")
    
    return final_pipeline