import pandas as pd
import logging

from zenml import step

from typing import Annotated , Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=True)
def data_load(
    data_path,
    ) -> pd.DataFrame :

    try :
        logging.info(f"Loading data from {data_path}...")
        path = Path(data_path)
        extension = path.suffix.lower()

        if extension == '.xlsx':
            logging.info("Excel file detected. Loading with pandas...")
            df = pd.read_excel(data_path)
        
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        logging.info("Data loaded successfully.")
        logging.info(f"Shape of the loaded data: {df.shape}")
        return df
    
    except Exception as e:
        logging.error("Something went wrong while loading the data.")
        raise e
    