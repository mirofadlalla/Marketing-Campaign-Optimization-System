import pandas as pd
import logging

from zenml import step

from typing import Annotated , Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def data_load(
    data_path : str,
    ) -> pd.DataFrame :

    try :
        path = Path(data_path)
        extension = path.suffix.lower()

        if extension == '.xlsx':
            df = pd.read_excel(data_path)

        return df
    
    except Exception as e:
        logging.error("Something went wrong while loading the data.")
        raise e
    