from ast import Global
import asyncio
import logging
import pickle 
from fastapi import FastAPI , HTTPException , Depends 
from pydantic import BaseModel , Field
from contextlib import asynccontextmanager
from fastapi.security import APIKeyHeader
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd

from get_production_model import return_model
# from rate_limitizer import RateLimiter

model = None

def load_model(model_name : str = 'Conversion_Rate_Prediction_Model'):
    global model 
    try :
        model =return_model(model_name)
    except Exception as e:
        raise e

@asynccontextmanager
async def lifespan(app : FastAPI ):
    load_model('Conversion_Rate_Prediction_Model')
    print("Models Loaded Sussfully ... ")

    yield

    print("Models Closed Sussfully ... ")

app = FastAPI(lifespan=lifespan)
# rate_limit = RateLimiter("omar_1234")

class ConversionRate(BaseModel):
    Company: str = Field(..., description="Company name")
    Campaign_Type: str = Field(..., description="Type of campaign")
    Target_Audience: str = Field(..., description="Target audience")
    Duration: str = Field(..., description="Campaign duration")
    Channel_Used: str = Field(..., description="Channel used for campaign")
    Acquisition_Cost: int = Field(..., description="Acquisition cost in USD")
    ROI: float = Field(..., description="Return on Investment")
    Location: str = Field(..., description="Location")
    Clicks: int = Field(..., description="Number of clicks")
    Impressions: int = Field(..., description="Number of impressions")
    Engagement_Score: int = Field(..., description="Engagement score")
    Customer_Segment: str = Field(..., description="Customer segment")


@app.post("/predict")
async def predict_conversion_rate(data: ConversionRate):
    # if not rate_limit.check():
    #     raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    try:
        logging.info(f"Received prediction request: {data.dict()}")
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)
        return {"predicted_conversion_rate": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))