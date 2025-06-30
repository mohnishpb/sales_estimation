from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from model_inference import predict_price

app = FastAPI(title="VIN Price Estimator")

class VINRequest(BaseModel):
    df: list[dict]

class DataRequest(BaseModel):
    features: dict

def get_vin_data(vin: str):
    vin_search = vin[:8]

    try:
        data = pd.read_csv('data.csv')
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="data.csv not found")

    str_cols = data.select_dtypes(include=['object', 'string']).columns
    data[str_cols] = data[str_cols].apply(lambda x: x.str.strip())

    data_cleaned = data[data['VIN'].str.strip().str[:8] == vin_search].copy()

    if data_cleaned.empty:
        return pd.DataFrame()

    cols = [
        "VIN", "Lot Year", "Lot Make", "Lot Model", "Sale Price",
        "Lot Run Condition", "Sale Title Type", "Damage Type Description",
        "Odometer Reading", "Lot Fuel Type"
    ]
    return data_cleaned[cols].reset_index(drop=True)

def estimate_price_by_vin(df_similar: list[dict]):
    df_similar = pd.DataFrame(df_similar)
    if df_similar.empty:
        return None

    Q1 = df_similar['Sale Price'].quantile(0.25)
    Q3 = df_similar['Sale Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_cleaned = df_similar[(df_similar['Sale Price'] >= lower) & (df_similar['Sale Price'] <= upper)]

    if df_cleaned.empty:
        return float(df_similar['Sale Price'].median())

    return float(df_cleaned['Sale Price'].median())

@app.get("/get-data/{vin}")
def get_data(vin: str):
    df = get_vin_data(vin)
    if df.empty:
        raise HTTPException(status_code=404, detail="No similar VINs found")
    return df.to_dict(orient="records")

@app.post("/estimate_price/")
def estimate_price(request: VINRequest):
    estimated_price = estimate_price_by_vin(request.df)
    return {
        "matches_found": len(request.df),
        "estimated_price": estimated_price
    }

@app.post("/predict_price/")
def predict_price_from_features(request: DataRequest):
    result = predict_price(request.features)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)