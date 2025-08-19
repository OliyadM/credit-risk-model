# src/api/main.py
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the best model
model_path = "./models/random_forest_best.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        model = None
else:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, random_state=42)
    logger.warning("Model file not found, using default Random Forest. Train and save it first.")

# Pydantic model for input validation
class CustomerData(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_min: float
    Amount_max: float
    Value_sum: float
    Value_mean: float
    Value_std: float
    Value_min: float
    Value_max: float
    Hour_mean: float
    Hour_std: float
    Day_mean: float
    Day_std: float
    Month_mean: float
    Month_std: float
    WeekDay_mean: float
    WeekDay_std: float
    DayOfYear_mean: float
    DayOfYear_std: float
    ProviderId_ProviderId_1: int
    ProviderId_ProviderId_4: int
    ProviderId_ProviderId_5: int
    ProviderId_ProviderId_6: int
    ProductId_ProductId_10: int
    ProductId_ProductId_15: int
    ProductId_ProductId_3: int
    ProductId_ProductId_6: int
    ProductCategory_airtime: int
    ProductCategory_financial_services: int
    ChannelId_ChannelId_2: int
    ChannelId_ChannelId_3: int
    PricingStrategy_2: int
    PricingStrategy_4: int
    Transaction_Count: int

@app.post("/predict")
async def predict_risk(customer_data: CustomerData):
    logger.info("Received prediction request with data: %s", customer_data.dict())
    if model is None:
        logger.error("Model not available for prediction")
        return {"error": "Model not loaded", "is_high_risk": -1, "risk_probability": -1.0}
    
    try:
        data = pd.DataFrame([customer_data.dict().values()], columns=customer_data.dict().keys())
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1][0]
        logger.info("Prediction successful: is_high_risk=%d, probability=%.3f", prediction[0], probability)
        return {"is_high_risk": int(prediction[0]), "risk_probability": float(probability)}
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        return {"error": str(e), "is_high_risk": -1, "risk_probability": -1.0}

if __name__ == "__main__":
    logger.info("Starting FastAPI application on http://0.0.0.0:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)