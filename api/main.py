"""FastAPI backend for Restaurant Demand Forecasting."""

from datetime import timedelta
from pathlib import Path
from typing import List, Optional
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.features import create_features_for_date

# Initialize FastAPI application
app = FastAPI(
    title="Restaurant Demand Forecasting API",
    description="API for predicting restaurant booking demand",
    version="1.0.0"
)

# Configure CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration - loaded at startup
MODEL_PATH = Path("models/final_xgb_model.pkl")
model: Optional[object] = None
feature_columns: Optional[list] = None


@app.on_event("startup")
async def load_model():
    """Load XGBoost model and feature columns at startup."""
    global model, feature_columns
    
    if not MODEL_PATH.exists():
        print(f"Warning: Model not found at {MODEL_PATH}")
        return
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
            
            # Handle both dict format and direct model object
            if isinstance(model_data, dict):
                model = model_data.get("model")
                feature_columns = model_data.get("feature_columns")
            else:
                model = model_data
                # Default feature columns matching features.py
                feature_columns = ["day_of_week", "month", "lag_7", "rolling_7"]
        
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format", example="2026-02-01")
    horizon_days: int = Field(..., ge=1, le=14, description="Number of days to predict (max 14)", example=7)


class DailyAggregation(BaseModel):
    """Daily aggregated prediction result."""
    date: str
    total_bookings: float
    avg_per_hour: float


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    dates: List[str]
    predictions: List[float]
    aggregated_daily: List[DailyAggregation]


def generate_predictions(start_date: str, horizon_days: int) -> tuple[List[str], List[float]]:
    """Generate predictions for future dates."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Parse start date
    try:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Generate hourly timestamps for forecast horizon
    timestamps = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day in range(horizon_days):
        for hour in range(24):
            timestamps.append(current + timedelta(days=day, hours=hour))
    
    # Create features for each timestamp
    features_list = [create_features_for_date(ts) for ts in timestamps]
    df_features = pd.DataFrame(features_list)
    
    # Ensure feature order matches training
    if feature_columns:
        # Fill missing columns with 0
        missing_cols = [col for col in feature_columns if col not in df_features.columns]
        for col in missing_cols:
            df_features[col] = 0.0
        # Reorder to match training feature order
        df_features = df_features[feature_columns]
    else:
        # Default features from features.py
        default_cols = ["day_of_week", "month", "lag_7", "rolling_7"]
        for col in default_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        df_features = df_features[default_cols]
    
    # Generate predictions
    predictions = model.predict(df_features)
    
    # Format dates and predictions
    dates = [ts.strftime("%Y-%m-%d %H:00:00") for ts in timestamps]
    predictions_list = [float(p) for p in predictions]
    
    return dates, predictions_list


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate demand forecasts for specified date range."""
    dates, predictions = generate_predictions(request.start_date, request.horizon_days)
    
    # Aggregate daily predictions
    daily_aggregated = {}
    for date_str, pred in zip(dates, predictions):
        date_only = date_str.split()[0]  # Extract YYYY-MM-DD
        if date_only not in daily_aggregated:
            daily_aggregated[date_only] = []
        daily_aggregated[date_only].append(pred)
    
    aggregated_daily = [
        DailyAggregation(
            date=date,
            total_bookings=sum(preds),
            avg_per_hour=sum(preds) / len(preds)
        )
        for date, preds in sorted(daily_aggregated.items())
    ]
    
    return PredictionResponse(
        dates=dates,
        predictions=predictions,
        aggregated_daily=aggregated_daily
    )


@app.get("/")
async def root():
    """Root endpoint - serves frontend interface."""
    from fastapi.responses import FileResponse
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "API is running. Use /predict endpoint", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
