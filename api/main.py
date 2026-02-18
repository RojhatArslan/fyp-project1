"""FastAPI backend for Restaurant Demand Forecasting."""

from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from api.features import create_features_for_date
except ImportError:
    from features import create_features_for_date

# Paths resolved before lifespan so they are available at startup
_base_dir = Path(__file__).resolve().parent.parent
MODEL_PATH = _base_dir / "models" / "final_xgb_model.pkl"
HISTORICAL_DATA_PATH = _base_dir / "data" / "processed" / "daily_bookings.csv"
model: Optional[object] = None
feature_columns: Optional[list] = None
historical_df: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global model, feature_columns, historical_df
    
    print(f"Starting model load...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model path exists: {MODEL_PATH.exists()}")
    
    # Load model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    model = model_data.get("model")
                    feature_columns = model_data.get("feature_columns")
                else:
                    model = model_data
                    feature_columns = None
            
            print(f"Model loaded from {MODEL_PATH}")
            if feature_columns:
                print(f"Feature columns: {feature_columns}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
    
    # Load historical data
    try:
        if HISTORICAL_DATA_PATH.exists():
            temp_df = pd.read_csv(HISTORICAL_DATA_PATH)
            if "date" in temp_df.columns:
                temp_df["ds"] = pd.to_datetime(temp_df["date"])
            elif "ds" in temp_df.columns:
                temp_df["ds"] = pd.to_datetime(temp_df["ds"])
            if "ds" in temp_df.columns:
                historical_df = temp_df.sort_values("ds").reset_index(drop=True)
                print(f"Historical data loaded: {len(historical_df)} records")
            else:
                historical_df = None
        else:
            historical_df = None
    except Exception as e:
        print(f"Error loading historical data: {e}")
        historical_df = None
    
    yield
    
    # Shutdown (if needed)
    print("Shutting down...")


# Initialize FastAPI application
app = FastAPI(
    title="Restaurant Demand Forecasting API",
    description="API for predicting restaurant booking demand",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and historical data (loaded at startup, see lifespan above)




# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format", json_schema_extra={"example": "2026-02-01"})
    horizon_days: int = Field(..., ge=1, le=14, description="Number of days to predict (max 14)", json_schema_extra={"example": 7})
    include_previous: bool = Field(False, description="Include historical bookings for previous 30 days for comparison")


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
    previous_dates: Optional[List[str]] = None
    previous_bookings: Optional[List[float]] = None
    previous_year_dates: Optional[List[str]] = None  # Same period last year
    previous_year_bookings: Optional[List[float]] = None


def _get_hourly_patterns() -> Optional[pd.Series]:
    """Build average bookings per (day_of_week, hour) from historical data. Used for simple forecast."""
    if historical_df is None and not _ensure_historical_loaded():
        return None
    if historical_df is None:
        return None
    df = historical_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["dow"] = df["ds"].dt.dayofweek
    if "hour" not in df.columns:
        return None
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    bookings_col = "total_bookings" if "total_bookings" in df.columns else "bookings"
    if bookings_col not in df.columns:
        return None
    return df.groupby(["dow", "hour"], as_index=True)[bookings_col].mean()


def generate_predictions(start_date: str, horizon_days: int) -> tuple[List[str], List[float]]:
    """Generate predictions: use historical hourly patterns when available, else model."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Build hourly timestamps for the horizon
    timestamps = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    for day in range(horizon_days):
        for hour in range(24):
            timestamps.append(current + timedelta(days=day, hours=hour))

    # Prefer simple forecast from historical patterns (varies by hour and day)
    patterns = _get_hourly_patterns()
    if patterns is not None and len(patterns) > 0:
        dates = [ts.strftime("%Y-%m-%d %H:00:00") for ts in timestamps]
        predictions_list = []
        for ts in timestamps:
            key = (ts.weekday(), int(ts.hour))
            try:
                pred = float(patterns.loc[key])
            except (KeyError, TypeError):
                pred = 1.0
            predictions_list.append(pred)
        return dates, predictions_list

    # Fallback: XGBoost model (can be constant when lag/rolling features are zero)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded and no historical data for simple forecast")

    features_list = [create_features_for_date(ts) for ts in timestamps]
    df_features = pd.DataFrame(features_list)
    if feature_columns:
        missing_cols = [col for col in feature_columns if col not in df_features.columns]
        for col in missing_cols:
            df_features[col] = 0.0
        df_features = df_features[feature_columns]
    else:
        default_cols = ["hour", "day_of_week", "month", "is_peak_hour",
                        "NewYearsDay", "ValentinesDay", "MothersDay",
                        "EasterSunday", "ChristmasDay", "total_guests",
                        "avg_party_size", "lag_7", "rolling_7"]
        for col in default_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        df_features = df_features[default_cols]

    predictions = model.predict(df_features)
    dates = [ts.strftime("%Y-%m-%d %H:00:00") for ts in timestamps]
    predictions_list = [float(p) for p in predictions]
    return dates, predictions_list


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate demand forecasts for specified date range. Optionally include previous historical data."""
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
    
    # Same period last year (one value per forecast day for comparison chart)
    previous_year_dates_list: Optional[List[str]] = None
    previous_year_bookings_list: Optional[List[float]] = None
    if _ensure_historical_loaded() and historical_df is not None:
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            start_last_year = start.replace(year=start.year - 1)
            n_days = request.horizon_days
            target_dates = [start_last_year + timedelta(days=i) for i in range(n_days)]
            bookings_col = "bookings" if "bookings" in historical_df.columns else "total_bookings"
            df = historical_df.copy()
            df["ds"] = pd.to_datetime(df["ds"])
            daily = df.groupby(df["ds"].dt.date)[bookings_col].sum()
            previous_year_bookings_list = [float(daily.get(d.date(), 0.0)) for d in target_dates]
            previous_year_dates_list = [d.strftime("%Y-%m-%d") for d in target_dates]
        except Exception:
            pass

    return PredictionResponse(
        dates=dates,
        predictions=predictions,
        aggregated_daily=aggregated_daily,
        previous_dates=None,
        previous_bookings=None,
        previous_year_dates=previous_year_dates_list,
        previous_year_bookings=previous_year_bookings_list,
    )


@app.get("/")
async def root():
    """Root endpoint - serves frontend interface."""
    from fastapi.responses import FileResponse
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "API is running. Use /predict endpoint", "docs": "/docs"}


def _ensure_historical_loaded() -> bool:
    """Load historical data from file if not already loaded."""
    global historical_df
    if historical_df is not None:
        return True
    if not HISTORICAL_DATA_PATH.exists():
        return False
    try:
        temp_df = pd.read_csv(HISTORICAL_DATA_PATH)
        if "date" in temp_df.columns:
            temp_df["ds"] = pd.to_datetime(temp_df["date"])
        elif "ds" in temp_df.columns:
            temp_df["ds"] = pd.to_datetime(temp_df["ds"])
        if "ds" in temp_df.columns:
            historical_df = temp_df.sort_values("ds").reset_index(drop=True)
            return True
    except Exception:
        pass
    return False


@app.get("/historical")
async def get_historical(start_date: str, end_date: str):
    """Get historical booking data for date range."""
    if historical_df is None and not _ensure_historical_loaded():
        raise HTTPException(status_code=503, detail="Historical data not loaded")
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Filter historical data - handle both datetime and date columns
    if "ds" in historical_df.columns:
        mask = (pd.to_datetime(historical_df["ds"]) >= start) & (pd.to_datetime(historical_df["ds"]) <= end)
        filtered = historical_df[mask].copy()
    else:
        return {"dates": [], "bookings": [], "message": "No 'ds' column found in historical data"}
    
    if len(filtered) == 0:
        return {"dates": [], "bookings": [], "message": "No data found for date range"}
    
    # Check which columns exist - handle hourly data aggregation
    bookings_col = "bookings" if "bookings" in filtered.columns else ("total_bookings" if "total_bookings" in filtered.columns else None)
    guests_col = "total_guests" if "total_guests" in filtered.columns else None
    
    if bookings_col is None:
        return {"dates": [], "bookings": [], "message": "No bookings column found"}
    
    # Aggregate by date (for daily view) - handle hourly data
    filtered["ds"] = pd.to_datetime(filtered["ds"])
    daily_agg = filtered.groupby(filtered["ds"].dt.date).agg({
        bookings_col: "sum",
        **({guests_col: "sum"} if guests_col else {})
    }).reset_index()
    
    result = {
        "dates": [str(d) for d in daily_agg["ds"]],
        "bookings": daily_agg[bookings_col].tolist()
    }
    
    if guests_col:
        result["total_guests"] = daily_agg[guests_col].tolist()
    
    return result


@app.get("/health")
async def health():
    """Health check endpoint. Use this to see why forecast might not vary (e.g. no historical data)."""
    patterns = _get_hourly_patterns()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "historical_data_loaded": historical_df is not None,
        "historical_records": len(historical_df) if historical_df is not None else 0,
        "hourly_patterns_available": patterns is not None and len(patterns) > 0,
        "hourly_pattern_count": len(patterns) if patterns is not None else 0,
        "historical_data_path": str(HISTORICAL_DATA_PATH),
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    import socket
    import uvicorn

    def find_free_port(start: int = 8000, max_tries: int = 5) -> int:
        for i in range(max_tries):
            port = start + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        return 8000  # fallback

    port = find_free_port()
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
