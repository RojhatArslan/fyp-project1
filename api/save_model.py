"""Utility script to train and save XGBoost model for API use."""

import pickle
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_features


def train_and_save_model(data_path: Path, model_output_path: Path):
    """Train XGBoost model and save with feature columns."""
    # Load and prepare features
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    
    # Features and target
    X = df.drop(columns=["ds", "bookings"])
    y = df["bookings"]
    
    # Train model
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    
    model.fit(X, y)
    
    # Save model with feature columns
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": model,
        "feature_columns": X.columns.tolist()
    }
    
    with open(model_output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_output_path}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    return model, X.columns.tolist()


if __name__ == "__main__":
    DATA_PATH = Path("data/processed/daily_features.csv")
    MODEL_PATH = Path("models/final_xgb_model.pkl")
    
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    train_and_save_model(DATA_PATH, MODEL_PATH)
