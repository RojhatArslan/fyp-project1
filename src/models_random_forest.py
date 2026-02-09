import pandas as pd
from pathlib import Path


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


def train_random_forest(data_path: Path): # Train and evaluate a Random Forest regression model

    df = pd.read_csv(data_path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    
    X = df.drop(columns=["ds", "bookings"])
    y = df["bookings"]


    split_idx = int(len(df) * 0.8)# Time-based split (80% train, 20% test)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

   
    model = RandomForestRegressor(    # Train model
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

  
    y_pred = model.predict(X_test) # Predict

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Results")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    return {
        "model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


if __name__ == "__main__":
    DATA = Path("data/processed/daily_features.csv")
    train_random_forest(DATA)
