#Walk forward is a time aware way of testing models when my data has a order, like bookings over days and hours
 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


df = pd.read_csv("data/processed/hourly_bookings_cleaned.csv")


df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h") # Create datetime column


df = df.sort_values("datetime").reset_index(drop=True) # Sort by time 


y = df["total_bookings"] # Target variable


X = df.drop(columns=[  # Feature columns (drop non-features)
    "date",
    "hour",
    "datetime",
    "total_bookings",
    "day_of_week",   # categorical (already encoded indirectly via hour/month)
    "month"          # same reasoning unless one-hot encoded
])

train_size = int(len(df) * 0.6) # Walk-forward parameters
step_size = int(len(df) * 0.1)

results = []

for start in range(0, len(df) - train_size, step_size):  # iteratees over the dataset in time order and make seure we dont run past the end of the dataset
    train_end = start + train_size # Defines where the training period ends
    test_end = train_end + step_size # Defines the end of the test period

    X_train = X.iloc[start:train_end]
    y_train = y.iloc[start:train_end]
    # Selects the feature matrix X and target vector y for the training period

    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]
    # Selects the next unseen time window for testing.

    model = XGBRegressor( # Fresh XGboost model for each fold , which allows parameters to balance the bias
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train) # Trains using only past data
    preds = model.predict(X_test) # Predicts the target variable for the test period

    results.append({
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    })

results_df = pd.DataFrame(results)
print(results_df)
