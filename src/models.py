import pandas as pd 
from pathlib import Path


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, mean_absolute_percentage_error, r2_score



def train_linear_regression(data_path: Path):

    df = pd.read_csv(data_path)
    df = df.sort_values("ds").reset_index(drop=True)

    X = df.drop(columns=["ds", "bookings"]) # Featuress and targets
    y = df["bookings"]


    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
        
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred,squared= False)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")

    return mae,rmse,r2
    