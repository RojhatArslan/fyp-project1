import pandas as pd 
from pathlid import Path


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error import mean_absolute_error, mean_absolute_percentage_error, r2_score


def train_linear_regression(data_path: Path)
    df = pd.read_csv(data_path)
    df = df.sort_values("ds").reset_index(drop=True)

    X = df.drop(columns=["ds", "bookings"]) # Featuress and targets
    y = df["bookings"]


    