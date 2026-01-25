import pandas as pd
from pathlib import Path


def build_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Create feature-engineered dataset for daily demand forecasting.
    """

    df = pd.read_csv(input_path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month

    # Lag features
    df["lag_1"] = df["bookings"].shift(1)
    df["lag_7"] = df["bookings"].shift(7)

    # Rolling features
    df["rolling_7"] = df["bookings"].rolling(window=7).mean()

    # Drop rows with NaNs introduced by lags/rolling
    df = df.dropna().reset_index(drop=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    INPUT = Path("data/processed/daily_bookings.csv")
    OUTPUT = Path("data/processed/daily_features.csv")

    build_features(INPUT, OUTPUT)
