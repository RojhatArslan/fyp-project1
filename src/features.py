import pandas as pd
from pathlib import Path


def build_features(input_path: Path, output_path: Path) -> pd.DataFrame:
   
    df = pd.read_csv(input_path)

     
    date_col = df.columns[0]  # Ensure first column is datetime and named ds
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "ds"})

    df = df.sort_values("ds").reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df["ds"].dt.dayofweek  # Calendar features
    df["month"] = df["ds"].dt.month
    if "bookings" not in df.columns:
        if "total_bookings" in df.columns:
            df = df.rename(columns={"total_bookings": "bookings"})
        else:
            raise ValueError("No bookings column found in input data.")

   
  
    df["lag_7"] = df["bookings"].shift(7)


    df["rolling_7"] = df["bookings"].rolling(window=7).mean()    # Rolling features


    df = df.dropna().reset_index(drop=True)  # Drop rows with NaNs introduced by lags/rolling

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    INPUT = Path("data/processed/daily_bookings.csv")
    OUTPUT = Path("data/processed/daily_features.csv")

    build_features(INPUT, OUTPUT)
