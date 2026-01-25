import pandas as pd
from pathlib import Path


def prepare_daily_bookings(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Load raw booking data, aggregate to daily demand, and save the result.

    Parameters
    ----------
    input_path : Path
        Path to raw bookings CSV.
    output_path : Path
        Path to save aggregated daily bookings CSV.

    Returns
    -------
    pd.DataFrame
        Daily aggregated bookings with columns: ds, bookings.
    """

    # Load raw data
    df = pd.read_csv(input_path)

    # Parse booking datetime
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"], errors="coerce")

    # Drop rows with missing critical values
    df = df.dropna(subset=["booking_datetime"])

    # Create date column
    df["ds"] = df["booking_datetime"].dt.date

    # Daily aggregation
    daily = (
        df.groupby("ds")
        .size()
        .reset_index(name="bookings")
    )

    # Convert ds back to datetime (recommended for time-series)
    daily["ds"] = pd.to_datetime(daily["ds"])

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save aggregated data
    daily.to_csv(output_path, index=False)

    return daily


if __name__ == "__main__":
    INPUT = Path("data/raw/bookings.csv")
    OUTPUT = Path("data/processed/daily_bookings.csv")

    prepare_daily_bookings(INPUT, OUTPUT)