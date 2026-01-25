import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_eda(input_path: Path, output_dir: Path) -> None: # Perform basic exploratory data analysis on daily bookings data.
 

    df = pd.read_csv(input_path, parse_dates=["ds"])

    output_dir.mkdir(parents=True, exist_ok=True)

  
    plt.figure(figsize=(10, 4)) # 1. Daily bookings over time
    plt.plot(df["ds"], df["bookings"])
    plt.title("Daily Restaurant Bookings Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "daily_trend.png", dpi=300)
    plt.close()

    df["day_of_week"] = df["ds"].dt.day_name()  # 2. Average bookings by day of week
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    avg_by_dow = df.groupby("day_of_week")["bookings"].mean().reindex(order)

    plt.figure(figsize=(8, 4))
    avg_by_dow.plot(kind="bar")
    plt.title("Average Daily Bookings by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Bookings")
    plt.tight_layout()
    plt.savefig(output_dir / "weekday_seasonality.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    INPUT = Path("data/processed/daily_bookings.csv")
    OUTPUT = Path("results/figures")

    run_eda(INPUT, OUTPUT)
