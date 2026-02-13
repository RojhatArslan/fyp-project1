"""Feature engineering utilities for demand forecasting."""

from datetime import datetime


def is_holiday(date: datetime) -> int:
    """Check if date is a holiday (simplified implementation)."""
    month_day = (date.month, date.day)
    holidays = {
        (1, 1): "New Year's Day",
        (2, 14): "Valentine's Day",
        (12, 25): "Christmas",
    }
    return 1 if month_day in holidays else 0


def create_features_for_date(date: datetime) -> dict:
    """Create feature vector matching training features."""
    return {
        "day_of_week": date.weekday(),  # 0=Monday, 6=Sunday
        "month": date.month,  # 1-12
        "lag_7": 0.0,  # Historical feature - set to 0 for future dates
        "rolling_7": 0.0,  # Historical feature - set to 0 for future dates
    }
