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
        "hour": date.hour,  # 0-23
        "day_of_week": date.weekday(),  # 0=Monday, 6=Sunday
        "month": date.month,  # 1-12
        "is_peak_hour": 1 if date.hour in [18, 19, 20, 21] else 0,  # Peak dinner hours
        "NewYearsDay": 1 if (date.month, date.day) == (1, 1) else 0,
        "ValentinesDay": 1 if (date.month, date.day) == (2, 14) else 0,
        "MothersDay": 0,  # Simplified - would need actual date calculation
        "EasterSunday": 0,  # Simplified - would need actual date calculation
        "ChristmasDay": 1 if (date.month, date.day) == (12, 25) else 0,
        "total_guests": 0.0,  # Cannot predict for future - set to 0
        "avg_party_size": 0.0,  # Cannot predict for future - set to 0
        "lag_7": 0.0,  # Historical feature - unavailable for future dates
        "rolling_7": 0.0,  # Historical feature - unavailable for future dates
    }
