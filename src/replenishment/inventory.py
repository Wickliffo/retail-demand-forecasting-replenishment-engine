from statistics import NormalDist

import pandas as pd


def calculate_inventory_policy(
    historical_demand: pd.Series,
    lead_time_days: int,
    service_level: float = 0.95,
) -> dict:
    """
    Calculate a baseline inventory policy from historical daily demand.

    Safety stock:
        z * daily_demand_std * sqrt(lead_time_days)

    Reorder point:
        expected_lead_time_demand + safety_stock

    Assumptions:
    - daily demand observations are sufficiently independent
    - demand distribution is approximately normal
    - lead time is fixed
    - daily demand distribution is reasonably stable
    """
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least one day.")

    if not 0 < service_level < 1:
        raise ValueError("Service level must be between 0 and 1.")

    if historical_demand.empty:
        raise ValueError("Historical demand cannot be empty.")

    if historical_demand.isna().any():
        raise ValueError("Historical demand contains missing values.")

    if (historical_demand < 0).any():
        raise ValueError("Historical demand cannot be negative.")

    daily_mean = float(historical_demand.mean())
    daily_std = float(historical_demand.std(ddof=1))

    if pd.isna(daily_std):
        raise ValueError("At least two observations are required.")

    z_score = NormalDist().inv_cdf(service_level)

    expected_lead_time_demand = daily_mean * lead_time_days

    safety_stock = (
        z_score
        * daily_std
        * lead_time_days**0.5
    )

    reorder_point = expected_lead_time_demand + safety_stock

    return {
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "z_score": z_score,
        "average_daily_demand": daily_mean,
        "daily_demand_std": daily_std,
        "expected_lead_time_demand": expected_lead_time_demand,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
    }
