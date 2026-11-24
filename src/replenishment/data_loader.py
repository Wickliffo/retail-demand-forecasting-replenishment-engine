from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "sales"}


class DataValidationError(ValueError):
    """Raised when historical demand data fails validation."""


def load_daily_sales(
    path: str | Path,
    christmas_missing_sales_policy: str = "christmas-zero",
) -> pd.Series:
    """
    Load and validate daily retail demand.

    Explicit missing-sales policy:
    - Missing December 25 observations may be treated as zero.
    - This is a documented working assumption, not a verified business fact.
    - Missing sales on any other date are rejected.
    - Original missing-data flags must agree with actual missing values.
    """

    if christmas_missing_sales_policy not in {
        "christmas-zero",
        "reject",
    }:
        raise ValueError(
            "Policy must be 'christmas-zero' or 'reject'."
        )

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["sales"] = pd.to_numeric(
        frame["sales"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise DataValidationError(
            "Invalid or missing date values found."
        )

    if frame["date"].duplicated().any():
        raise DataValidationError("Duplicate dates found.")

    # Verify the original missing-data indicator where available.
    if "is_missing_sales" in frame.columns:
        flags = (
            frame["is_missing_sales"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

        if flags.isna().any():
            raise DataValidationError(
                "Missing-data flag contains invalid values."
            )

        actual_missing = frame["sales"].isna()

        if not actual_missing.equals(flags.astype(bool)):
            raise DataValidationError(
                "is_missing_sales does not match actual missing sales."
            )

    missing_mask = frame["sales"].isna()
    missing_dates = frame.loc[missing_mask, "date"]

    imputed_dates = []

    if missing_mask.any():

        unexpected_dates = missing_dates[
            ~(
                (missing_dates.dt.month == 12)
                & (missing_dates.dt.day == 25)
            )
        ]

        if len(unexpected_dates):
            examples = [
                str(date.date())
                for date in unexpected_dates.head(5)
            ]

            raise DataValidationError(
                "Missing sales found outside December 25. "
                f"Unexpected dates: {examples}"
            )

        if christmas_missing_sales_policy == "reject":
            raise DataValidationError(
                "Missing Christmas sales found and policy is reject."
            )

        imputed_dates = [
            str(date.date())
            for date in missing_dates
        ]

        frame.loc[missing_mask, "sales"] = 0.0

    if not np.isfinite(frame["sales"]).all():
        raise DataValidationError(
            "Infinite or non-numeric sales values found."
        )

    if (frame["sales"] < 0).any():
        raise DataValidationError(
            "Negative sales values found."
        )

    frame = frame.sort_values("date").set_index("date")

    expected_dates = pd.date_range(
        start=frame.index.min(),
        end=frame.index.max(),
        freq="D",
    )

    missing_calendar_dates = expected_dates.difference(
        frame.index
    )

    if len(missing_calendar_dates):
        examples = [
            str(date.date())
            for date in missing_calendar_dates[:5]
        ]

        raise DataValidationError(
            "Daily calendar contains missing dates: "
            f"{examples}"
        )

    series = frame["sales"].astype(float)
    series.name = "sales"

    # Retain preprocessing evidence for the pipeline report.
    series.attrs["data_quality"] = {
        "original_missing_sales_count": len(imputed_dates),
        "imputed_dates": imputed_dates,
        "imputation_policy": christmas_missing_sales_policy,
        "imputation_value": 0.0,
        "assumption": (
            "Missing December 25 sales treated as zero. "
            "This is a working assumption, not a verified business fact."
        ),
    }

    return series
