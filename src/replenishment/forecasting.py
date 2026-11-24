"""
Reusable SARIMA forecasting engine.

Model configuration is inherited from Stage 2 research:

    SARIMA(1, 1, 1)(0, 1, 1, 7)

The model is fitted using statsmodels SARIMAX without exogenous variables.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_NAME = "SARIMA_111_011_7"

ORDER = (1, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 7)

FORECAST_HORIZON = 30

FOLD_STARTS = [
    "2017-02-17",
    "2017-03-19",
    "2017-04-18",
    "2017-05-18",
    "2017-06-17",
]


@dataclass(frozen=True)
class SARIMAConfig:
    """Configuration for the validated SARIMA model."""

    order: tuple = ORDER
    seasonal_order: tuple = SEASONAL_ORDER
    trend: str | None = None
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    max_iterations: int = 200


@dataclass
class ModelFit:
    """Fitted model and associated diagnostics."""

    fitted_model: object
    converged: bool
    fit_warnings: list[str]
    model_name: str
    training_observations: int
    training_start: str
    training_end: str
    fit_iterations: int | None

    def diagnostics(self) -> dict:
        return {
            "model": self.model_name,
            "converged": self.converged,
            "warnings": self.fit_warnings,
            "training_observations": self.training_observations,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "fit_iterations": self.fit_iterations,
        }


def validate_demand_series(demand: pd.Series) -> None:
    """Validate the time series before model fitting."""

    if not isinstance(demand, pd.Series):
        raise TypeError("Demand must be a pandas Series.")

    if demand.empty:
        raise ValueError("Demand series cannot be empty.")

    if not isinstance(demand.index, pd.DatetimeIndex):
        raise TypeError("Demand index must be a DatetimeIndex.")

    if demand.index.has_duplicates:
        raise ValueError("Demand index contains duplicate dates.")

    if not demand.index.is_monotonic_increasing:
        raise ValueError("Demand dates must be chronological.")

    if demand.isna().any():
        raise ValueError("Demand contains missing observations.")

    values = pd.to_numeric(demand, errors="coerce")

    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError("Demand contains non-numeric or infinite values.")

    if (values < 0).any():
        raise ValueError("Demand cannot contain negative values.")

    if len(demand) < 100:
        raise ValueError(
            "At least 100 daily observations are required "
            "for this seasonal model."
        )

    expected_index = pd.date_range(
        start=demand.index.min(),
        end=demand.index.max(),
        freq="D",
    )

    if not demand.index.equals(expected_index):
        raise ValueError(
            "Demand must have a continuous daily calendar."
        )


def fit_sarima(
    demand: pd.Series,
    config: SARIMAConfig | None = None,
    require_convergence: bool = True,
) -> ModelFit:
    """
    Fit the validated SARIMA model.

    The default configuration reproduces the Stage 2 model.
    """

    validate_demand_series(demand)

    config = config or SARIMAConfig()

    model = SARIMAX(
        demand.astype(float),
        order=config.order,
        seasonal_order=config.seasonal_order,
        trend=config.trend,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        fitted_model = model.fit(
            disp=False,
            maxiter=config.max_iterations,
        )

    warning_messages = [
        str(item.message)
        for item in captured
    ]

    mle_results = getattr(
        fitted_model,
        "mle_retvals",
        {},
    )

    converged = bool(
        mle_results.get("converged", False)
    )

    iterations = mle_results.get("iterations")

    if require_convergence and not converged:
        raise RuntimeError(
            "SARIMA model did not converge. "
            f"Optimizer information: {mle_results}. "
            f"Warnings: {warning_messages}"
        )

    return ModelFit(
        fitted_model=fitted_model,
        converged=converged,
        fit_warnings=warning_messages,
        model_name=MODEL_NAME,
        training_observations=len(demand),
        training_start=str(demand.index.min().date()),
        training_end=str(demand.index.max().date()),
        fit_iterations=(
            int(iterations)
            if iterations is not None
            else None
        ),
    )


def forecast_sarima(
    fitted: ModelFit,
    steps: int = FORECAST_HORIZON,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate point forecasts and model-based prediction intervals.

    Negative forecast bounds are clipped to zero because demand
    cannot be negative. The point forecast is also clipped to zero.

    The intervals are statsmodels model-based intervals and have not
    yet been empirically calibrated using out-of-fold errors.
    """

    if steps < 1:
        raise ValueError("Forecast steps must be at least one.")

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one.")

    forecast_result = fitted.fitted_model.get_forecast(
        steps=steps
    )

    raw_mean = np.asarray(
        forecast_result.predicted_mean,
        dtype=float,
    )

    confidence = forecast_result.conf_int(alpha=alpha)

    raw_lower = confidence.iloc[:, 0].to_numpy(dtype=float)
    raw_upper = confidence.iloc[:, 1].to_numpy(dtype=float)

    last_date = pd.Timestamp(
        fitted.training_end
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )

    forecast = pd.DataFrame(
        {
            "forecast": np.maximum(raw_mean, 0.0),
            "lower_bound": np.maximum(raw_lower, 0.0),
            "upper_bound": np.maximum(raw_upper, 0.0),
        },
        index=future_dates,
    )

    forecast.index.name = "date"

    forecast["forecast"] = forecast["forecast"].astype(float)

    forecast.attrs["model"] = fitted.model_name
    forecast.attrs["confidence_level"] = 1 - alpha
    forecast.attrs["interval_method"] = (
        "Statsmodels model-based prediction interval"
    )
    forecast.attrs["negative_values_clipped"] = True
    forecast.attrs["interval_calibration"] = (
        "Not empirically calibrated"
    )

    return forecast


def build_validation_folds(
    demand: pd.Series,
    horizon: int = FORECAST_HORIZON,
) -> list[dict]:
    """Reproduce the five chronological Stage 2 validation folds."""

    validate_demand_series(demand)

    if horizon != FORECAST_HORIZON:
        raise ValueError(
            "The original Stage 2 validation horizon is 30 days. "
            "Use 30 to reproduce the original experiment."
        )

    folds = []

    for fold_number, start_string in enumerate(
        FOLD_STARTS,
        start=1,
    ):
        validation_start = pd.Timestamp(start_string)

        validation_end = (
            validation_start
            + pd.Timedelta(days=horizon - 1)
        )

        training = demand.loc[
            demand.index < validation_start
        ]

        validation = demand.loc[
            validation_start:validation_end
        ]

        expected_dates = pd.date_range(
            start=validation_start,
            end=validation_end,
            freq="D",
        )

        if len(validation) != horizon:
            raise ValueError(
                f"Fold {fold_number}: expected {horizon} "
                f"validation observations, received {len(validation)}."
            )

        if not validation.index.equals(expected_dates):
            raise ValueError(
                f"Fold {fold_number}: validation calendar mismatch."
            )

        if training.empty:
            raise ValueError(
                f"Fold {fold_number}: training series is empty."
            )

        folds.append(
            {
                "fold": fold_number,
                "training": training,
                "validation": validation,
                "validation_start": validation_start,
                "validation_end": validation_end,
            }
        )

    return folds


def evaluate_sarima_cv(
    demand: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-run the five expanding-window validation folds.

    Returns:
        fold_metrics: one row per fold
        out_of_fold_forecasts: all validation predictions
    """

    folds = build_validation_folds(demand)

    metrics = []
    forecasts = []

    for fold in folds:
        fitted = fit_sarima(fold["training"])

        prediction = forecast_sarima(
            fitted,
            steps=len(fold["validation"]),
        )

        actual = fold["validation"].to_numpy(dtype=float)
        predicted = prediction["forecast"].to_numpy(dtype=float)

        errors = predicted - actual

        fold_mae = float(np.mean(np.abs(errors)))
        fold_rmse = float(np.sqrt(np.mean(errors ** 2)))

        metrics.append(
            {
                "fold": fold["fold"],
                "model": MODEL_NAME,
                "training_observations": len(fold["training"]),
                "validation_observations": len(fold["validation"]),
                "validation_start": str(
                    fold["validation_start"].date()
                ),
                "validation_end": str(
                    fold["validation_end"].date()
                ),
                "mae": fold_mae,
                "rmse": fold_rmse,
                "mean_error": float(np.mean(errors)),
                "converged": fitted.converged,
                "warning_count": len(fitted.fit_warnings),
            }
        )

        fold_output = prediction.copy()
        fold_output["actual"] = actual
        fold_output["error"] = errors
        fold_output["fold"] = fold["fold"]

        forecasts.append(fold_output)

    fold_metrics = pd.DataFrame(metrics)

    out_of_fold = pd.concat(
        forecasts,
        axis=0,
    )

    out_of_fold.index.name = "date"

    return fold_metrics, out_of_fold


def summarize_cv_metrics(
    fold_metrics: pd.DataFrame,
    out_of_fold: pd.DataFrame,
) -> dict:
    """Report fold-average and pooled validation metrics separately."""

    if fold_metrics.empty or out_of_fold.empty:
        raise ValueError("Validation results cannot be empty.")

    errors = out_of_fold["error"].to_numpy(dtype=float)

    return {
        "model": MODEL_NAME,
        "fold_count": len(fold_metrics),
        "validation_observations": len(out_of_fold),
        "mean_fold_mae": float(fold_metrics["mae"].mean()),
        "mean_fold_rmse": float(fold_metrics["rmse"].mean()),
        "pooled_mae": float(np.mean(np.abs(errors))),
        "pooled_rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
