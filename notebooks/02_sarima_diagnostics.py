"""
M12: SARIMA residual diagnostics and model adequacy.

Champion:
    SARIMA(1,1,1)(0,1,1,7)

Purpose:
    - Reproduce the five chronological validation folds.
    - Fit the champion SARIMA on each fold's training data.
    - Examine in-sample residuals.
    - Examine out-of-fold forecast errors.
    - Run Ljung-Box autocorrelation diagnostics.
    - Export diagnostic tables, plots and a Markdown report.

Data assumption:
    Missing calendar sales observations are treated as zero.
    This is an experimental assumption, not verified business truth.
"""

from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew, kurtosis


# ============================================================
# M12.1 PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "store1_grocery1_daily_prepared.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

LJUNG_BOX_LAGS = [7, 14, 21, 28]

# Conservative diagnostic configuration.
# These are not model-selection rules.
MIN_RESIDUALS_FOR_DIAGNOSTICS = 100


# ============================================================
# M12.2 DATA LOADING AND VALIDATION
# ============================================================

def load_prepared_series(path: Path) -> pd.Series:

    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print("\nLoaded dataset:")
    print(df.head())

    print("\nDataset columns:")
    print(df.columns.tolist())

    if "date" not in df.columns:
        raise ValueError(
            "The prepared CSV must contain a 'date' column."
        )

    if "sales" not in df.columns:
        raise ValueError(
            "The prepared CSV must contain a 'sales' column."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    df = df.sort_values("date")

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected in prepared data."
        )

    series = (
        df.set_index("date")["sales"]
        .astype(float)
        .sort_index()
    )

    full_calendar = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="D"
    )

    series = series.reindex(full_calendar)

    series.index.name = "date"

    # Explicit assumption for this experiment.
    series = series.fillna(0.0)

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(
            "Non-finite values found after preparation."
        )

    if (series < 0).any():
        raise ValueError(
            "Negative sales values detected."
        )

    print("\nSeries information:")
    print(f"Observations: {len(series)}")
    print(f"Start date: {series.index.min()}")
    print(f"End date: {series.index.max()}")
    print(f"Frequency: {series.index.freq}")

    return series


# ============================================================
# M12.3 VALIDATION FOLD CONSTRUCTION
# ============================================================

def build_folds(series: pd.Series):

    folds = []

    for fold_number, start_string in enumerate(
        FOLD_STARTS,
        start=1
    ):

        validation_start = pd.Timestamp(start_string)

        validation_end = (
            validation_start
            + pd.Timedelta(days=FORECAST_HORIZON - 1)
        )

        training = series.loc[
            series.index < validation_start
        ]

        validation = series.loc[
            validation_start:validation_end
        ]

        expected_validation_dates = pd.date_range(
            start=validation_start,
            end=validation_end,
            freq="D"
        )

        if len(validation) != FORECAST_HORIZON:

            raise ValueError(
                f"Fold {fold_number}: "
                f"Expected {FORECAST_HORIZON} "
                f"validation observations but received "
                f"{len(validation)}."
            )

        if not validation.index.equals(
            expected_validation_dates
        ):

            raise ValueError(
                f"Fold {fold_number}: "
                "Validation calendar is inconsistent."
            )

        if training.empty:

            raise ValueError(
                f"Fold {fold_number}: "
                "Training series is empty."
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


# ============================================================
# M12.4 RESIDUAL CLEANING
# ============================================================

def clean_residuals(residuals):

    residuals = pd.Series(residuals)

    residuals = residuals.replace(
        [np.inf, -np.inf],
        np.nan
    )

    residuals = residuals.dropna()

    return residuals


# ============================================================
# M12.5 RESIDUAL SUMMARY STATISTICS
# ============================================================

def calculate_residual_statistics(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    if len(residuals) == 0:

        raise ValueError(
            "No valid residual observations."
        )

    statistics = {

        "fold": fold_number,

        "residual_type": residual_type,

        "count": len(residuals),

        "mean": residuals.mean(),

        "median": residuals.median(),

        "std": residuals.std(),

        "variance": residuals.var(),

        "min": residuals.min(),

        "max": residuals.max(),

        "mean_absolute_residual": (
            residuals.abs().mean()
        ),

        "skewness": skew(
            residuals,
            bias=False
        ),

        "kurtosis_excess": kurtosis(
            residuals,
            fisher=True,
            bias=False
        ),

    }

    return statistics


# ============================================================
# M12.6 LJUNG-BOX TEST
# ============================================================

def run_ljung_box(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    if len(residuals) < (
        MIN_RESIDUALS_FOR_DIAGNOSTICS
    ):

        return pd.DataFrame()

    applicable_lags = [
        lag
        for lag in LJUNG_BOX_LAGS
        if lag < len(residuals)
    ]

    if not applicable_lags:

        return pd.DataFrame()

    result = acorr_ljungbox(
        residuals,
        lags=applicable_lags,
        return_df=True
    )

    result = result.reset_index()

    result = result.rename(
        columns={
            "index": "lag",
            "lb_stat": "ljung_box_statistic",
            "lb_pvalue": "p_value",
        }
    )

    result["fold"] = fold_number

    result["residual_type"] = residual_type

    return result


# ============================================================
# M12.7 RESIDUAL PLOTS
# ============================================================

def save_residual_time_plot(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        residuals.index,
        residuals,
        linewidth=0.8
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"Fold {fold_number} "
        f"{residual_type} Residuals"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Residual")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    filename = (
        f"m12_fold{fold_number}_"
        f"{residual_type.lower()}_residuals.png"
    )

    fig.savefig(
        REPORTS_DIR / filename,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# M12.8 RESIDUAL DISTRIBUTION PLOT
# ============================================================

def save_residual_distribution(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        residuals,
        bins=30
    )

    ax.axvline(
        residuals.mean(),
        linestyle="--",
        linewidth=1,
        label="Residual mean"
    )

    ax.axvline(
        0,
        linestyle=":",
        linewidth=1,
        label="Zero"
    )

    ax.set_title(
        f"Fold {fold_number} "
        f"{residual_type} Residual Distribution"
    )

    ax.set_xlabel("Residual")

    ax.set_ylabel("Frequency")

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    filename = (
        f"m12_fold{fold_number}_"
        f"{residual_type.lower()}_distribution.png"
    )

    fig.savefig(
        REPORTS_DIR / filename,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# M12.9 RESIDUAL ACF PLOT
# ============================================================

def save_residual_acf(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    from statsmodels.tsa.stattools import acf

    if len(residuals) < 2:
        raise ValueError(
            "At least two residual observations "
            "are required for the ACF."
        )

    max_lags = min(
        35,
        len(residuals) - 1
    )

    autocorrelation = acf(
        residuals,
        nlags=max_lags,
        fft=True
    )

    lags = np.arange(
        len(autocorrelation)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        lags,
        autocorrelation,
        width=0.7
    )

    # Approximate reference limits.
    confidence_limit = (
        1.96 / np.sqrt(len(residuals))
    )

    ax.axhline(
        confidence_limit,
        linestyle="--",
        linewidth=1,
        label="Approx. 95% reference limit"
    )

    ax.axhline(
        -confidence_limit,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        f"Fold {fold_number} "
        f"{residual_type} Residual ACF"
    )

    ax.set_xlabel("Lag")

    ax.set_ylabel("Autocorrelation")

    ax.set_xticks(
        lags
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        f"m12_fold{fold_number}_"
        f"{residual_type.lower()}_acf.png"
    )

    fig.savefig(
        REPORTS_DIR / filename,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# M12.10 RESIDUAL SQUARED SERIES
# ============================================================

def save_squared_residual_plot(
    residuals,
    fold_number,
    residual_type
):

    residuals = clean_residuals(residuals)

    squared_residuals = residuals ** 2

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        squared_residuals.index,
        squared_residuals,
        linewidth=0.8
    )

    ax.set_title(
        f"Fold {fold_number} "
        f"Squared {residual_type} Residuals"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Squared residual")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    filename = (
        f"m12_fold{fold_number}_"
        f"{residual_type.lower()}_squared.png"
    )

    fig.savefig(
        REPORTS_DIR / filename,
        dpi=150
    )

    plt.close(fig)


# ============================================================
# M12.11 FIT MODEL AND COLLECT DIAGNOSTICS
# ============================================================

def run_diagnostics(series):

    folds = build_folds(series)

    residual_statistics = []

    ljung_box_results = []

    fold_forecasts = []

    fold_fit_information = []

    all_validation_errors = []

    for fold_data in folds:

        fold_number = fold_data["fold"]

        training = fold_data["training"]

        validation = fold_data["validation"]

        print("\n" + "=" * 70)

        print(
            f"FOLD {fold_number}: "
            f"{MODEL_NAME}"
        )

        print("=" * 70)

        print(
            f"Training observations: {len(training)}"
        )

        print(
            f"Training end: {training.index[-1]}"
        )

        print(
            f"Validation start: {validation.index[0]}"
        )

        print(
            f"Validation end: {validation.index[-1]}"
        )

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        start_time = time.perf_counter()

        model = SARIMAX(
            training,

            order=ORDER,

            seasonal_order=SEASONAL_ORDER,

            trend=None,

            enforce_stationarity=False,

            enforce_invertibility=False,

        )

        with warnings.catch_warnings(
            record=True
        ) as captured_warnings:

            warnings.simplefilter("always")

            fitted_model = model.fit(
                disp=False,
                maxiter=200
            )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        convergence = (
            fitted_model.mle_retvals.get(
                "converged",
                False
            )
        )

        warning_messages = [
            str(w.message)
            for w in captured_warnings
        ]

        print(
            f"\nConverged: {convergence}"
        )

        print(
            f"Fit time: {elapsed_seconds:.2f}s"
        )

        print(
            f"Warnings: {len(warning_messages)}"
        )

        # ----------------------------------------------------
        # IN-SAMPLE RESIDUALS
        # ----------------------------------------------------

        in_sample_residuals = clean_residuals(
            fitted_model.resid
        )

        in_sample_residuals.index = (
            training.index[
                :len(fitted_model.resid)
            ]
        )

        # Statsmodels may include residuals affected
        # by initialization at the beginning.
        # These are retained in this experiment.
        # Interpret the first residuals cautiously.

        # ----------------------------------------------------
        # VALIDATION FORECAST
        # ----------------------------------------------------

        forecast_result = (
            fitted_model.get_forecast(
                steps=len(validation)
            )
        )

        predicted_mean = (
            forecast_result.predicted_mean
        )

        predicted_mean = pd.Series(
            predicted_mean,
            index=validation.index,
            name="forecast"
        )

        forecast_errors = (
            predicted_mean - validation
        )

        all_validation_errors.append(
            forecast_errors
        )

        # ----------------------------------------------------
        # IN-SAMPLE STATISTICS
        # ----------------------------------------------------

        in_sample_stats = (
            calculate_residual_statistics(
                in_sample_residuals,
                fold_number,
                "in_sample"
            )
        )

        residual_statistics.append(
            in_sample_stats
        )

        # ----------------------------------------------------
        # VALIDATION ERROR STATISTICS
        # ----------------------------------------------------

        validation_stats = (
            calculate_residual_statistics(
                forecast_errors,
                fold_number,
                "validation"
            )
        )

        residual_statistics.append(
            validation_stats
        )

        # ----------------------------------------------------
        # LJUNG-BOX: IN-SAMPLE
        # ----------------------------------------------------

        lb_in_sample = run_ljung_box(
            in_sample_residuals,
            fold_number,
            "in_sample"
        )

        if not lb_in_sample.empty:

            ljung_box_results.append(
                lb_in_sample
            )

        # ----------------------------------------------------
        # LJUNG-BOX: VALIDATION ERRORS
        # ----------------------------------------------------

        lb_validation = run_ljung_box(
            forecast_errors,
            fold_number,
            "validation"
        )

        if not lb_validation.empty:

            ljung_box_results.append(
                lb_validation
            )

        # ----------------------------------------------------
        # PLOTS
        # ----------------------------------------------------

        save_residual_time_plot(
            in_sample_residuals,
            fold_number,
            "in_sample"
        )

        save_residual_distribution(
            in_sample_residuals,
            fold_number,
            "in_sample"
        )

        save_residual_acf(
            in_sample_residuals,
            fold_number,
            "in_sample"
        )

        save_squared_residual_plot(
            in_sample_residuals,
            fold_number,
            "in_sample"
        )

        save_residual_time_plot(
            forecast_errors,
            fold_number,
            "validation"
        )

        save_residual_distribution(
            forecast_errors,
            fold_number,
            "validation"
        )

        save_residual_acf(
            forecast_errors,
            fold_number,
            "validation"
        )

        save_squared_residual_plot(
            forecast_errors,
            fold_number,
            "validation"
        )

        # ----------------------------------------------------
        # SAVE FORECAST RECORDS
        # ----------------------------------------------------

        fold_forecast_df = pd.DataFrame({

            "date": validation.index,

            "fold": fold_number,

            "actual": validation.values,

            "forecast": predicted_mean.values,

            "error_forecast_minus_actual": (
                forecast_errors.values
            ),

            "absolute_error": (
                forecast_errors.abs().values
            ),

            "squared_error": (
                forecast_errors.pow(2).values
            ),

        })

        fold_forecasts.append(
            fold_forecast_df
        )

        # ----------------------------------------------------
        # FIT INFORMATION
        # ----------------------------------------------------

        fold_fit_information.append({

            "fold": fold_number,

            "model": MODEL_NAME,

            "training_observations": (
                len(training)
            ),

            "training_start": (
                training.index.min()
            ),

            "training_end": (
                training.index.max()
            ),

            "validation_start": (
                validation.index.min()
            ),

            "validation_end": (
                validation.index.max()
            ),

            "converged": convergence,

            "fit_seconds": elapsed_seconds,

            "warning_count": (
                len(warning_messages)
            ),

            "warnings": " | ".join(
                warning_messages
            ),

        })

        # ----------------------------------------------------
        # PRINT FOLD RESULTS
        # ----------------------------------------------------

        print("\nValidation metrics:")

        print(
            f"MAE: "
            f"{forecast_errors.abs().mean():.4f}"
        )

        print(
            f"RMSE: "
            f"{np.sqrt(
                forecast_errors.pow(2).mean()
            ):.4f}"
        )

        print(
            f"Mean error: "
            f"{forecast_errors.mean():.4f}"
        )

        print(
            "\nIn-sample residual mean:"
        )

        print(
            f"{in_sample_residuals.mean():.4f}"
        )

        print(
            "\nIn-sample residual std:"
        )

        print(
            f"{in_sample_residuals.std():.4f}"
        )

    # ========================================================
    # M12.12 COMBINE RESULTS
    # ========================================================

    residual_stats_df = pd.DataFrame(
        residual_statistics
    )

    ljung_box_df = (
        pd.concat(
            ljung_box_results,
            ignore_index=True
        )
        if ljung_box_results
        else pd.DataFrame()
    )

    forecast_df = pd.concat(
        fold_forecasts,
        ignore_index=True
    )

    fit_info_df = pd.DataFrame(
        fold_fit_information
    )

    # ========================================================
    # M12.13 OVERALL OOF ERROR
    # ========================================================

    combined_validation_errors = pd.concat(
        all_validation_errors
    )

    overall_oof_metrics = {

        "model": MODEL_NAME,

        "observations": len(
            combined_validation_errors
        ),

        "MAE": (
            combined_validation_errors
            .abs()
            .mean()
        ),

        "RMSE": np.sqrt(
            combined_validation_errors
            .pow(2)
            .mean()
        ),

        "mean_error": (
            combined_validation_errors.mean()
        ),

        "median_error": (
            combined_validation_errors.median()
        ),

        "std_error": (
            combined_validation_errors.std()
        ),

    }

    overall_oof_df = pd.DataFrame(
        [overall_oof_metrics]
    )

    return (
        residual_stats_df,
        ljung_box_df,
        forecast_df,
        fit_info_df,
        overall_oof_df
    )


# ============================================================
# M12.14 SAVE OUTPUTS
# ============================================================

def save_outputs(
    residual_stats_df,
    ljung_box_df,
    forecast_df,
    fit_info_df,
    overall_oof_df
):

    residual_stats_df.to_csv(
        REPORTS_DIR
        / "m12_residual_statistics.csv",
        index=False
    )

    if not ljung_box_df.empty:

        ljung_box_df.to_csv(
            REPORTS_DIR
            / "m12_ljung_box_results.csv",
            index=False
        )

    forecast_df.to_csv(
        REPORTS_DIR
        / "m12_validation_residuals.csv",
        index=False
    )

    fit_info_df.to_csv(
        REPORTS_DIR
        / "m12_fold_fit_information.csv",
        index=False
    )

    overall_oof_df.to_csv(
        REPORTS_DIR
        / "m12_overall_oof_metrics.csv",
        index=False
    )

    print("\nSaved diagnostic CSV files.")


# ============================================================
# M12.15 REPORT GENERATION
# ============================================================

def generate_report(
    residual_stats_df,
    ljung_box_df,
    fit_info_df,
    overall_oof_df
):

    lines = []

    lines.append(
        "# M12: SARIMA Residual Diagnostics"
    )

    lines.append("")

    lines.append(
        f"Model: {MODEL_NAME}"
    )

    lines.append("")

    lines.append(
        "Order: SARIMA(1,1,1)(0,1,1,7)"
    )

    lines.append("")

    lines.append(
        "Missing calendar sales observations "
        "were treated as zero."
    )

    lines.append("")

    lines.append(
        "This is an experimental assumption "
        "requiring business verification."
    )

    lines.append("")

    lines.append(
        "## Overall out-of-fold metrics"
    )

    lines.append("")

    lines.append(
        overall_oof_df.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "## Residual statistics"
    )

    lines.append("")

    lines.append(
        residual_stats_df.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "## Ljung-Box results"
    )

    lines.append("")

    if ljung_box_df.empty:

        lines.append(
            "No Ljung-Box results generated."
        )

    else:

        lines.append(
            ljung_box_df.to_string(
                index=False
            )
        )

    lines.append("")

    lines.append(
        "## Model fitting information"
    )

    lines.append("")

    lines.append(
        fit_info_df.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "## Interpretation guidance"
    )

    lines.append("")

    lines.append(
        "- Ljung-Box p-values below 0.05 "
        "provide evidence against the null of "
        "no autocorrelation up to the tested lag."
    )

    lines.append(
        "- Ljung-Box p-values above 0.05 do "
        "not prove that residuals are independent."
    )

    lines.append(
        "- Residual normality is not a strict "
        "requirement for accurate point forecasting."
    )

    lines.append(
        "- Residual variance patterns should "
        "be examined visually."
    )

    lines.append(
        "- Validation forecast errors are "
        "different from in-sample fitted residuals."
    )

    lines.append(
        "- Diagnostic results must be considered "
        "across folds, not from one fold alone."
    )

    lines.append(
        "- This model is not yet an inventory "
        "optimization model."
    )

    lines.append("")

    report_path = (
        REPORTS_DIR
        / "m12_sarima_diagnostics.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"\nReport saved: {report_path}"
    )


# ============================================================
# M12.16 MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "M12: SARIMA RESIDUAL DIAGNOSTICS"
    )

    print("=" * 70)

    print(f"Model: {MODEL_NAME}")

    print(f"Order: {ORDER}")

    print(
        f"Seasonal order: {SEASONAL_ORDER}"
    )

    print(
        f"Reports directory: {REPORTS_DIR}"
    )

    series = load_prepared_series(
        DATA_PATH
    )

    (
        residual_stats_df,
        ljung_box_df,
        forecast_df,
        fit_info_df,
        overall_oof_df
    ) = run_diagnostics(series)

    save_outputs(
        residual_stats_df,
        ljung_box_df,
        forecast_df,
        fit_info_df,
        overall_oof_df
    )

    generate_report(
        residual_stats_df,
        ljung_box_df,
        fit_info_df,
        overall_oof_df
    )

    print("\n" + "=" * 70)

    print(
        "M12 DIAGNOSTICS COMPLETED"
    )

    print("=" * 70)

    print(
        "\nOverall out-of-fold metrics:"
    )

    print(
        overall_oof_df.to_string(
            index=False
        )
    )

    print(
        "\nResidual statistics:"
    )

    print(
        residual_stats_df.to_string(
            index=False
        )
    )

    print(
        "\nLjung-Box results:"
    )

    if ljung_box_df.empty:

        print(
            "No Ljung-Box results available."
        )

    else:

        print(
            ljung_box_df.to_string(
                index=False
            )
        )

    print(
        "\nAll outputs saved in:"
    )

    print(REPORTS_DIR)


if __name__ == "__main__":

    main()