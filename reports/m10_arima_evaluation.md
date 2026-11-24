# M10 — ARIMA Candidate Evaluation

## Objective

Evaluate a controlled set of non-seasonal ARIMA models using five expanding chronological validation folds with a 30-day forecast horizon.

## Experimental Design

- Five chronological validation folds
- 30-day forecast horizon per fold
- Differencing order d = 1
- Nine ARIMA candidates
- Seasonal-naive baseline with a 7-day period
- Final holdout excluded from model selection

## Candidate Models

ARIMA(0,1,0), ARIMA(1,1,0), ARIMA(0,1,1), ARIMA(1,1,1), ARIMA(2,1,0), ARIMA(0,1,2), ARIMA(2,1,1), ARIMA(1,1,2), and ARIMA(2,1,2).

## Results

All 45 candidate fits were executed successfully across the five folds.

The ARIMA(0,1,1) model ranked first among non-seasonal ARIMA candidates, with:

- Mean MAE: 576.624
- Mean RMSE: 803.218
- Mean error: -40.353
- Successful folds: 5
- Converged folds: 5

ARIMA(2,1,2) ranked second on mean MAE, but one fold reported a convergence warning.

## Baseline Comparison

The seasonal-naive model, using the last seven training observations as its repeating forecast pattern, was evaluated across the same five chronological validation folds.

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| Seasonal Naive (7) | 531.353 | 774.362 |
| ARIMA(0,1,1) | 576.624 | 803.218 |

The seasonal-naive baseline outperformed all tested non-seasonal ARIMA candidates on mean MAE.

The best ARIMA candidate was approximately 8.52% worse than the seasonal-naive baseline on mean MAE.

## Interpretation

The results suggest that the weekly structure in the selected daily sales series is an important predictive component.

Non-seasonal ARIMA models with first differencing did not outperform the weekly seasonal-naive baseline. This does not invalidate ARIMA modelling, but suggests that explicit seasonal modelling should be investigated.

## Data Assumption

Missing Christmas observations were treated as zero-sales days for this modelling experiment to preserve the daily calendar. This is an assumption requiring verification against the underlying operational context.

## Decision

ARIMA(0,1,1) is retained as the best non-seasonal ARIMA candidate from the initial search, but it is not selected as the forecasting champion because seasonal naive performed better.

The next modelling phase will investigate weekly seasonal structure using seasonal ARIMA-type models.

## Generated Artifacts

- m10_arima_fold_metrics.csv
- m10_arima_forecasts.csv
- m10_arima_model_comparison.csv
- m10_arima_failures.csv
- m10_seasonal_naive_cv_metrics.csv
- m10_seasonal_naive_cv_summary.csv

## Next Milestone

M11 — Investigate and evaluate weekly seasonal ARIMA models using chronological validation.
