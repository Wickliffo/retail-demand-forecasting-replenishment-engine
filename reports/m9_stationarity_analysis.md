# M9 — Stationarity and Differencing Analysis

## Objective

Investigate the stationarity properties of the selected Store 1 — GROCERY I daily sales series to determine an appropriate differencing order for subsequent ARIMA-family forecasting models.

## Methodology

Stationarity diagnostics were performed using the Augmented Dickey-Fuller (ADF) and KPSS tests across five chronological validation folds.

Each fold's training series was evaluated at three levels:

- Original series (d = 0)
- First difference (d = 1)
- Second difference (d = 2)

The diagnostic process was restricted to the training portions of the chronological folds, keeping the final 30-day holdout separate.

## Results

### Original Series

Across all five folds:

- ADF p-values ranged from approximately 0.00067 to 0.00207.
- KPSS returned p-values of 0.01.

The tests produced conflicting evidence. ADF rejected the unit-root null hypothesis, while KPSS rejected the stationarity null hypothesis.

Consequently, the original series was not conclusively classified as stationary based solely on these tests.

### First Difference

Across all five folds:

- ADF consistently rejected the unit-root null hypothesis.
- KPSS consistently supported stationarity, with reported p-values of 0.10.

Both tests supported stationarity after first differencing.

### Second Difference

The second-differenced series also passed both stationarity diagnostics across the five folds. However, additional differencing was not justified merely because it produced stationary results.

## Decision

The preliminary differencing order selected for ARIMA candidate evaluation is:

**d = 1**

First differencing consistently achieved stationarity according to both diagnostics across the five validation folds. Second differencing was not selected because it may introduce unnecessary differencing and additional dynamics.

This remains a preliminary modelling decision subject to forecasting performance and residual diagnostics.

## Limitations

- Stationarity tests do not independently establish the best forecasting model.
- KPSS p-values of 0.01 and 0.10 reflect lookup-table bounds and should not be interpreted as exact probabilities.
- Stationarity does not guarantee good forecast accuracy.
- Model selection will be based on chronological validation rather than stationarity tests alone.

## Next Milestone

M10 — ARIMA candidate evaluation using chronological validation and the preliminary differencing order d = 1.
