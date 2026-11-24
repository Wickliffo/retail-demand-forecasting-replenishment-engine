# Project Status

## Project

Demand Forecasting and Replenishment Engine

## Current Status

- Stage 1: Data engineering and demand characterization — completed.
- Stage 2: Forecasting and model validation — completed.
- Stage 3: Exogenous feature analysis — completed.
- Stage 4: Inventory policy and constrained replenishment — pending.
- Stage 5: Inventory backtesting and decision evaluation — pending.
- Stage 6: Engine packaging, tests, and final release — pending.

## Current Dataset

The analytical series is Store 1, GROCERY I from the
Corporacion Favorita grocery sales forecasting dataset.

The dataset is Ecuadorian retail data. This project does
not claim the dataset represents Kenya.

## Stage 1

Data preparation, calendar reconstruction, missing-date
analysis, sales characterization, and exploratory analysis.

## Stage 2

Forecasting model selection and time-series validation.

The selected baseline:

SARIMA(1,1,1)(0,1,1,7)

Validation used five expanding-window folds with a
30-day forecast horizon.

## Stage 3

Exogenous feature analysis using a fixed SARIMA champion.

Seven configurations were evaluated across five folds.

The leading configuration was M3_EVENTS.

### Results

Baseline mean MAE: 318.607551

Event model mean MAE: 303.676164

Baseline mean RMSE: 513.501623

Event model mean RMSE: 423.588790

MAE improvement: approximately 4.69%

RMSE improvement: approximately 17.51%

Fold-level MAE results: four wins and one loss.

All 35 experimental model fits completed successfully.

## Stage 3 Limitations

- One store-product series.
- Five validation folds.
- Thirty-day validation horizons.
- Retrospective promotion features.
- Event-feature improvement is not universal.
- Results do not establish causality.
- Inventory outcomes have not yet been evaluated.

## Stage 4

Planned work:

1. Demand uncertainty.
2. Lead-time demand.
3. Inventory position.
4. Safety stock.
5. Reorder-point calculations.
6. Service-level policies.
7. Constrained order optimization.
8. Order quantities and replenishment constraints.

## Stage 5

Planned inventory simulation and backtesting:

- Inventory movement.
- Stockouts.
- Holding costs.
- Replenishment costs.
- Service levels.
- Order decisions.
- Policy comparisons.

## Stage 6

Planned finalization:

- Reusable Python modules.
- Automated tests.
- Reproducibility.
- Documentation.
- Final repository release.
