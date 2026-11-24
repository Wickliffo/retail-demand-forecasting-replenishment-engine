# M8 — Seasonal Naive Forecast Error Analysis

## 1. Objective

This milestone investigates forecast errors produced by the Seasonal Naive benchmark. The objective is to understand error magnitude, direction, distribution and weekday variation rather than relying exclusively on aggregate performance metrics.

## 2. Forecast and Evaluation Period

- **Model:** Seasonal Naive
- **Seasonal period:** 7 days
- **Forecast horizon:** 30 days
- **Evaluation period:** 17 July 2017 to 15 August 2017

The analysis uses the forecast and actual values from the previously evaluated final holdout.

## 3. Error Definition

Forecast error is defined as:

\[
e_t = y_t-\hat{y}_t
\]

Positive errors indicate underforecasting, while negative errors indicate overforecasting.

## 4. Error Statistics

| Metric | Result |
|---|---:|
| Mean error | -171.1333 |
| Median error | -124.5 |
| Error standard deviation | 446.7529 |
| Maximum underforecast error | 658 |
| Maximum overforecast error | -1912 |

The negative mean and median errors indicate that the forecasts exceeded actual demand on average during this evaluation period.

The error distribution was not symmetrical. The largest overforecast was substantially larger in magnitude than the largest underforecast.

## 5. Largest Errors

### Largest underforecast

- **Date:** 18 July 2017
- **Actual:** 3,370
- **Forecast:** 2,712
- **Error:** +658

### Largest overforecast

- **Date:** 11 August 2017
- **Actual:** 1,270
- **Forecast:** 3,182
- **Error:** -1,912

The largest overforecast represents a substantial deviation between forecast and actual demand and contributes materially to the error distribution.

## 6. Weekday Analysis

| Weekday | Actual Mean | Forecast Mean | Mean Error | Mean Absolute Error |
|---|---:|---:|---:|---:|
| Monday | 2651.20 | 2849.00 | -197.80 | 244.60 |
| Tuesday | 2843.40 | 2712.00 | 131.40 | 335.80 |
| Wednesday | 2977.25 | 3185.00 | -207.75 | 238.75 |
| Thursday | 2456.50 | 2571.00 | -114.50 | 135.00 |
| Friday | 2348.75 | 3182.00 | -833.25 | 833.25 |
| Saturday | 2171.50 | 2297.00 | -125.50 | 339.50 |
| Sunday | 1108.50 | 1028.00 | 80.50 | 120.00 |

Friday recorded the highest mean absolute error in this evaluation period.

The observed weekday variation suggests that forecast accuracy was not uniform across the week.

## 7. Aggregate Demand Comparison

| Metric | Value |
|---|---:|
| Total actual demand | 71,723 |
| Total forecast demand | 76,857 |
| Total demand deviation | 5,134 |
| Aggregate forecast deviation | 7.1581% |

Forecast demand exceeded actual demand by 5,134 units across the evaluation period.

## 8. Visualizations and Artifacts

The following outputs were generated:

- `m8_error_series.png`
- `m8_error_distribution.png`
- `m8_weekday_absolute_error.png`
- `m8_seasonal_error_analysis.csv`
- `m8_weekday_error_summary.csv`

These artifacts are stored in the reports directory.

## 9. Interpretation

The Seasonal Naive model improved substantially over the ordinary Naive benchmark in M6. However, the M8 analysis reveals that aggregate accuracy does not fully describe forecasting behavior.

The model experienced uneven errors across weekdays, with Friday displaying particularly high mean absolute error during the evaluated period.

The unusually large overforecast on 11 August 2017 further demonstrates the importance of examining individual errors and their distribution.

These observations are descriptive and should not be interpreted as evidence of persistent weekday effects or future demand behavior.

## 10. Limitations

- The analysis uses a single 30-day evaluation period.
- The final holdout has already been exposed through prior benchmark analysis.
- Weekday error estimates are based on a small number of observations.
- The results do not establish causal explanations for the observed errors.
- The analysis does not evaluate inventory costs or stockout consequences.

## 11. Conclusion

The Seasonal Naive model captures the weekly demand pattern more effectively than the ordinary Naive benchmark, but it remains susceptible to large and uneven forecast errors.

The analysis establishes the need for additional forecasting models, chronological validation, uncertainty quantification and inventory-oriented evaluation.

The next modeling phases will investigate statistical models with trend, seasonal and exogenous structures.
