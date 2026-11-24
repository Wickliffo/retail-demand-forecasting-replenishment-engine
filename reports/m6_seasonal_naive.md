# M6 — Seasonal Naive Forecast Benchmark

## 1. Objective

The Seasonal Naive model establishes a benchmark that incorporates weekly demand patterns. Unlike the ordinary Naive model, which repeats the final observed training value, Seasonal Naive forecasts demand by repeating observations from the same seasonal position in the previous cycle.

For this project, the seasonal period is seven days, representing a weekly demand cycle.

## 2. Dataset

- **Series:** Store 1 — GROCERY I
- **Frequency:** Daily
- **Seasonal period:** 7 days
- **Forecast horizon:** 30 days
- **Training period:** Data available before the final 30-day test period
- **Test period:** 17 July 2017 to 15 August 2017

The final test period was retained for benchmark evaluation.

## 3. Methodology

The Seasonal Naive model uses the most recent seven observed training values to generate forecasts.

The forecast repeats the weekly sequence:

\[
\hat{y}_{T+h}=y_{T+h-7}
\]

where:

- \(T\) is the end of the training data.
- \(h\) is the forecast horizon.
- \(7\) represents the weekly seasonal period.

The seven observed values are repeated across the 30-day forecast horizon.

No model fitting or parameter estimation is required.

## 4. Benchmark Results

| Metric | Value |
|---|---:|
| MAE | 318.933333 |
| RMSE | 471.404073 |
| Mean Error | -171.133333 |
| Test Mean | 2390.766667 |
| Relative MAE | 13.340212% |

The negative mean error indicates that the forecast, on average, exceeded actual demand during the evaluation period.

## 5. Comparison with the Naive Benchmark

| Metric | Naive | Seasonal Naive |
|---|---:|---:|
| MAE | 1368.033333 | 318.933333 |
| RMSE | 1511.514858 | 471.404073 |
| Mean Error | 1362.766667 | -171.133333 |

The Seasonal Naive model reduced MAE by approximately **76.69%** relative to the ordinary Naive benchmark.

This demonstrates the importance of incorporating weekly demand patterns when forecasting this series.

## 6. Visualizations

The following visualizations were generated:

- `m6_seasonal_forecast.png` — actual demand versus the 30-day Seasonal Naive forecast.
- `m6_seasonal_context.png` — historical context and forecast comparison.

These plots are stored in the `reports/` directory.

## 7. Limitations

- The model assumes that the most recent weekly pattern will repeat.
- It cannot explicitly model trends or changing seasonal behavior.
- It does not account for promotions, holidays, external events or other explanatory variables.
- Its performance over a single 30-day holdout does not establish general forecasting performance.
- The model can systematically overforecast or underforecast when demand patterns change.

## 8. Conclusion

The Seasonal Naive model substantially outperformed the ordinary Naive benchmark on the final 30-day holdout.

The result provides a stronger baseline for subsequent forecasting models. However, its performance must be evaluated across multiple chronological validation windows before conclusions about general forecasting reliability can be drawn.

The next phase introduces chronological validation windows while keeping the final holdout isolated from model selection.