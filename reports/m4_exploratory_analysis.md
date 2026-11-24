# M4 — Exploratory Time-Series Analysis

## 1. Dataset
- Series: Store 1 — GROCERY I
- Frequency: Daily
- Target: Sales
- Calendar entries: 1,688
- Observed sales: 1,684
- Missing observations: 4
- Missing dates: Christmas (2013–2016)

## 2. Distribution
- Mean: 2,223.17
- Median: 2,283.50
- Standard deviation: 779.29
- Variance: 607,285.66
- Skewness: 0.6484
- Kurtosis: 5.9821
- Minimum: 0
- Maximum: 9,065

The series exhibits substantial variability and moderately positive skewness. Its distribution has relatively heavy tails, with unusually large observations. These observations were retained because they have not been established as data errors.

## 3. Visual behavior
- The series contains substantial daily fluctuations.
- The highest observation was 9,065 on 2016-04-18.
- Several high observations occurred during April 2016 and December 2016.
- Annual averages suggest increasing sales levels.
- Visual trend and structural changes require further investigation.

## 4. Weekly patterns
- Highest average weekday: Wednesday (2,770.36).
- Lowest average weekday: Sunday (1,031.08).
- The large difference between Sunday and other weekdays suggests potential weekly seasonality.

## 5. Monthly patterns
- Highest average month: December (2,771.87).
- Lowest average month: August (2,033.32).
- December appears elevated relative to other months.
- Monthly averages combine multiple years and may be affected by trend and calendar composition.

## 6. Yearly behavior
- 2013: 1,856.16
- 2014: 2,038.03
- 2015: 2,235.35
- 2016: 2,518.37
- 2017: 2,614.39

Annual averages show an increasing pattern. The 2017 data is incomplete, ending on August 15.

## 7. Rolling analysis
- 7-day rolling mean: Pending visual interpretation.
- 28-day rolling mean: Pending visual interpretation.
- 28-day rolling standard deviation: Pending visual interpretation.
- Variance behavior: Pending visual interpretation.

## 8. First differencing
First differencing computes changes between consecutive observations.

The differenced series will be examined for reduced trend and improved stability before determining whether differencing is appropriate for future models.

## 9. Autocorrelation
- ACF observations: Pending visual interpretation.
- PACF observations: Pending visual interpretation.
- Potential weekly structure: Investigate lag 7 and multiples of 7.

## 10. Stationarity tests

### ADF
- Statistic: -4.1504
- p-value: 0.0007985
- Interpretation: Reject the unit-root null hypothesis at conventional significance levels.

### KPSS
- Statistic: 5.9847
- p-value: 0.01 (reported lower lookup bound)
- Interpretation: Reject the stationarity null hypothesis at conventional significance levels.

The two tests produce evidence that does not support a straightforward stationarity conclusion. Further structural and seasonal investigation is warranted.

Both tests were performed on observations after dropping missing values, which compressed the four calendar gaps.

## 11. Modeling implications
- Trend: Investigate further.
- Weekly seasonality: Potentially strong, especially given the Sunday pattern.
- Monthly patterns: December appears elevated.
- Transformations: Investigate, but do not apply automatically.
- Candidate model families: Naive, seasonal naive, ARIMA, SARIMA, ARIMAX and SARIMAX.
- Missing dates: Preserve their distinction from zero sales.
- Outliers: Investigate rather than automatically remove.
- Evaluation: Use chronological validation and compare models against baselines.

Exploratory findings are hypotheses for subsequent modelling and evaluation, not proof of forecasting performance.