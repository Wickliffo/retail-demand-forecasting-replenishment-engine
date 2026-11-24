# M3 — Time-Series Preparation

## Objective
Construct a continuous daily time-series dataset for Store 1 and GROCERY I, validate its integrity, and preserve genuine missing observations without introducing fabricated demand.

## Series Definition
- Store: 1
- Product family: GROCERY I
- Target: sales
- Frequency: Daily
- Start date: 2013-01-01
- End date: 2017-08-15

## Calendar Reconstruction
A complete daily calendar was constructed between the minimum and maximum dates of the selected series.

- Original observations: 1,684
- Complete calendar entries: 1,688
- Missing observations: 4

## Missing Observations
The missing observations occur on Christmas Day:

- 2013-12-25
- 2014-12-25
- 2015-12-25
- 2016-12-25

These observations were preserved as NaN rather than replaced with zero. This avoids treating unavailable observations as confirmed zero demand.

## Data Validation
- Unique dates: True
- Chronological order: True
- Daily frequency: Confirmed
- Negative observed sales: 0
- Infinite observed values: 0
- Zero-sales observations: 6
- Missing observations: 4

## Target Summary
| Statistic | Value |
|---|---:|
| Count | 1,684 |
| Mean | 2,223.17 |
| Standard deviation | 779.29 |
| Minimum | 0 |
| 25th percentile | 1,873.50 |
| Median | 2,283.50 |
| 75th percentile | 2,649.00 |
| Maximum | 9,065 |

## Output
`data/processed/store1_grocery1_daily_prepared.csv`

The prepared dataset contains 1,688 calendar entries and six columns: date, sales, day_of_week, month, year and is_missing_sales.

## Conclusion
The selected time series has been reconstructed on a complete daily calendar and validated. Missing holiday observations remain explicitly represented. The dataset is ready for exploratory time-series analysis.
