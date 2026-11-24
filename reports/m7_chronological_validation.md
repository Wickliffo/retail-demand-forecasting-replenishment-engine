# M7 — Chronological Validation

## 1. Objective

This milestone establishes a chronological validation framework for forecasting model development.

Unlike random splitting, time-series validation preserves temporal ordering. Models are trained using historical observations and evaluated on observations that occur subsequently.

## 2. Validation design

- **Forecast horizon:** 30 days
- **Number of folds:** 5
- **Validation window spacing:** 30 days
- **Training strategy:** Expanding window
- **Development data:** Data preceding the final 30-day holdout

Each successive fold increases the training dataset while moving the validation window forward chronologically.

## 3. Final holdout separation

The final holdout spans 17 July 2017 to 15 August 2017.

The validation framework is constructed from the development series, ending on 16 July 2017, to prevent validation folds from extending into the final holdout.

## 4. Validation artifacts

- `m7_validation_fold_structure.csv`
- `m7_validation_windows.png`

## 5. Limitations

Five validation windows provide a limited representation of forecasting performance across different demand conditions. Results may not generalize to other stores, product categories, longer horizons or changing market conditions.

## 6. Conclusion

Chronological validation provides the structure required for systematic forecasting evaluation. Subsequent model comparisons should use these validation windows rather than relying exclusively on a single test period.