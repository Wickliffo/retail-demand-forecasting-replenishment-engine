# M2 — Time-Series Selection and Calendar Investigation

## Selected Series

- Store: 1
- Product family: GROCERY I
- Target variable: sales
- Frequency: Daily
- Observations: 1,684
- Start date: 2013-01-01
- End date: 2017-08-15

## Calendar Integrity

Four calendar dates are missing:

- 2013-12-25
- 2014-12-25
- 2015-12-25
- 2016-12-25

The event dataset records Christmas (Navidad) as a national holiday on these dates. The sales dataset contains no records on December 25 across stores and product families.

These omissions are systematic and associated with Christmas. They will be handled explicitly during time-series preparation rather than automatically imputed as zero-demand or ordinary operating days.

## Sales Extremes

The highest sales observations for Store 1, GROCERY I include:

- 2016-04-18: 9,065
- 2016-04-19: 8,221
- 2016-04-20: 5,438
- 2016-12-23: 5,386
- 2016-12-21: 5,162

The lowest observations include zero-sales records on January 1 in multiple years and July 7, 2015.

## Monthly Average Sales

December had the highest monthly average sales (2,771.87), while August had the lowest (2,033.32).

These averages are descriptive and do not independently establish seasonal patterns.

## April 2016 Event Investigation

The events dataset records the 2016 Manabi earthquake and subsequent days as national events between April 16 and April 23.

Store 1, GROCERY I sales increased sharply on April 18, reaching 9,065, approximately 3.45 times the mean of April 15–17.

Cross-store GROCERY I statistics also show elevated sales during this period. April 18 mean sales across the 54 stores were 10,800.42, with a median of 9,156.

Store 1's April 18 sales were close to the cross-store median.

The data supports an association between the earthquake event period and broad sales changes. Causality has not been established.

## Event Data

The events dataset contains 350 records and six columns:

- date
- type
- locale
- locale_name
- description
- transferred

No missing values were observed in the event dataset during initial reconnaissance.

Event records include national, regional and local events. Geographic scope will be considered when event features are constructed.

## Limitations

- Event association does not establish causality.
- Monthly averages are not sufficient to establish seasonality.
- The cross-store April 18 comparison is not a store-specific historical baseline.
- Missing Christmas dates require explicit treatment.
- Dataset context is Ecuadorian, not Kenyan.

## M2 Status

Time-series selection and calendar investigation completed. Calendar anomalies, sales extremes and event-related behavior have been documented. No imputation, outlier removal or forecasting model fitting has been performed.