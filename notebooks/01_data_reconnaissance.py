import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# Load the dataset
df = pd.read_csv("data/raw/train.csv")

print("--- HEAD ---")
print(df.head())

print("\n--- DESCRIBE ---")
print(df.describe())

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- DTYPES ---")
print(df.dtypes)

# Check for missing values
missing_df=(
    df.isna().sum().rename_axis("Column").reset_index(name="Missing Values")

)
print("\n--- MISSING VALUES ---")
print(missing_df.to_string(index=False))

# Check for duplicates
exact_duplicates = df.duplicated().sum()
grain_cols =["date","store_nbr","family"]
grain_dupe_mask = df.duplicated(subset=grain_cols)

print("\n--- STEP 2: DUPLICATES ---")
print(f"Completely duplicated rows: {exact_duplicates}")
print(f"Any duplicates on {grain_cols}?: {grain_dupe_mask.any()}")
print(f"Duplicate rows on {grain_cols}: {grain_dupe_mask.sum()}")
#investigate temporal coverage
df["date"] = pd.to_datetime(df["date"])
min_date = df["date"].min()
max_date = df["date"].max()
unique_dates = df["date"].nunique()
expected_range = pd.date_range(start=min_date, end=max_date, freq="D")
is_continuous_daily = (unique_dates == len(expected_range)) and (
    df["date"].dt.normalize().isin(expected_range).all()
)
missing_days = sorted(set(expected_range) - set(df["date"].dt.normalize().unique()))

print("\n--- STEP 3: TEMPORAL COVERAGE ---")
print(f"Earliest date: {min_date.date()}")
print(f"Latest date: {max_date.date()}")
print(f"Total unique dates: {unique_dates}")
print(f"Expected daily calendar days: {len(expected_range)}")
print(f"Continuous daily calendar?: {is_continuous_daily}")
print(f"Missing calendar dates count: {len(missing_days)}")

#Business dimensions
n_stores = df["store_nbr"].nunique()
n_families = df["family"].nunique()
families = sorted(df["family"].dropna().unique())
unique_grain = not df.duplicated(subset=grain_cols).any()

print("\n--- STEP 4: BUSINESS DIMENSIONS ---")
print(f"Number of unique stores: {n_stores}")
print(f"Number of unique product families: {n_families}")
print("Unique values of family:", families)
print(f"Each (date, store_nbr, family) appears only once?: {unique_grain}")

for d in missing_days:
    print(d.date())

# --- STEP 5: SALES DISTRIBUTION ---
mean_sales = df["sales"].mean()
median_sales = df["sales"].median()
std_sales = df["sales"].std()
max_sales = df["sales"].max()

zero_sales_mask = df["sales"] == 0
zero_sales_count = zero_sales_mask.sum()
zero_sales_pct = (zero_sales_count / len(df)) * 100

negative_sales_mask = df["sales"] < 0
negative_sales_count = negative_sales_mask.sum()
has_negative_sales = negative_sales_count > 0

print("\n--- STEP 5: SALES DISTRIBUTION ---")
print(f"Mean sales: {mean_sales:.4f}")
print(f"Median sales: {median_sales:.4f}")
print(f"Standard deviation: {std_sales:.4f}")
print(f"Zero-sales records: {zero_sales_count}")
print(f"Zero-sales percentage: {zero_sales_pct:.2f}%")
print(f"Negative-sales records: {negative_sales_count}")
print(f"Any negative sales values?: {has_negative_sales}")
print(f"Maximum sales: {max_sales}")

# --- STEP 6: DAILY SALES AGGREGATION INVESTIGATION ---
daily_sales = df.groupby("date")["sales"].sum()
avg_daily_sales = daily_sales.mean()
min_daily_sales = daily_sales.min()
min_daily_date = daily_sales.idxmin()
max_daily_sales = daily_sales.max()
max_daily_date = daily_sales.idxmax()
top_5_days = daily_sales.nlargest(5)

print("\n--- STEP 6: DAILY SALES AGGREGATION INVESTIGATION ---")
print(f"Average daily sales: {avg_daily_sales:.2f}")
print(f"Minimum daily sales: {min_daily_sales:.2f} (on {min_daily_date.date()})")
print(f"Maximum daily sales: {max_daily_sales:.2f} (on {max_daily_date.date()})")
print("\nTop 5 days with highest total sales:")
for dt, total in top_5_days.items():
    print(f"  {dt.date()}: {total:.2f}")

from pathlib import Path

# --- STEP 7 & 8: FIRST TIME SERIES + CALENDAR INVESTIGATION ---

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Extract Store 1 and GROCERY I
ts_s1_grocery1 = (
    df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")]
    .sort_values("date")
    .set_index("date")
    .copy()
)

print(f"\nNumber of observations (Store 1, GROCERY I): {len(ts_s1_grocery1)}")

# Define project root and reports directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Plot full time series ---

plot_path = reports_dir / "store1_grocery1_daily_sales.png"

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ts_s1_grocery1.index, ts_s1_grocery1["sales"], linewidth=0.9)

ax.set_title("Daily Sales — Store 1, GROCERY I")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Saved time series plot to: {plot_path}")

# --- 2. Highest sales days ---

print("\nTop 5 highest sales days (Store 1, GROCERY I):")
print(ts_s1_grocery1["sales"].nlargest(5))

# --- 3. Lowest sales days ---

print("\nLowest 5 sales days (Store 1, GROCERY I):")
print(ts_s1_grocery1["sales"].nsmallest(5))

# --- 4. Monthly average sales ---

monthly_avg = ts_s1_grocery1["sales"].groupby(
    ts_s1_grocery1.index.month
).mean()

print("\nAverage daily sales by month (1=Jan ... 12=Dec):")
print(monthly_avg.round(2))

# --- 5. Investigate missing calendar dates ---

expected_dates = pd.date_range(
    start=ts_s1_grocery1.index.min(),
    end=ts_s1_grocery1.index.max(),
    freq="D"
)

missing_dates = expected_dates.difference(ts_s1_grocery1.index)

print("\nMissing dates (Store 1, GROCERY I):")
print(missing_dates)

print(f"\nNumber of missing calendar dates: {len(missing_dates)}")

# --- 6. Investigate April 2016 sales spike ---

april_spike = ts_s1_grocery1.loc["2016-04-15":"2016-04-23"]

print("\nSales around April 2016 spike:")
print(april_spike["sales"])

# --- 7. Plot April 2016 spike investigation ---

april_plot_path = reports_dir / "store1_grocery1_april_2016_spike.png"

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(april_spike.index, april_spike["sales"], marker="o", linewidth=1.2)

ax.set_title("Sales Investigation — April 15–23, 2016")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

fig.tight_layout()
fig.savefig(april_plot_path, dpi=150)
plt.close(fig)

print(f"\nSaved April investigation plot to: {april_plot_path}")

# --- STEP 9: EVENT DATA RECONNAISSANCE ---

events_path = PROJECT_ROOT / "data" / "raw" / "holidays_events.csv"

events_df = pd.read_csv(events_path)

print("\nEvent dataset shape:")
print(events_df.shape)

print("\nFirst 10 rows:")
print(events_df.head(10))

print("\nColumn names:")
print(events_df.columns.tolist())

print("\nData types:")
print(events_df.dtypes)

print("\nMissing values:")
print(events_df.isnull().sum())

print("\nEvent types:")
print(events_df["type"].value_counts())

print("\nEvent locations:")
print(events_df["locale"].value_counts())

print("\nEvent descriptions:")
print(events_df["description"].value_counts().head(15))

# --- STEP 10: INVESTIGATE APRIL 2016 EVENTS ---

events_df["date"] = pd.to_datetime(events_df["date"])

april_events = events_df.loc[
    (events_df["date"] >= "2016-04-15") &
    (events_df["date"] <= "2016-04-23")
].sort_values("date")

print("\nEvents from April 15–23, 2016:")
print(april_events.to_string(index=False))

# --- STEP 11: CHRISTMAS EVENT INVESTIGATION ---

christmas_events = events_df.loc[
    (events_df["date"].dt.month == 12) &
    (events_df["date"].dt.day == 25)
].sort_values("date")

print("\nChristmas event records:")
print(christmas_events.to_string(index=False))

# --- STEP 12: TRANSFERRED EVENTS ---

transferred_events = events_df.loc[
    events_df["transferred"]
].sort_values("date")

print("\nTransferred events:")
print(transferred_events.to_string(index=False))

# --- STEP 13: NATIONAL EVENTS ---

national_events = events_df.loc[
    events_df["locale"] == "National"
].sort_values("date")

print("\nNational events:")
print(national_events.head(20).to_string(index=False))

print(f"\nTotal national event records: {len(national_events)}")

# --- STEP 14: REGIONAL EVENTS ---

regional_events = events_df.loc[
    events_df["locale"] == "Regional"
].sort_values("date")

print("\nRegional events:")
print(regional_events.to_string(index=False))

print(f"\nTotal regional event records: {len(regional_events)}")

# --- STEP 15: LOCAL EVENTS ---

local_events = events_df.loc[
    events_df["locale"] == "Local"
].sort_values("date")

print("\nLocal events:")
print(local_events.head(20).to_string(index=False))

print(f"\nTotal local event records: {len(local_events)}")

# --- STEP 16: APRIL 18 SALES ACROSS ALL STORES ---

april_18_all_stores = df.loc[
    (df["date"] == "2016-04-18") &
    (df["family"] == "GROCERY I")
].sort_values("sales", ascending=False)

print("\nApril 18, 2016 GROCERY I sales across stores:")
print(april_18_all_stores[["store_nbr", "sales"]].to_string(index=False))

print("\nApril 18 cross-store statistics:")
print(april_18_all_stores["sales"].describe())

# --- STEP 17: APRIL 19 SALES ACROSS ALL STORES ---

april_19_all_stores = df.loc[
    (df["date"] == "2016-04-19") &
    (df["family"] == "GROCERY I")
].sort_values("sales", ascending=False)

print("\nApril 19, 2016 GROCERY I sales across stores:")
print(april_19_all_stores[["store_nbr", "sales"]].to_string(index=False))

print("\nApril 19 cross-store statistics:")
print(april_19_all_stores["sales"].describe())

# --- STEP 18: APRIL 15–23 CROSS-STORE COMPARISON ---

april_period = df.loc[
    (df["date"] >= "2016-04-15") &
    (df["date"] <= "2016-04-23") &
    (df["family"] == "GROCERY I")
]

april_cross_store = (
    april_period.groupby("date")["sales"]
    .agg(["mean", "median", "std", "min", "max", "count"])
)

print("\nGROCERY I cross-store sales statistics, April 15–23:")
print(april_cross_store)

# --- STEP 19: STORE 1 ALL FAMILIES ON APRIL 18 ---

store1_april18 = df.loc[
    (df["date"] == "2016-04-18") &
    (df["store_nbr"] == 1)
].sort_values("sales", ascending=False)

print("\nStore 1 sales across product families on April 18:")
print(store1_april18[["family", "sales"]].to_string(index=False))

# --- STEP 20: STORE COMPARISON ON APRIL 18 ---

store_comparison = df.loc[
    (df["date"] == "2016-04-18") &
    (df["family"] == "GROCERY I")
].copy()

store_comparison["store_daily_sales_ratio"] = (
    store_comparison["sales"] /
    store_comparison["sales"].median()
)

print("\nApril 18 store comparison:")
print(
    store_comparison[
        ["store_nbr", "sales", "store_daily_sales_ratio"]
    ].sort_values("sales", ascending=False).to_string(index=False)
)

# --- STEP 21: DATASET-WIDE EARTHQUAKE PERIOD INVESTIGATION ---

earthquake_period = df.loc[
    (df["date"] >= "2016-04-15") &
    (df["date"] <= "2016-04-23")
]

earthquake_daily_sales = (
    earthquake_period.groupby("date")
    .agg(
        total_sales=("sales", "sum"),
        mean_sales=("sales", "mean"),
        median_sales=("sales", "median"),
        records=("sales", "count")
    )
)

print("\nDataset-wide sales around April 2016:")
print(earthquake_daily_sales)

# --- STEP 22: CHRISTMAS SALES RECORD INVESTIGATION ---

christmas_sales = df.loc[
    (df["date"].dt.month == 12) &
    (df["date"].dt.day == 25)
]

print("\nChristmas sales records:")
print(christmas_sales)

print(f"\nNumber of Christmas sales records: {len(christmas_sales)}")

# --- STEP 25: RECONSTRUCT SELECTED SERIES ---

df["date"] = pd.to_datetime(df["date"])

ts_s1_grocery1 = (
    df.loc[
        (df["store_nbr"] == 1) &
        (df["family"] == "GROCERY I"),
        ["date", "sales"]
    ]
    .sort_values("date")
    .set_index("date")
)

print("\nReconstructed series:")
print(ts_s1_grocery1.head())

print("\nSeries shape:")
print(ts_s1_grocery1.shape)

# --- STEP 26: INDEX INTEGRITY ---

print("\nIndex is unique:")
print(ts_s1_grocery1.index.is_unique)

print("\nIndex is monotonic increasing:")
print(ts_s1_grocery1.index.is_monotonic_increasing)

print("\nIndex type:")
print(type(ts_s1_grocery1.index))

# --- STEP 26: INDEX INTEGRITY ---

print("\nIndex is unique:")
print(ts_s1_grocery1.index.is_unique)

print("\nIndex is monotonic increasing:")
print(ts_s1_grocery1.index.is_monotonic_increasing)

print("\nIndex type:")
print(type(ts_s1_grocery1.index))

# --- STEP 27: FULL CALENDAR ---

full_calendar = pd.date_range(
    start=ts_s1_grocery1.index.min(),
    end=ts_s1_grocery1.index.max(),
    freq="D"
)

print("\nFull calendar length:")
print(len(full_calendar))

print("\nCalendar frequency:")
print(full_calendar.freq)

# --- STEP 28: REINDEX TO FULL CALENDAR ---

ts_calendar = ts_s1_grocery1.reindex(full_calendar)

ts_calendar.index.name = "date"

print("\nCalendar-reindexed series:")
print(ts_calendar.head())

print("\nCalendar-reindexed series shape:")
print(ts_calendar.shape)

# --- STEP 29: MISSING OBSERVATION INVESTIGATION ---

missing_sales = ts_calendar["sales"].isna()

print("\nMissing sales observations:")
print(ts_calendar.loc[missing_sales])

print("\nMissing observation count:")
print(missing_sales.sum())

print("\nMissing observation percentage:")
print(round(missing_sales.mean() * 100, 4))

# --- STEP 30: MISSING DATE CLASSIFICATION ---

missing_date_report = pd.DataFrame({
    "date": ts_calendar.index[missing_sales]
})

missing_date_report["reason"] = "National Christmas holiday"

print("\nMissing date classification:")
print(missing_date_report)

# --- STEP 31: PRESERVE MISSING SALES ---

print("\nMissing sales remain NaN:")
print(ts_calendar["sales"].isna().sum())

print("\nZero sales count:")
print((ts_calendar["sales"] == 0).sum())

# --- STEP 32: TARGET VALIDATION ---

observed_sales = ts_calendar["sales"].dropna()

print("\nNegative observed sales:")
print((observed_sales < 0).sum())

print("\nInfinite values:")
print(np.isinf(observed_sales).sum())

print("\nObserved target summary:")
print(observed_sales.describe())

# --- STEP 33: DAILY FREQUENCY VALIDATION ---

print("\nCalendar is daily:")
print(ts_calendar.index.equals(full_calendar))

print("\nCalendar length:")
print(len(ts_calendar))

print("\nCalendar start:")
print(ts_calendar.index.min())

print("\nCalendar end:")
print(ts_calendar.index.max())

# --- STEP 34: DATE ALIGNMENT ---

print("\nFirst 10 prepared calendar entries:")
print(ts_calendar.head(10))

print("\nLast 10 prepared calendar entries:")
print(ts_calendar.tail(10))

# --- STEP 35: CALENDAR METADATA ---

prepared_calendar_info = pd.DataFrame({
    "date": ts_calendar.index,
    "sales": ts_calendar["sales"].values
})

prepared_calendar_info["day_of_week"] = (
    prepared_calendar_info["date"].dt.day_of_week
)

prepared_calendar_info["month"] = (
    prepared_calendar_info["date"].dt.month
)

prepared_calendar_info["year"] = (
    prepared_calendar_info["date"].dt.year
)

prepared_calendar_info["is_missing_sales"] = (
    prepared_calendar_info["sales"].isna()
)

print("\nCalendar metadata preview:")
print(prepared_calendar_info.head())

# --- STEP 36: PREPARED DATASET ---

prepared_ts = prepared_calendar_info.copy()

print("\nPrepared dataset:")
print(prepared_ts.head())

print("\nPrepared dataset shape:")
print(prepared_ts.shape)

# --- STEP 37: EXPORT PREPARED SERIES ---

processed_dir = PROJECT_ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

prepared_path = (
    processed_dir / "store1_grocery1_daily_prepared.csv"
)

prepared_ts.to_csv(prepared_path, index=False)

print("\nPrepared dataset saved:")
print(prepared_path)

# --- STEP 38: ROUND-TRIP VALIDATION ---

prepared_check = pd.read_csv(
    prepared_path,
    parse_dates=["date"]
)

print("\nReloaded prepared dataset:")
print(prepared_check.head())

print("\nReloaded shape:")
print(prepared_check.shape)

print("\nReloaded missing sales:")
print(prepared_check["sales"].isna().sum())

print("\nReloaded date uniqueness:")
print(prepared_check["date"].is_unique)

print("\nReloaded date order:")
print(prepared_check["date"].is_monotonic_increasing)


# --- STEP 41: LOAD PREPARED DATASET ---

prepared_path = (
    PROJECT_ROOT / "data" / "processed" /
    "store1_grocery1_daily_prepared.csv"
)

prepared_ts = pd.read_csv(
    prepared_path,
    parse_dates=["date"]
)

print("\nPrepared dataset loaded:")
print(prepared_ts.head())

print("\nDataset shape:")
print(prepared_ts.shape)

# --- STEP 42: SET TIME INDEX ---

ts = (
    prepared_ts
    .set_index("date")
    .sort_index()
)

print("\nTime-series dataframe:")
print(ts.head())

print("\nIndex:")
print(ts.index)

# --- STEP 43: EXTRACT TARGET ---

y = ts["sales"]

print("\nTarget series:")
print(y.head())

print("\nTarget length:")
print(len(y))

print("\nMissing values:")
print(y.isna().sum())

# --- STEP 44: FULL TIME SERIES PLOT ---

plt.figure(figsize=(16, 6))

plt.plot(y.index, y, linewidth=0.8)

plt.title("Store 1 - GROCERY I Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_full_series.png",
    dpi=300,
)

plt.close()



# --- STEP 45: SALES DISTRIBUTION ---

observed_y = y.dropna()

plt.figure(figsize=(10, 6))

sns.histplot(
    observed_y,
    bins=40,
    kde=True
)

plt.title("Distribution of Daily Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_sales_distribution.png",
    dpi=300
)

plt.close()
# --- STEP 46: SALES BOXPLOT ---

plt.figure(figsize=(10, 5))

sns.boxplot(x=observed_y)

plt.title("Boxplot of Daily Sales")
plt.xlabel("Sales")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_sales_boxplot.png",
    dpi=300
)

plt.close()
# --- STEP 47: DISTRIBUTION STATISTICS ---

distribution_stats = pd.Series({
    "Mean": observed_y.mean(),
    "Median": observed_y.median(),
    "Standard Deviation": observed_y.std(),
    "Variance": observed_y.var(),
    "Skewness": observed_y.skew(),
    "Kurtosis": observed_y.kurt(),
    "Minimum": observed_y.min(),
    "Maximum": observed_y.max()
})

print("\nDistribution statistics:")
print(distribution_stats)

# --- STEP 48: TOP SALES OBSERVATIONS ---

top_sales = (
    ts[["sales"]]
    .dropna()
    .sort_values("sales", ascending=False)
    .head(15)
)

print("\nTop 15 sales observations:")
print(top_sales)

# --- STEP 49: WEEKLY PATTERN ---

weekly_sales = ts.groupby(
    ts.index.day_of_week
)["sales"].mean()

weekly_sales.index = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

print("\nAverage sales by weekday:")
print(weekly_sales)

plt.figure(figsize=(10, 5))

weekly_sales.plot(kind="bar")

plt.title("Average Daily Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_weekly_pattern.png",
    dpi=300
)

plt.close()

# --- STEP 50: MONTHLY PATTERN ---

monthly_sales = ts.groupby(
    ts.index.month
)["sales"].mean()

print("\nAverage sales by month:")
print(monthly_sales)

plt.figure(figsize=(10, 5))

monthly_sales.plot(kind="bar")

plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_monthly_pattern.png",
    dpi=300
)

plt.close()

# --- STEP 51: YEARLY PATTERN ---

yearly_sales = ts.groupby(
    ts.index.year
)["sales"].mean()

print("\nAverage sales by year:")
print(yearly_sales)

plt.figure(figsize=(10, 5))

yearly_sales.plot(kind="bar")

plt.title("Average Sales by Year")
plt.xlabel("Year")
plt.ylabel("Average Daily Sales")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_yearly_pattern.png",
    dpi=300
)

plt.close()

# --- STEP 52: MONTHLY DISTRIBUTION ---

month_data = ts[["sales"]].dropna().copy()

month_data["month"] = month_data.index.month

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=month_data,
    x="month",
    y="sales"
)

plt.title("Daily Sales Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_monthly_boxplot.png",
    dpi=300
)

plt.close()

# --- STEP 53: ROLLING MEANS ---

rolling_7 = y.rolling(
    window=7,
    min_periods=7
).mean()

rolling_28 = y.rolling(
    window=28,
    min_periods=28
).mean()

plt.figure(figsize=(16, 6))

plt.plot(y.index, y, label="Daily sales", alpha=0.4)
plt.plot(rolling_7.index, rolling_7, label="7-day mean")
plt.plot(rolling_28.index, rolling_28, label="28-day mean")

plt.title("Daily Sales with Rolling Means")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_rolling_means.png",
    dpi=300
)

plt.close()
# --- STEP 54: ROLLING VOLATILITY ---

rolling_std_28 = y.rolling(
    window=28,
    min_periods=28
).std()

plt.figure(figsize=(16, 5))

plt.plot(
    rolling_std_28.index,
    rolling_std_28
)

plt.title("28-Day Rolling Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Rolling Standard Deviation")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_rolling_std.png",
    dpi=300
)

plt.close()
# --- STEP 55: FIRST DIFFERENCE ---

first_difference = y.diff()

print("\nFirst difference:")
print(first_difference.head(10))

plt.figure(figsize=(16, 5))

plt.plot(
    first_difference.index,
    first_difference
)

plt.title("First Difference of Daily Sales")
plt.xlabel("Date")
plt.ylabel("Change in Sales")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_first_difference.png",
    dpi=300
)

plt.close()
# --- STEP 56: ACF ---

acf_series = y.dropna()

fig, ax = plt.subplots(figsize=(12, 6))

plot_acf(
    acf_series,
    lags=60,
    ax=ax
)

ax.set_title("Autocorrelation Function (ACF)")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_acf.png",
    dpi=300
)

plt.close()
# --- STEP 57: PACF ---

fig, ax = plt.subplots(figsize=(12, 6))

plot_pacf(
    acf_series,
    lags=40,
    method="ywm",
    ax=ax
)

ax.set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m4_pacf.png",
    dpi=300
)

plt.close()
# --- STEP 58: ADF TEST ---

adf_series = y.dropna()

adf_result = adfuller(
    adf_series,
    autolag="AIC"
)

print("\nADF Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Used Lags: {adf_result[2]}")
print(f"Observations: {adf_result[3]}")

print("\nCritical Values:")

for key, value in adf_result[4].items():
    print(f"{key}: {value}")

# --- STEP 59: KPSS TEST ---

kpss_result = kpss(
    adf_series,
    regression="c",
    nlags="auto"
)

print("\nKPSS Test Results:")
print(f"KPSS Statistic: {kpss_result[0]}")
print(f"p-value: {kpss_result[1]}")
print(f"Used Lags: {kpss_result[2]}")

print("\nCritical Values:")

for key, value in kpss_result[3].items():
    print(f"{key}: {value}")

# --- STEP 61: FORECASTING OBJECTIVE ---

forecast_horizon = 30

print("\nForecast horizon:")
print(forecast_horizon)

print("\nForecasting objective:")
print("Forecast the next 30 daily sales observations.")

# --- STEP 63: TEST BOUNDARY ---

test_start = y.index[-forecast_horizon]

print("\nTest start:")
print(test_start)

print("\nTest end:")
print(y.index[-1])

# --- STEP 64: TRAIN AND TEST ---

train = y.loc[y.index < test_start]

test = y.loc[y.index >= test_start]

print("\nTraining length:")
print(len(train))

print("\nTest length:")
print(len(test))

print("\nTraining end:")
print(train.index[-1])

print("\nTest start:")
print(test.index[0])

# --- STEP 65: HOLDOUT INSPECTION ---

print("\nTraining data preview:")
print(train.tail())

print("\nTest data:")
print(test)

print("\nMissing training values:")
print(train.isna().sum())

print("\nMissing test values:")
print(test.isna().sum())

# --- STEP 66: LAST TRAINING OBSERVATION ---

last_train_value = train.dropna().iloc[-1]

last_train_date = train.dropna().index[-1]

print("\nLast observed training value:")
print(last_train_value)

print("\nDate of last observed training value:")
print(last_train_date)

# --- STEP 67: NAIVE FORECAST ---

naive_forecast = pd.Series(
    last_train_value,
    index=test.index,
    name="naive_forecast"
)

print("\nNaive forecast:")
print(naive_forecast)

# --- STEP 68: FORECAST LENGTH VALIDATION ---

print("\nForecast length:")
print(len(naive_forecast))

print("\nTest length:")
print(len(test))

print("\nLengths match:")
print(len(naive_forecast) == len(test))

# --- STEP 69: ALIGNMENT VALIDATION ---

print("\nForecast index equals test index:")
print(naive_forecast.index.equals(test.index))

print("\nForecast dates:")
print(naive_forecast.index)

# --- STEP 70: EVALUATION DATASET ---

evaluation = pd.DataFrame({
    "actual": test,
    "forecast": naive_forecast
})

print("\nEvaluation dataset:")
print(evaluation)

# --- STEP 71: FORECAST ERRORS ---

evaluation["error"] = (
    evaluation["actual"] - evaluation["forecast"]
)

evaluation["absolute_error"] = (
    evaluation["error"].abs()
)

evaluation["squared_error"] = (
    evaluation["error"] ** 2
)

print("\nEvaluation with errors:")
print(evaluation)

# --- STEP 72: MAE ---

mae = evaluation["absolute_error"].mean()

print("\nMean Absolute Error:")
print(mae)

# --- STEP 73: RMSE ---

rmse = np.sqrt(
    evaluation["squared_error"].mean()
)

print("\nRoot Mean Squared Error:")
print(rmse)

# --- STEP 74: MEAN ERROR ---

mean_error = evaluation["error"].mean()

print("\nMean Forecast Error:")
print(mean_error)

# --- STEP 75: MAE RELATIVE TO TEST MEAN ---

test_mean = evaluation["actual"].mean()

mae_relative = (
    mae / test_mean
) * 100

print("\nTest mean actual sales:")
print(test_mean)

print("\nMAE as percentage of test mean:")
print(mae_relative)

# --- STEP 76: WORST FORECAST ERRORS ---

worst_errors = (
    evaluation
    .sort_values("absolute_error", ascending=False)
    .head(10)
)

print("\nTop 10 forecast errors:")
print(worst_errors)

# --- STEP 77: BEST FORECAST ERRORS ---

best_errors = (
    evaluation
    .sort_values("absolute_error")
    .head(10)
)

print("\nTop 10 smallest forecast errors:")
print(best_errors)

# --- STEP 78: FORECAST BIAS ---

bias = evaluation["error"].mean()

print("\nForecast bias:")
print(bias)

if bias > 0:
    print("The forecast underestimates actual sales on average.")

elif bias < 0:
    print("The forecast overestimates actual sales on average.")

else:
    print("The forecast has zero mean error.")

    # --- STEP 79: ACTUAL VS FORECAST ---

plt.figure(figsize=(14, 6))

plt.plot(
    evaluation.index,
    evaluation["actual"],
    label="Actual Sales",
    marker="o"
)

plt.plot(
    evaluation.index,
    evaluation["forecast"],
    label="Naive Forecast",
    linestyle="--"
)

plt.title("Naive Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m5_naive_forecast.png",
    dpi=300
)

plt.close()

# --- STEP 80: FULL FORECAST CONTEXT ---

plt.figure(figsize=(16, 6))

plt.plot(
    train.tail(90).index,
    train.tail(90),
    label="Training history (last 90 days)"
)

plt.plot(
    test.index,
    test,
    label="Actual test sales"
)

plt.plot(
    naive_forecast.index,
    naive_forecast,
    label="Naive forecast",
    linestyle="--"
)

plt.title("Naive Forecast in Historical Context")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "reports" / "m5_naive_context.png",
    dpi=300
)

plt.close()

# --- STEP 81: EVALUATION SAMPLE VALIDATION ---

valid_evaluation = evaluation.dropna(
    subset=["actual", "forecast"]
)

print("\nEvaluation rows:")
print(len(evaluation))

print("\nValid evaluation rows:")
print(len(valid_evaluation))

print("\nMissing actual values:")
print(evaluation["actual"].isna().sum())

print("\nMissing forecasts:")
print(evaluation["forecast"].isna().sum())

# --- STEP 82: METRIC SUMMARY ---

naive_metrics = pd.DataFrame({
    "model": ["Naive"],
    "horizon": [forecast_horizon],
    "mae": [mae],
    "rmse": [rmse],
    "mean_error": [mean_error],
    "test_mean": [test_mean],
    "mae_relative_percent": [mae_relative]
})

print("\nNaive metric summary:")
print(naive_metrics)

# --- STEP 83: EXPORT FORECAST RESULTS ---

forecast_path = (
    PROJECT_ROOT / "reports" / "m5_naive_forecast_results.csv"
)

evaluation.to_csv(
    forecast_path,
    index=True
)

print("\nForecast results saved:")
print(forecast_path)

# --- STEP 84: EXPORT BENCHMARK METRICS ---

metrics_path = (
    PROJECT_ROOT / "reports" / "m5_naive_metrics.csv"
)

naive_metrics.to_csv(
    metrics_path,
    index=False
)

print("\nBenchmark metrics saved:")
print(metrics_path)

# --- STEP 85: VERIFY EXPORTED METRICS ---

print("\nNaive metrics export verification:")

print(metrics_path.exists())
print(pd.read_csv(metrics_path))

#--------Seasonal Naive Forecasting--------#
# --- STEP 86: DEFINE SEASONAL PERIOD ---

seasonal_period = 7
# --- STEP 87: EXTRACT LAST SEVEN TRAINING OBSERVATIONS ---

seasonal_values = train.iloc[-seasonal_period:]

# --- STEP 88: INSPECT SEASONAL VALUES ---

print(seasonal_values)

# --- STEP 89: CHECK MISSING SEASONAL OBSERVATIONS ---

print("Missing values:", seasonal_values.isna().sum())

# --- STEP 90: VALIDATE SEASONAL PATTERN ---

assert len(seasonal_values) == seasonal_period
assert seasonal_values.isna().sum() == 0

#----Understand and generate predictions---#
# --- STEP 91: INSPECT WEEKDAY LABELS ---

print(seasonal_values.index.day_name())

# --- STEP 92: CREATE SEASONAL PATTERN TABLE ---

seasonal_pattern = pd.DataFrame({
    "weekday": seasonal_values.index.day_name(),
    "sales": seasonal_values.values
})

print(seasonal_pattern)

# --- STEP 93: REPEAT WEEKLY PATTERN ---

seasonal_forecast_values = np.tile(
    seasonal_values.to_numpy(),
    int(np.ceil(forecast_horizon / seasonal_period))
)[:forecast_horizon]

# --- STEP 94: CONSTRUCT SEASONAL FORECAST ---

seasonal_forecast = pd.Series(
    seasonal_forecast_values,
    index=test.index,
    name="seasonal_forecast"
)

# --- STEP 95: VERIFY FORECAST ---

print("Forecast length:", len(seasonal_forecast))
print("Test length:", len(test))

print(
    "Index alignment:",
    seasonal_forecast.index.equals(test.index)
)

assert len(seasonal_forecast) == len(test)
assert seasonal_forecast.index.equals(test.index)

# --- STEP 96: CREATE SEASONAL EVALUATION DATAFRAME ---

seasonal_evaluation = pd.DataFrame({
    "actual": test,
    "forecast": seasonal_forecast
})

print(seasonal_evaluation)

# --- STEP 97: CALCULATE ERROR ---

seasonal_evaluation["error"] = (
    seasonal_evaluation["actual"]
    - seasonal_evaluation["forecast"]
)

# --- STEP 98: CALCULATE ABSOLUTE ERROR ---

seasonal_evaluation["absolute_error"] = (
    seasonal_evaluation["error"].abs()
)

# --- STEP 99: CALCULATE SQUARED ERROR ---

seasonal_evaluation["squared_error"] = (
    seasonal_evaluation["error"] ** 2
)

# --- STEP 100: INSPECT SEASONAL EVALUATION ---

print(seasonal_evaluation)

# --- STEP 101: CALCULATE SEASONAL MAE ---

seasonal_mae = (
    seasonal_evaluation["absolute_error"].mean()
)

print("Seasonal MAE:", seasonal_mae)

# --- STEP 102: CALCULATE SEASONAL RMSE ---

seasonal_rmse = np.sqrt(
    seasonal_evaluation["squared_error"].mean()
)

print("Seasonal RMSE:", seasonal_rmse)

# --- STEP 103: CALCULATE SEASONAL MEAN ERROR ---

seasonal_mean_error = (
    seasonal_evaluation["error"].mean()
)

print("Seasonal mean error:", seasonal_mean_error)

# --- STEP 104: CALCULATE TEST MEAN ---

seasonal_test_mean = (
    seasonal_evaluation["actual"].mean()
)

print("Test mean:", seasonal_test_mean)

# --- STEP 105: CALCULATE RELATIVE MAE ---

seasonal_mae_relative = (
    seasonal_mae / seasonal_test_mean
) * 100

print(
    "Relative MAE (%):",
    seasonal_mae_relative
)

# --- STEP 106: COMPARE BASELINE MODELS ---

benchmark_comparison = pd.DataFrame({
    "model": ["Naive", "Seasonal Naive"],
    "MAE": [mae, seasonal_mae],
    "RMSE": [rmse, seasonal_rmse],
    "Mean Error": [mean_error, seasonal_mean_error]
})

print(benchmark_comparison)

# --- STEP 107: CALCULATE MAE IMPROVEMENT ---

mae_improvement = (
    (mae - seasonal_mae) / mae
) * 100

print("MAE improvement (%):", mae_improvement)

# --- STEP 108: IDENTIFY LOWER MAE MODEL ---

if seasonal_mae < mae:
    print("Seasonal Naive has lower MAE.")
elif seasonal_mae > mae:
    print("Naive has lower MAE.")
else:
    print("Both models have equal MAE.")

# --- STEP 109: IDENTIFY LOWER RMSE MODEL ---

if seasonal_rmse < rmse:
    print("Seasonal Naive has lower RMSE.")
elif seasonal_rmse > rmse:
    print("Naive has lower RMSE.")
else:
    print("Both models have equal RMSE.")

# --- STEP 110: INSPECT SEASONAL FORECAST BIAS ---

if seasonal_mean_error > 0:
    print("Seasonal forecast tends to underestimate.")
elif seasonal_mean_error < 0:
    print("Seasonal forecast tends to overestimate.")
else:
    print("Seasonal forecast has zero mean error.")

# --- STEP 111: PLOT SEASONAL FORECAST ---

plt.figure(figsize=(14, 6))

plt.plot(
    test.index,
    test,
    label="Actual",
    marker="o"
)

plt.plot(
    seasonal_forecast.index,
    seasonal_forecast,
    label="Seasonal Naive",
    marker="o"
)

plt.title("Seasonal Naive Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    reports_dir / "m6_seasonal_forecast.png",
    dpi=300
)

plt.close()

# --- STEP 112: PLOT SEASONAL FORECAST CONTEXT ---

context = train.iloc[-90:]

plt.figure(figsize=(14, 6))

plt.plot(
    context.index,
    context,
    label="Training history"
)

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    seasonal_forecast.index,
    seasonal_forecast,
    label="Seasonal forecast"
)

plt.title("Seasonal Naive Forecast with Historical Context")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    reports_dir / "m6_seasonal_context.png",
    dpi=300
)

plt.close()

# --- STEP 114: EXPORT SEASONAL METRICS ---

seasonal_metrics = pd.DataFrame({
    "model": ["Seasonal Naive"],
    "horizon": [forecast_horizon],
    "mae": [seasonal_mae],
    "rmse": [seasonal_rmse],
    "mean_error": [seasonal_mean_error],
    "test_mean": [seasonal_test_mean],
    "mae_relative_percent": [seasonal_mae_relative]
})

seasonal_metrics_path = (
    reports_dir / "m6_seasonal_metrics.csv"
)

seasonal_metrics.to_csv(
    seasonal_metrics_path,
    index=False
)

print(seasonal_metrics)
# --- FIX: EXPORT SEASONAL EVALUATION ---

seasonal_results_path = (
    reports_dir / "m6_seasonal_forecast_results.csv"
)

seasonal_evaluation.to_csv(
    seasonal_results_path
)

print("File exists:", seasonal_results_path.exists())

#--Establish validation design--#
# --- STEP 117: DEFINE VALIDATION HORIZON ---

validation_horizon = 30
n_validation_windows = 5
validation_step = 30

# --- STEP 118: EXCLUDE FINAL HOLDOUT FROM DEVELOPMENT SERIES ---

development_series = y.iloc[:-forecast_horizon].copy()

assert len(development_series) == len(y) - forecast_horizon

assert development_series.index[-1] == pd.Timestamp("2017-07-16")

print("Development observations:", len(development_series))
print("Development end:", development_series.index[-1])
print("Final holdout starts:", y.index[-forecast_horizon])


# --- STEP 119: DEFINE REQUIRED VALIDATION OBSERVATIONS ---

required_observations = (
    validation_horizon * n_validation_windows
)

assert len(development_series) >= required_observations


# --- STEP 120: DEFINE VALIDATION START POSITIONS ---

validation_starts = [
    len(development_series) - required_observations
    + i * validation_step
    for i in range(n_validation_windows)
]


# --- STEP 121: CONSTRUCT FOLD METADATA ---

fold_records = []

for fold_number, start in enumerate(validation_starts, start=1):

    train_fold = development_series.iloc[:start]

    validation_fold = development_series.iloc[
        start:start + validation_horizon
    ]

    fold_records.append({
        "fold": fold_number,
        "train_start": train_fold.index[0],
        "train_end": train_fold.index[-1],
        "validation_start": validation_fold.index[0],
        "validation_end": validation_fold.index[-1],
        "train_length": len(train_fold),
        "validation_length": len(validation_fold)
    })

fold_table = pd.DataFrame(fold_records)


# --- STEP 122: VALIDATE FOLD COUNT ---

assert len(fold_table) == n_validation_windows


# --- STEP 123: VALIDATE HORIZON LENGTH ---

assert (
    fold_table["validation_length"] == validation_horizon
).all()


# --- STEP 124: VALIDATE DEVELOPMENT BOUNDARY ---

assert (
    pd.to_datetime(fold_table["validation_end"]).max()
    <= pd.Timestamp("2017-07-16")
)


# --- STEP 125: VALIDATE TRAINING/VALIDATION ORDER ---

assert (
    pd.to_datetime(fold_table["validation_start"])
    > pd.to_datetime(fold_table["train_end"])
).all()


# --- STEP 126: VALIDATE EXPANDING TRAINING WINDOWS ---

assert (
    fold_table["train_length"].diff().dropna() > 0
).all()


# --- STEP 127: VALIDATE NON-OVERLAPPING VALIDATION WINDOWS ---

for i in range(len(fold_table) - 1):

    current_end = pd.Timestamp(
        fold_table.loc[i, "validation_end"]
    )

    next_start = pd.Timestamp(
        fold_table.loc[i + 1, "validation_start"]
    )

    assert next_start > current_end


# --- STEP 128: DISPLAY CORRECTED FOLDS ---

print("\nCorrected chronological validation folds:")

print(fold_table.to_string(index=False))


# --- STEP 129: EXPORT FOLD TABLE ---

fold_table_path = reports_dir / "m7_validation_fold_structure.csv"

fold_table.to_csv(
    fold_table_path,
    index=False
)


# --- STEP 130: VERIFY FOLD EXPORT ---

fold_check = pd.read_csv(
    fold_table_path,
    parse_dates=[
        "train_start",
        "train_end",
        "validation_start",
        "validation_end"
    ]
)

assert len(fold_check) == 5

assert fold_check["validation_end"].max() <= pd.Timestamp("2017-07-16")

print("\nFold export verified.")


# --- STEP 131: PLOT CORRECTED VALIDATION WINDOWS ---

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    development_series.index,
    development_series,
    label="Development series"
)

for _, fold in fold_table.iterrows():

    ax.axvspan(
        fold["validation_start"],
        fold["validation_end"],
        alpha=0.2
    )

ax.set_title("M7: Corrected Chronological Validation Windows")

ax.set_xlabel("Date")

ax.set_ylabel("Sales")

ax.legend()

ax.grid(True)

fig.tight_layout()

fig.savefig(
    reports_dir / "m7_validation_windows.png",
    dpi=300
)

plt.close(fig)


# --- STEP 132: VALIDATE FINAL HOLDOUT SEPARATION ---

test = y.iloc[-forecast_horizon:].copy()

assert test.index[0] == pd.Timestamp("2017-07-17")

assert test.index[-1] == pd.Timestamp("2017-08-15")

assert fold_table["validation_end"].max() < test.index[0]

print("\nM7 corrected successfully.")

print("Validation end:", fold_table.iloc[-1]["validation_end"])

print("Holdout start:", test.index[0])

print("Holdout end:", test.index[-1])

# --- STEP 144: PLOT VALIDATION WINDOWS ---

plt.figure(figsize=(14, 6))

plt.plot(
    y.index,
    y,
    label="Historical sales"
)

for _, fold in fold_table.iterrows():

    plt.axvspan(
        fold["validation_start"],
        fold["validation_end"],
        alpha=0.2
    )

plt.title("Chronological Validation Windows")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    reports_dir / "m7_validation_windows.png",
    dpi=300
)

plt.close()



# --- STEP 147: ERROR ANALYSIS DATASET ---

m8_error_analysis = pd.DataFrame(
    {
        "actual": test.values,
        "forecast": seasonal_forecast.values,
    },
    index=test.index
)

print(m8_error_analysis.head())
print(m8_error_analysis.shape)


# --- STEP 148: FORECAST ERRORS ---

m8_error_analysis["error"] = (
    m8_error_analysis["actual"]
    - m8_error_analysis["forecast"]
)

print(m8_error_analysis.head())


# --- STEP 149: ABSOLUTE ERRORS ---

m8_error_analysis["absolute_error"] = (
    m8_error_analysis["error"].abs()
)

print(m8_error_analysis.head())


# --- STEP 150: SQUARED ERRORS ---

m8_error_analysis["squared_error"] = (
    m8_error_analysis["error"] ** 2
)

print(m8_error_analysis.head())


# --- STEP 151: ERROR DATA VALIDATION ---

assert len(m8_error_analysis) == forecast_horizon

assert m8_error_analysis[
    ["actual", "forecast", "error"]
].isna().sum().sum() == 0

print("Error dataset validated.")


# --- STEP 152: LARGEST ABSOLUTE ERRORS ---

largest_errors = (
    m8_error_analysis
    .sort_values("absolute_error", ascending=False)
)

print(largest_errors.head(10))


# --- STEP 153: SMALLEST ABSOLUTE ERRORS ---

smallest_errors = (
    m8_error_analysis
    .sort_values("absolute_error")
)

print(smallest_errors.head(10))


# --- STEP 154: MEAN ERROR ---

mean_error = m8_error_analysis["error"].mean()

print("Mean error:", mean_error)


# --- STEP 155: MEDIAN ERROR ---

median_error = m8_error_analysis["error"].median()

print("Median error:", median_error)


# --- STEP 156: ERROR STANDARD DEVIATION ---

error_std = m8_error_analysis["error"].std()

print("Error standard deviation:", error_std)


# --- STEP 157: ERROR DISTRIBUTION ---

print(m8_error_analysis["error"].describe())



# --- STEP 158: MAXIMUM UNDERFORECAST ---

maximum_underforecast_error = (
    m8_error_analysis["error"].max()
)

print("Maximum underforecast error:", maximum_underforecast_error)


# --- STEP 159: MAXIMUM OVERFORECAST ---

maximum_overforecast_error = (
    m8_error_analysis["error"].min()
)

print("Maximum overforecast error:", maximum_overforecast_error)


# --- STEP 160: UNDERFORECAST OBSERVATION ---

underforecast_observation = (
    m8_error_analysis.loc[
        m8_error_analysis["error"].idxmax()
    ]
)

print(underforecast_observation)


# --- STEP 161: OVERFORECAST OBSERVATION ---

overforecast_observation = (
    m8_error_analysis.loc[
        m8_error_analysis["error"].idxmin()
    ]
)

print(overforecast_observation)


# --- STEP 162: WEEKDAY IDENTIFICATION ---

m8_error_analysis["weekday"] = (
    m8_error_analysis.index.day_name()
)

print(m8_error_analysis.head())


# --- STEP 163: WEEKDAY ERROR ANALYSIS ---

weekday_errors = (
    m8_error_analysis
    .groupby("weekday")["error"]
    .agg(["mean", "median", "std", "count"])
)

print(weekday_errors)


# --- STEP 164: WEEKDAY ABSOLUTE ERROR ---

weekday_absolute_errors = (
    m8_error_analysis
    .groupby("weekday")["absolute_error"]
    .mean()
)

print(weekday_absolute_errors)


# --- STEP 165: ACTUAL DEMAND BY WEEKDAY ---

weekday_actual = (
    m8_error_analysis
    .groupby("weekday")["actual"]
    .mean()
)

print(weekday_actual)


# --- STEP 166: FORECAST BY WEEKDAY ---

weekday_forecast = (
    m8_error_analysis
    .groupby("weekday")["forecast"]
    .mean()
)

print(weekday_forecast)


# --- STEP 167: WEEKDAY COMPARISON ---

weekday_comparison = pd.DataFrame(
    {
        "actual_mean": weekday_actual,
        "forecast_mean": weekday_forecast,
        "mean_error": weekday_errors["mean"],
        "mean_absolute_error": weekday_absolute_errors,
    }
)

print(weekday_comparison)


# --- STEP 168: TOTAL ACTUAL DEMAND ---

total_actual = m8_error_analysis["actual"].sum()

print("Total actual demand:", total_actual)


# --- STEP 169: TOTAL FORECAST DEMAND ---

total_forecast = m8_error_analysis["forecast"].sum()

print("Total forecast demand:", total_forecast)


# --- STEP 170: TOTAL DEMAND DEVIATION ---

total_demand_deviation = (
    total_forecast - total_actual
)

print("Total demand deviation:", total_demand_deviation)


# --- STEP 171: AGGREGATE FORECAST DEVIATION ---

aggregate_forecast_deviation_pct = (
    (total_forecast - total_actual)
    / total_actual
    * 100
)

print(
    "Aggregate forecast deviation (%):",
    aggregate_forecast_deviation_pct
)


# --- STEP 172: ERROR TIME SERIES ---

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    m8_error_analysis.index,
    m8_error_analysis["error"]
)

ax.axhline(
    0,
    linestyle="--"
)

ax.set_title("M8: Seasonal Naive Forecast Errors")
ax.set_xlabel("Date")
ax.set_ylabel("Error: Actual - Forecast")

fig.tight_layout()

fig.savefig(
    reports_dir / "m8_error_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --- STEP 173: ERROR DISTRIBUTION PLOT ---

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    m8_error_analysis["error"],
    bins=15
)

ax.axvline(
    0,
    linestyle="--"
)

ax.set_title("M8: Forecast Error Distribution")
ax.set_xlabel("Error")
ax.set_ylabel("Frequency")

fig.tight_layout()

fig.savefig(
    reports_dir / "m8_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --- STEP 174: WEEKDAY ERROR PLOT ---

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday_plot_data = weekday_absolute_errors.reindex(
    weekday_order
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    weekday_plot_data.index,
    weekday_plot_data.values
)

ax.set_title("M8: Mean Absolute Error by Weekday")
ax.set_xlabel("Weekday")
ax.set_ylabel("Mean Absolute Error")

fig.tight_layout()

fig.savefig(
    reports_dir / "m8_weekday_absolute_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --- STEP 175: EXPORT ERROR ANALYSIS ---

m8_error_path = (
    reports_dir / "m8_seasonal_error_analysis.csv"
)

m8_error_analysis.to_csv(m8_error_path)

weekday_comparison_path = (
    reports_dir / "m8_weekday_error_summary.csv"
)

weekday_comparison.to_csv(weekday_comparison_path)


# --- STEP 176: VERIFY M8 ARTIFACTS ---

m8_artifacts = [
    reports_dir / "m8_error_series.png",
    reports_dir / "m8_error_distribution.png",
    reports_dir / "m8_weekday_absolute_error.png",
    reports_dir / "m8_seasonal_error_analysis.csv",
    reports_dir / "m8_weekday_error_summary.csv",
]

for artifact in m8_artifacts:

    print(
        artifact.name,
        artifact.exists()
    )




# --- STEP 177: IMPORT STATIONARITY DIAGNOSTIC TOOLS ---

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- STEP 178: VERIFY THE CORRECTED VALIDATION FOLDS ---

assert len(fold_table) == 5, "Expected five chronological validation folds."

assert pd.Timestamp(fold_table.iloc[-1]["validation_end"]) <= pd.Timestamp("2017-07-16"), \
    "Validation folds overlap the final holdout."


# --- STEP 179: ENSURE THE SERIES INDEX IS DATETIME ---

y.index = pd.to_datetime(y.index)

y = y.sort_index()

assert y.index.is_unique
assert y.index.is_monotonic_increasing


# --- STEP 180: DEFINE STATIONARITY TEST FUNCTION ---

def stationarity_tests(series, series_name):

    series = series.dropna()

    adf_result = adfuller(series, autolag="AIC")

    kpss_result = kpss(
        series,
        regression="c",
        nlags="auto"
    )

    return {
        "series": series_name,
        "observations": len(series),
        "adf_statistic": adf_result[0],
        "adf_pvalue": adf_result[1],
        "adf_lags": adf_result[2],
        "kpss_statistic": kpss_result[0],
        "kpss_pvalue": kpss_result[1],
        "kpss_lags": kpss_result[2]
    }


# --- STEP 181: DEFINE STATIONARITY INTERPRETATION ---

def interpret_stationarity(adf_pvalue, kpss_pvalue):

    adf_stationary = adf_pvalue < 0.05

    kpss_stationary = kpss_pvalue > 0.05

    if adf_stationary and kpss_stationary:
        return "Both tests support stationarity"

    elif not adf_stationary and not kpss_stationary:
        return "Both tests suggest nonstationarity"

    else:
        return "Tests disagree - further investigation required"


# --- STEP 182: PREPARE STATIONARITY RESULTS STORAGE ---

stationarity_records = []


# --- STEP 183: ITERATE THROUGH CHRONOLOGICAL VALIDATION FOLDS ---

for fold_number, fold in fold_table.iterrows():

    train_end = pd.Timestamp(fold["train_end"])

    train_fold = y.loc[:train_end].copy()

    train_fold = train_fold.dropna()

    print(f"\nFold {fold_number + 1}")

    print(f"Training end: {train_end}")

    print(f"Training observations: {len(train_fold)}")


# --- STEP 184: TEST ORIGINAL SERIES FOR STATIONARITY ---

    original_results = stationarity_tests(
        train_fold,
        f"Fold {fold_number + 1} Original"
    )

    original_results["fold"] = fold_number + 1

    original_results["differencing_order"] = 0

    original_results["interpretation"] = interpret_stationarity(
        original_results["adf_pvalue"],
        original_results["kpss_pvalue"]
    )

    stationarity_records.append(original_results)


# --- STEP 185: COMPUTE FIRST DIFFERENCE ---

    first_difference = train_fold.diff().dropna()


# --- STEP 186: TEST FIRST DIFFERENCE ---

    first_diff_results = stationarity_tests(
        first_difference,
        f"Fold {fold_number + 1} First Difference"
    )

    first_diff_results["fold"] = fold_number + 1

    first_diff_results["differencing_order"] = 1

    first_diff_results["interpretation"] = interpret_stationarity(
        first_diff_results["adf_pvalue"],
        first_diff_results["kpss_pvalue"]
    )

    stationarity_records.append(first_diff_results)


# --- STEP 187: COMPUTE SECOND DIFFERENCE FOR DIAGNOSTIC COMPARISON ---

    second_difference = train_fold.diff().diff().dropna()


# --- STEP 188: TEST SECOND DIFFERENCE ---

    second_diff_results = stationarity_tests(
        second_difference,
        f"Fold {fold_number + 1} Second Difference"
    )

    second_diff_results["fold"] = fold_number + 1

    second_diff_results["differencing_order"] = 2

    second_diff_results["interpretation"] = interpret_stationarity(
        second_diff_results["adf_pvalue"],
        second_diff_results["kpss_pvalue"]
    )

    stationarity_records.append(second_diff_results)


# --- STEP 189: COMPILE STATIONARITY RESULTS ---

stationarity_table = pd.DataFrame(stationarity_records)


# --- STEP 190: DISPLAY STATIONARITY RESULTS ---

print("\nStationarity Diagnostic Results:")

print(
    stationarity_table[
        [
            "fold",
            "series",
            "differencing_order",
            "adf_pvalue",
            "kpss_pvalue",
            "interpretation"
        ]
    ].to_string(index=False)
)


# --- STEP 191: EXPORT STATIONARITY RESULTS ---

stationarity_table.to_csv(
    reports_dir / "m9_stationarity_diagnostics.csv",
    index=False
)


# --- STEP 192: SELECT THE LAST DEVELOPMENT TRAINING WINDOW ---

last_fold = fold_table.iloc[-1]

last_train_end = pd.Timestamp(last_fold["train_end"])

identification_series = y.loc[:last_train_end].copy()


# --- STEP 193: REMOVE MISSING OBSERVATIONS FOR ACF/PACF ANALYSIS ---

identification_series = identification_series.dropna()


# --- STEP 194: COMPUTE FIRST DIFFERENCE FOR ACF/PACF ---

identification_first_diff = identification_series.diff().dropna()


# --- STEP 195: PLOT ORIGINAL DEVELOPMENT TRAINING SERIES ---

plt.figure(figsize=(14, 5))

plt.plot(
    identification_series.index,
    identification_series.values
)

plt.title("M9 - Development Training Series for Model Identification")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_identification_series.png",
    dpi=300
)

plt.show()


# --- STEP 196: PLOT FIRST DIFFERENCE ---

plt.figure(figsize=(14, 5))

plt.plot(
    identification_first_diff.index,
    identification_first_diff.values
)

plt.title("M9 - First Difference of Development Training Series")

plt.xlabel("Date")

plt.ylabel("Change in Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_first_difference.png",
    dpi=300
)

plt.show()


# --- STEP 197: PLOT ACF OF ORIGINAL SERIES ---

plt.figure(figsize=(12, 5))

plot_acf(
    identification_series,
    lags=40,
    zero=False
)

plt.title("M9 - ACF of Original Development Training Series")

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_acf_original.png",
    dpi=300
)

plt.show()


# --- STEP 198: PLOT PACF OF ORIGINAL SERIES ---

plt.figure(figsize=(12, 5))

plot_pacf(
    identification_series,
    lags=40,
    method="ywm"
)

plt.title("M9 - PACF of Original Development Training Series")

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_pacf_original.png",
    dpi=300
)

plt.show()


# --- STEP 199: PLOT ACF OF FIRST DIFFERENCE ---

plt.figure(figsize=(12, 5))

plot_acf(
    identification_first_diff,
    lags=40,
    zero=False
)

plt.title("M9 - ACF of First Difference")

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_acf_first_difference.png",
    dpi=300
)

plt.show()


# --- STEP 200: PLOT PACF OF FIRST DIFFERENCE ---

plt.figure(figsize=(12, 5))

plot_pacf(
    identification_first_diff,
    lags=40,
    method="ywm"
)

plt.title("M9 - PACF of First Difference")

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_pacf_first_difference.png",
    dpi=300
)

plt.show()


# --- STEP 201: COMPUTE SECOND DIFFERENCE FOR VISUAL COMPARISON ---

identification_second_diff = identification_first_diff.diff().dropna()


# --- STEP 202: PLOT SECOND DIFFERENCE ---

plt.figure(figsize=(14, 5))

plt.plot(
    identification_second_diff.index,
    identification_second_diff.values
)

plt.title("M9 - Second Difference of Development Training Series")

plt.xlabel("Date")

plt.ylabel("Second Difference")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    reports_dir / "m9_second_difference.png",
    dpi=300
)

plt.show()


# --- STEP 203: COMPARE DIFFERENCING DISTRIBUTIONS ---

differencing_comparison = pd.DataFrame({
    "original": identification_series,
    "first_difference": identification_series.diff(),
    "second_difference": identification_series.diff().diff()
})

print("\nDifferencing Summary:")

print(
    differencing_comparison.describe().round(3)
)


# --- STEP 204: EXPORT DIFFERENCING SUMMARY ---

differencing_comparison.describe().to_csv(
    reports_dir / "m9_differencing_summary.csv"
)


# --- STEP 205: EXPORT IDENTIFICATION SERIES ---

identification_data = pd.DataFrame({
    "original": identification_series,
    "first_difference": identification_series.diff(),
    "second_difference": identification_series.diff().diff()
})

identification_data.to_csv(
    reports_dir / "m9_identification_data.csv"
)


# --- STEP 206: VERIFY GENERATED DIAGNOSTIC ARTIFACTS ---

print("\nM9 Stationarity and Differencing Diagnostics Complete.")

print("\nGenerated Files:")

for file in sorted(reports_dir.glob("m9_*")):

    print(file.name)

# ============================================================
# M10: ARIMA CANDIDATE EVALUATION
# ============================================================

# --- STEP 207: IMPORT MODELING LIBRARIES ---
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# --- STEP 208: VALIDATE EXPERIMENT INPUTS ---
assert len(fold_table) == 5
assert fold_table["validation_end"].max() <= pd.Timestamp("2017-07-16")


# --- STEP 209: DEFINE ARIMA CANDIDATES ---
arima_candidates = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
]

assert all(order[1] == 1 for order in arima_candidates)

print("ARIMA candidates:")
for order in arima_candidates:
    print(order)

# --- STEP 210: DEFINE FORECAST HORIZON ---
arima_forecast_horizon = 30

# --- STEP 211: INITIALIZE EXPERIMENT RECORDS ---
arima_fold_results = []
arima_forecasts = []
arima_failures = []

# --- STEP 212: RUN EXPANDING-WINDOW VALIDATION ---
for _, fold in fold_table.iterrows():

    fold_number = int(fold["fold"])

    train_end = pd.Timestamp(fold["train_end"])
    validation_start = pd.Timestamp(fold["validation_start"])
    validation_end = pd.Timestamp(fold["validation_end"])


# --- STEP 212: PREPARE DAILY TRAINING AND VALIDATION DATA ---
    train_fold = (
     y.loc[:train_end]
     .asfreq("D")
     .fillna(0)
    )

    validation_fold = y.loc[validation_start:validation_end].asfreq("D")

# Validate daily calendar structure without accessing freq/freqstr
    expected_train_index = pd.date_range(
     start=train_fold.index[0],
     end=train_fold.index[-1],
     freq="D"
    )

    expected_validation_index = pd.date_range(
     start=validation_start,
     end=validation_end,
     freq="D"
    )

    assert train_fold.index.equals(expected_train_index)
    assert validation_fold.index.equals(expected_validation_index)
    assert len(validation_fold) == arima_forecast_horizon
    assert train_fold.index.max() < validation_fold.index.min()

    print(f"\n{'=' * 60}")
    print(f"Fold {fold_number}")
    print(f"Training end: {train_fold.index[-1]}")
    print(f"Training observations: {len(train_fold)}")
    print(f"Validation start: {validation_fold.index[0]}")
    print(f"Validation end: {validation_fold.index[-1]}")
    print(f"{'=' * 60}")

    # --- STEP 213: FIT EACH CANDIDATE ---
    for order in arima_candidates:

        model_name = f"ARIMA{order}"

        try:

            with warnings.catch_warnings(record=True) as model_warnings:
                warnings.simplefilter("always")

                model = ARIMA(
                    train_fold,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fitted_model = model.fit()

                forecast_values = np.asarray(
                    fitted_model.forecast(
                        steps=arima_forecast_horizon
                    ),
                    dtype=float
                )

            # --- STEP 214: VALIDATE FORECAST OUTPUT ---
            assert len(forecast_values) == len(validation_fold)
            assert np.isfinite(forecast_values).all()

            # --- STEP 215: CALCULATE FOLD METRICS ---
            actual_values = validation_fold.to_numpy(dtype=float)

            mae = mean_absolute_error(
                actual_values,
                forecast_values
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual_values,
                    forecast_values
                )
            )

            mean_error = np.mean(
                actual_values - forecast_values
            )

            # --- STEP 216: RECORD MODEL RESULTS ---
            arima_fold_results.append({
                "fold": fold_number,
                "model": model_name,
                "order": str(order),
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "train_length": len(train_fold),
                "validation_length": len(validation_fold),
                "mae": mae,
                "rmse": rmse,
                "mean_error": mean_error,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "converged": fitted_model.mle_retvals.get(
                    "converged", None
                ),
                "warning_count": len(model_warnings)
            })

            # --- STEP 217: RECORD FORECAST VALUES ---
            for date, actual, forecast in zip(
                validation_fold.index,
                actual_values,
                forecast_values
            ):

                arima_forecasts.append({
                    "fold": fold_number,
                    "model": model_name,
                    "order": str(order),
                    "date": date,
                    "actual": actual,
                    "forecast": forecast,
                    "error": actual - forecast,
                    "absolute_error": abs(actual - forecast),
                    "squared_error": (actual - forecast) ** 2
                })

            print(
                f"{model_name} | "
                f"MAE: {mae:.2f} | "
                f"RMSE: {rmse:.2f} | "
                f"Mean Error: {mean_error:.2f} | "
                f"Converged: {fitted_model.mle_retvals.get('converged')}"
            )

        except Exception as exc:

            # --- STEP 218: RECORD FAILED MODEL FITS ---
            arima_failures.append({
                "fold": fold_number,
                "model": model_name,
                "order": str(order),
                "error": repr(exc)
            })

            print(
                f"{model_name} FAILED in Fold {fold_number}: "
                f"{repr(exc)}"
            )

# --- STEP 219: CONVERT RESULTS TO DATAFRAMES ---
arima_fold_metrics = pd.DataFrame(arima_fold_results)
arima_forecast_df = pd.DataFrame(arima_forecasts)
arima_failure_df = pd.DataFrame(arima_failures)

assert not arima_fold_metrics.empty
assert not arima_forecast_df.empty

# --- STEP 220: VALIDATE EXPERIMENT COMPLETENESS ---
expected_fits = len(fold_table) * len(arima_candidates)

successful_fits = len(arima_fold_metrics)
failed_fits = len(arima_failure_df)

print("\nExperiment execution summary:")
print("Expected fits:", expected_fits)
print("Successful fits:", successful_fits)
print("Failed fits:", failed_fits)

assert successful_fits + failed_fits == expected_fits

# --- STEP 221: AGGREGATE CROSS-FOLD PERFORMANCE ---
arima_model_summary = (
    arima_fold_metrics
    .groupby(["model", "order"], as_index=False)
    .agg(
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        std_rmse=("rmse", "std"),
        mean_error=("mean_error", "mean"),
        mean_aic=("aic", "mean"),
        mean_bic=("bic", "mean"),
        successful_folds=("fold", "count"),
        converged_folds=("converged", "sum"),
        total_warnings=("warning_count", "sum")
    )
)

# --- STEP 222: RANK CANDIDATES ---
arima_model_summary = (
    arima_model_summary
    .sort_values(
        ["mean_mae", "mean_rmse"],
        ascending=True
    )
    .reset_index(drop=True)
)

arima_model_summary["mae_rank"] = (
    arima_model_summary["mean_mae"].rank(
        method="min",
        ascending=True
    )
)

# --- STEP 223: DISPLAY FOLD RESULTS ---
print("\nARIMA fold-level results:")
print(
    arima_fold_metrics
    .sort_values(["model", "fold"])
    .to_string(index=False)
)

# --- STEP 224: DISPLAY AGGREGATED RESULTS ---
print("\nARIMA aggregated candidate ranking:")
print(arima_model_summary.to_string(index=False))

# --- STEP 225: DISPLAY FAILED FITS ---
if not arima_failure_df.empty:

    print("\nFailed model fits:")
    print(arima_failure_df.to_string(index=False))

else:

    print("\nNo model fitting failures.")

# --- STEP 226: EXPORT FOLD-LEVEL RESULTS ---
arima_fold_metrics.to_csv(
    reports_dir / "m10_arima_fold_metrics.csv",
    index=False
)

# --- STEP 227: EXPORT FORECAST VALUES ---
arima_forecast_df.to_csv(
    reports_dir / "m10_arima_forecasts.csv",
    index=False
)

# --- STEP 228: EXPORT AGGREGATED MODEL RANKING ---
arima_model_summary.to_csv(
    reports_dir / "m10_arima_model_comparison.csv",
    index=False
)

# --- STEP 229: EXPORT FAILURE LOG ---
arima_failure_df.to_csv(
    reports_dir / "m10_arima_failures.csv",
    index=False
)

# --- STEP 230: VERIFY EXPORTED RESULTS ---
assert (
    reports_dir / "m10_arima_fold_metrics.csv"
).exists()

assert (
    reports_dir / "m10_arima_forecasts.csv"
).exists()

assert (
    reports_dir / "m10_arima_model_comparison.csv"
).exists()

print("\nM10 initial ARIMA experiment completed.")
print("Reports saved in:", reports_dir)

# ============================================================
# M10.1: SEASONAL NAIVE ON THE SAME VALIDATION FOLDS
# ============================================================

# --- STEP 231: INITIALIZE SEASONAL BASELINE RESULTS ---
seasonal_cv_results = []

seasonal_period = 7
seasonal_horizon = 30

# --- STEP 232: EVALUATE SEASONAL NAIVE ACROSS FOLDS ---
for _, fold in fold_table.iterrows():

    fold_number = int(fold["fold"])

    train_end = pd.Timestamp(fold["train_end"])
    validation_start = pd.Timestamp(fold["validation_start"])
    validation_end = pd.Timestamp(fold["validation_end"])

    train_fold = (
        y.loc[:train_end]
        .asfreq("D")
        .fillna(0)
    )

    validation_fold = (
        y.loc[validation_start:validation_end]
        .asfreq("D")
    )

    # --- STEP 233: GENERATE SEASONAL NAIVE FORECAST ---
    last_season = train_fold.iloc[-seasonal_period:].to_numpy()

    forecast_values = np.tile(
        last_season,
        int(np.ceil(seasonal_horizon / seasonal_period))
    )[:seasonal_horizon]

    actual_values = validation_fold.to_numpy(dtype=float)

    # --- STEP 234: CALCULATE METRICS ---
    mae = mean_absolute_error(
        actual_values,
        forecast_values
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            forecast_values
        )
    )

    mean_error = np.mean(
        actual_values - forecast_values
    )

    # --- STEP 235: RECORD RESULTS ---
    seasonal_cv_results.append({
        "fold": fold_number,
        "model": "SeasonalNaive_7",
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end
    })

    print(
        f"Fold {fold_number} | "
        f"Seasonal Naive MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"Mean Error: {mean_error:.2f}"
    )

# --- STEP 236: CONVERT TO DATAFRAME ---
seasonal_cv_metrics = pd.DataFrame(seasonal_cv_results)

# --- STEP 237: AGGREGATE BASELINE RESULTS ---
seasonal_cv_summary = (
    seasonal_cv_metrics
    .agg({
        "mae": ["mean", "median", "std"],
        "rmse": ["mean", "median", "std"],
        "mean_error": ["mean", "median"]
    })
)

print("\nSeasonal-naive cross-validation summary:")
print(seasonal_cv_summary)

# --- STEP 238: EXPORT BASELINE FOLD RESULTS ---
seasonal_cv_metrics.to_csv(
    reports_dir / "m10_seasonal_naive_cv_metrics.csv",
    index=False
)

# --- STEP 239: EXPORT BASELINE SUMMARY ---
seasonal_cv_summary.to_csv(
    reports_dir / "m10_seasonal_naive_cv_summary.csv"
)

print("\nSeasonal-naive comparison completed.")


# ============================================================
# M11: WEEKLY SARIMA EXPERIMENTS
# Demand Forecasting & Replenishment Engine
# ============================================================

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "store1_grocery1_daily_prepared.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 30
SEASONAL_PERIOD = 7

FOLD_STARTS = [
    "2017-02-17",
    "2017-03-19",
    "2017-04-18",
    "2017-05-18",
    "2017-06-17",
]

FOLD_NAMES = [
    "Fold 1",
    "Fold 2",
    "Fold 3",
    "Fold 4",
    "Fold 5",
]


# ============================================================
# 2. MODEL CANDIDATES
# ============================================================

# (p, d, q), (P, D, Q, s)

MODEL_CANDIDATES = [
    {
        "name": "SARIMA_011_011_7",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 7),
    },
    {
        "name": "SARIMA_110_011_7",
        "order": (1, 1, 0),
        "seasonal_order": (0, 1, 1, 7),
    },
    {
        "name": "SARIMA_011_100_7",
        "order": (0, 1, 1),
        "seasonal_order": (1, 1, 0, 7),
    },
    {
        "name": "SARIMA_111_011_7",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 1, 7),
    },
    {
        "name": "SARIMA_011_001_7",
        "order": (0, 1, 1),
        "seasonal_order": (0, 0, 1, 7),
    },
    {
        "name": "SARIMA_111_111_7",
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 7),
    },
]


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

print("=" * 70)
print("M11: WEEKLY SARIMA EXPERIMENT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df.sort_values("date")
    .set_index("date")
)

# Enforce a complete daily calendar.
# Missing dates are assumed to represent zero sales for this experiment.
df = df.asfreq("D")

if "sales" not in df.columns:
    raise ValueError("The prepared dataset must contain a 'sales' column.")

if df["sales"].isna().any():
    print("\nMissing sales observations detected.")
    print("Applying the zero-sales assumption to missing calendar dates.")
    df["sales"] = df["sales"].fillna(0)

series = df["sales"].astype(float)

if not series.index.equals(
    pd.date_range(
        series.index.min(),
        series.index.max(),
        freq="D"
    )
):
    raise ValueError("The series does not have a complete daily calendar.")

if not np.isfinite(series.values).all():
    raise ValueError("Series contains infinite or invalid values.")

if (series < 0).any():
    raise ValueError("Negative sales values detected.")

print("\nDataset information:")
print(f"Observations: {len(series)}")
print(f"Start date: {series.index.min()}")
print(f"End date: {series.index.max()}")
print(f"Frequency: {series.index.freq}")


# ============================================================
# 4. BUILD CHRONOLOGICAL VALIDATION FOLDS
# ============================================================

folds = []

for fold_name, validation_start in zip(
    FOLD_NAMES,
    FOLD_STARTS
):

    validation_start = pd.Timestamp(validation_start)

    validation_end = validation_start + pd.Timedelta(
        days=HORIZON - 1
    )

    training_end = validation_start - pd.Timedelta(days=1)

    train = series.loc[:training_end]

    validation = series.loc[
        validation_start:validation_end
    ]

    expected_validation_dates = pd.date_range(
        validation_start,
        validation_end,
        freq="D"
    )

    if len(validation) != HORIZON:
        raise ValueError(
            f"{fold_name}: Expected {HORIZON} validation "
            f"observations, found {len(validation)}."
        )

    if not validation.index.equals(expected_validation_dates):
        raise ValueError(
            f"{fold_name}: Validation dates are not contiguous."
        )

    if train.index.max() >= validation.index.min():
        raise ValueError(
            f"{fold_name}: Training data overlaps validation."
        )

    folds.append(
        {
            "fold": fold_name,
            "training": train,
            "validation": validation,
            "training_end": train.index.max(),
            "validation_start": validation.index.min(),
            "validation_end": validation.index.max(),
        }
    )


# ============================================================
# 5. PRINT FOLD STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION STRUCTURE")
print("=" * 70)

fold_structure = []

for fold in folds:

    train = fold["training"]
    validation = fold["validation"]

    fold_structure.append(
        {
            "fold": fold["fold"],
            "training_observations": len(train),
            "training_start": train.index.min(),
            "training_end": train.index.max(),
            "validation_observations": len(validation),
            "validation_start": validation.index.min(),
            "validation_end": validation.index.max(),
        }
    )

    print(f"\n{fold['fold']}")
    print(f"Training observations: {len(train)}")
    print(f"Training end: {train.index.max()}")
    print(f"Validation start: {validation.index.min()}")
    print(f"Validation end: {validation.index.max()}")

fold_structure_df = pd.DataFrame(fold_structure)

fold_structure_df.to_csv(
    REPORTS_DIR / "m11_fold_structure.csv",
    index=False
)


# ============================================================
# 6. METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    errors = actual - forecast

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    mean_error = np.mean(errors)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "mean_error": mean_error,
    }


# ============================================================
# 7. SEASONAL NAIVE BENCHMARK
# ============================================================

def seasonal_naive_forecast(train, horizon, period=7):

    if len(train) < period:
        raise ValueError(
            "Training series is shorter than the seasonal period."
        )

    last_season = train.iloc[-period:].values

    repetitions = int(np.ceil(horizon / period))

    forecast_values = np.tile(
        last_season,
        repetitions
    )[:horizon]

    return forecast_values


# ============================================================
# 8. PREVIOUS ARIMA BENCHMARK
# ============================================================

def fit_arima_011(train, horizon):

    model = SARIMAX(
        train,
        order=(0, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False,
        maxiter=200
    )

    forecast = fitted_model.forecast(
        steps=horizon
    )

    return np.asarray(forecast, dtype=float)


# ============================================================
# 9. SARIMA EXPERIMENT
# ============================================================

sarima_metrics = []

sarima_forecasts = []

sarima_failures = []

sarima_convergence = []

total_fits = len(folds) * len(MODEL_CANDIDATES)

fit_counter = 0

print("\n" + "=" * 70)
print("FITTING SARIMA CANDIDATES")
print("=" * 70)

for candidate in MODEL_CANDIDATES:

    model_name = candidate["name"]

    order = candidate["order"]

    seasonal_order = candidate["seasonal_order"]

    print("\n" + "-" * 70)
    print(f"MODEL: {model_name}")
    print(f"Order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    print("-" * 70)

    for fold in folds:

        fold_name = fold["fold"]

        train = fold["training"]

        validation = fold["validation"]

        fit_counter += 1

        print(
            f"\n[{fit_counter}/{total_fits}] "
            f"{model_name} | {fold_name}"
        )

        start_time = time.perf_counter()

        try:

            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            with warnings.catch_warnings(record=True) as caught_warnings:

                warnings.simplefilter("always")

                fitted_model = model.fit(
                    disp=False,
                    maxiter=200
                )

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            forecast = fitted_model.forecast(
                steps=HORIZON
            )

            forecast = np.asarray(
                forecast,
                dtype=float
            )

            if len(forecast) != HORIZON:

                raise ValueError(
                    f"Expected {HORIZON} forecast values."
                )

            if not np.isfinite(forecast).all():

                raise ValueError(
                    "Forecast contains non-finite values."
                )

            actual = validation.values

            metrics = calculate_metrics(
                actual,
                forecast
            )

            converged = fitted_model.mle_retvals.get(
                "converged",
                None
            )

            sarima_metrics.append(
                {
                    "model": model_name,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "fold": fold_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "mean_error": metrics["mean_error"],
                    "converged": converged,
                    "fit_seconds": elapsed_seconds,
                    "warning_count": len(caught_warnings),
                }
            )

            sarima_convergence.append(
                {
                    "model": model_name,
                    "fold": fold_name,
                    "converged": converged,
                    "fit_seconds": elapsed_seconds,
                    "warning_count": len(caught_warnings),
                    "warnings": "; ".join(
                        str(w.message)
                        for w in caught_warnings
                    ),
                }
            )

            for date, actual_value, forecast_value in zip(
                validation.index,
                actual,
                forecast
            ):

                sarima_forecasts.append(
                    {
                        "model": model_name,
                        "fold": fold_name,
                        "date": date,
                        "actual": actual_value,
                        "forecast": forecast_value,
                        "error": (
                            actual_value - forecast_value
                        ),
                    }
                )

            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(
                f"Mean error: "
                f"{metrics['mean_error']:.2f}"
            )
            print(f"Converged: {converged}")
            print(
                f"Fit time: "
                f"{elapsed_seconds:.2f} seconds"
            )

            if caught_warnings:

                print(
                    f"Warnings: {len(caught_warnings)}"
                )

        except Exception as exc:

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            sarima_failures.append(
                {
                    "model": model_name,
                    "fold": fold_name,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "error": str(exc),
                    "fit_seconds": elapsed_seconds,
                }
            )

            print(f"FAILED: {exc}")


# ============================================================
# 10. SAVE SARIMA RESULTS
# ============================================================

metrics_df = pd.DataFrame(sarima_metrics)

forecasts_df = pd.DataFrame(sarima_forecasts)

failures_df = pd.DataFrame(sarima_failures)

convergence_df = pd.DataFrame(sarima_convergence)

metrics_df.to_csv(
    REPORTS_DIR / "m11_sarima_fold_metrics.csv",
    index=False
)

forecasts_df.to_csv(
    REPORTS_DIR / "m11_sarima_forecasts.csv",
    index=False
)

failures_df.to_csv(
    REPORTS_DIR / "m11_sarima_failures.csv",
    index=False
)

convergence_df.to_csv(
    REPORTS_DIR / "m11_sarima_convergence.csv",
    index=False
)


# ============================================================
# 11. AGGREGATE SARIMA PERFORMANCE
# ============================================================

if metrics_df.empty:

    raise RuntimeError(
        "No SARIMA model completed successfully."
    )

sarima_summary = (
    metrics_df
    .groupby("model")
    .agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"),
        mean_error=("mean_error", "mean"),
        converged_folds=("converged", "sum"),
        successful_folds=("fold", "count"),
        mean_fit_seconds=("fit_seconds", "mean"),
    )
    .reset_index()
)

sarima_summary = sarima_summary.sort_values(
    "mean_MAE"
)

sarima_summary.to_csv(
    REPORTS_DIR / "m11_sarima_model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("SARIMA MODEL RANKING")
print("=" * 70)

print(
    sarima_summary.to_string(
        index=False
    )
)


# ============================================================
# 12. EVALUATE BASELINES ON IDENTICAL FOLDS
# ============================================================

baseline_metrics = []

baseline_forecasts = []

for fold in folds:

    fold_name = fold["fold"]

    train = fold["training"]

    validation = fold["validation"]

    actual = validation.values

    # Seasonal naive
    seasonal_forecast = seasonal_naive_forecast(
        train,
        HORIZON,
        SEASONAL_PERIOD
    )

    seasonal_metrics = calculate_metrics(
        actual,
        seasonal_forecast
    )

    baseline_metrics.append(
        {
            "model": "Seasonal_Naive_7",
            "fold": fold_name,
            **seasonal_metrics,
        }
    )

    # Nonseasonal ARIMA(0,1,1)
    try:

        arima_forecast = fit_arima_011(
            train,
            HORIZON
        )

        arima_metrics = calculate_metrics(
            actual,
            arima_forecast
        )

        baseline_metrics.append(
            {
                "model": "ARIMA_011",
                "fold": fold_name,
                **arima_metrics,
            }
        )

    except Exception as exc:

        print(
            f"\nARIMA baseline failed in "
            f"{fold_name}: {exc}"
        )

        arima_forecast = None

    # Save seasonal naive forecasts
    for date, actual_value, forecast_value in zip(
        validation.index,
        actual,
        seasonal_forecast
    ):

        baseline_forecasts.append(
            {
                "model": "Seasonal_Naive_7",
                "fold": fold_name,
                "date": date,
                "actual": actual_value,
                "forecast": forecast_value,
                "error": actual_value - forecast_value,
            }
        )

    # Save ARIMA forecasts if successful
    if arima_forecast is not None:

        for date, actual_value, forecast_value in zip(
            validation.index,
            actual,
            arima_forecast
        ):

            baseline_forecasts.append(
                {
                    "model": "ARIMA_011",
                    "fold": fold_name,
                    "date": date,
                    "actual": actual_value,
                    "forecast": forecast_value,
                    "error": actual_value - forecast_value,
                }
            )


baseline_metrics_df = pd.DataFrame(
    baseline_metrics
)

baseline_forecasts_df = pd.DataFrame(
    baseline_forecasts
)

baseline_metrics_df.to_csv(
    REPORTS_DIR / "m11_baseline_fold_metrics.csv",
    index=False
)

baseline_forecasts_df.to_csv(
    REPORTS_DIR / "m11_baseline_forecasts.csv",
    index=False
)


# ============================================================
# 13. COMBINED MODEL COMPARISON
# ============================================================

sarima_comparison = metrics_df[
    [
        "model",
        "fold",
        "MAE",
        "RMSE",
        "mean_error",
    ]
].copy()

baseline_comparison = baseline_metrics_df[
    [
        "model",
        "fold",
        "MAE",
        "RMSE",
        "mean_error",
    ]
].copy()

combined_metrics = pd.concat(
    [
        sarima_comparison,
        baseline_comparison,
    ],
    ignore_index=True
)

combined_summary = (
    combined_metrics
    .groupby("model")
    .agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        std_MAE=("MAE", "std"),
        mean_RMSE=("RMSE", "mean"),
        mean_error=("mean_error", "mean"),
        folds=("fold", "count"),
    )
    .reset_index()
)

combined_summary = combined_summary.sort_values(
    "mean_MAE"
)

combined_summary.to_csv(
    REPORTS_DIR / "m11_combined_model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL COMPARISON: SARIMA AND BASELINES")
print("=" * 70)

print(
    combined_summary.to_string(
        index=False
    )
)


# ============================================================
# 14. VISUALIZE MODEL RANKING
# ============================================================

plt.figure(figsize=(12, 8))

plt.barh(
    combined_summary["model"],
    combined_summary["mean_MAE"]
)

plt.xlabel("Mean MAE across validation folds")

plt.ylabel("Model")

plt.title(
    "M11: SARIMA and Baseline Model Comparison"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "m11_model_ranking.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. VISUALIZE FOLD-LEVEL MAE
# ============================================================

plt.figure(figsize=(14, 8))

for model_name in combined_metrics["model"].unique():

    model_data = combined_metrics[
        combined_metrics["model"] == model_name
    ]

    plt.plot(
        model_data["fold"],
        model_data["MAE"],
        marker="o",
        label=model_name
    )

plt.xlabel("Validation fold")

plt.ylabel("MAE")

plt.title(
    "M11: Model Performance Across Validation Folds"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "m11_fold_performance.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("M11 COMPLETED")
print("=" * 70)

print(f"SARIMA fits attempted: {total_fits}")

print(
    f"SARIMA fits successful: "
    f"{len(metrics_df)}"
)

print(
    f"SARIMA failures: "
    f"{len(failures_df)}"
)

print("\nBest SARIMA candidate:")

print(
    sarima_summary.iloc[0].to_string()
)

print("\nOverall best model from tested candidates:")

print(
    combined_summary.iloc[0].to_string()
)

print("\nReports saved in:")

print(REPORTS_DIR)
