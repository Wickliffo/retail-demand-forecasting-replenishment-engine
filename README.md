# Demand Forecasting & Replenishment Optimization Engine

**From retail demand signals to statistically validated forecasts and constrained inventory decisions.**

A Python-based time-series forecasting and decision-engineering project investigating how retail demand can be modelled, evaluated and translated into replenishment decisions.

The project combines statistical forecasting, chronological backtesting, exogenous-variable analysis, forecast error evaluation and, in subsequent stages, inventory policy modelling and constrained optimization.

> **Project status:** Stages 1–5 completed.Inventory policy design and replenishment optimization are next.

---

## 1. The Problem

Retail demand is not constant.

Sales fluctuate with weekly seasonality, calendar effects, public holidays, local events, promotions and other factors. A forecasting system that ignores these patterns can produce unreliable demand estimates.

However, producing a forecast is only part of the problem.

An operational replenishment system also needs to answer:

- How much demand should be expected over the supplier lead time?
- How uncertain is that estimate?
- How much safety stock is appropriate?
- When should inventory be reordered?
- How should limited purchasing budgets and storage capacity affect order quantities?

A simple historical average or an unvalidated forecasting model cannot adequately answer these questions.

**This project approaches the problem in stages: establish reliable demand data, build statistical forecasting benchmarks, evaluate model improvements through chronological validation, quantify uncertainty, and eventually formulate replenishment as a constrained decision problem.**

## 2. Project Objectives

- Engineer a consistent daily demand time series from retail transaction data.
- Establish naive and seasonal-naive forecasting benchmarks.
- Investigate stationarity, differencing, autocorrelation and seasonal structure.
- Develop and evaluate ARIMA and seasonal ARIMA models.
- Investigate exogenous variables using the SARIMAX modelling framework.
- Evaluate forecasting performance using expanding-window validation.
- Diagnose forecast errors and model residuals.
- Investigate the contribution and limitations of calendar, event and promotion features.
- Develop a statistical foundation for safety stock and reorder-point calculations.
- Build toward constrained replenishment optimization and inventory backtesting.

## 3. Dataset and Scope

The initial experiments use the **Corporación Favorita grocery retail dataset**, which represents retail operations in Ecuador.

The selected forecasting series is:

| Attribute | Selection |
|---|---|
| Store | Store 1 |
| Product family | GROCERY I |
| Frequency | Daily |
| Historical period | January 2013 – August 2017 |
| Forecast horizon | 30 days |
| Validation strategy | Five expanding-window folds |

The selected series contains 1,688 calendar observations after calendar alignment.

The dataset includes sales, store metadata, holidays and events, promotions, transactions and other contextual information.

**Scope clarification:** This is an Ecuadorian retail dataset. The project does not claim that its results represent Kenyan retail demand or that the models have been validated in Kenyan operations.

Raw datasets are excluded from version control.

## 4. Technical Approach

### 4.1 Data Engineering and Time-Series Preparation

Before fitting forecasting models, the selected retail series was investigated and prepared as a consistent daily time series.

Work included:

- Filtering the required store and product family.
- Parsing and validating dates.
- Sorting observations chronologically.
- Aligning observations to a daily calendar.
- Investigating missing dates and zero-sales observations.
- Examining demand distributions and unusual observations.
- Analysing weekly and monthly demand patterns.
- Preparing reusable analytical datasets.

This stage matters because forecasting quality depends on the integrity and interpretation of the underlying time index.

A missing observation is not automatically equivalent to zero demand. The distinction needs to be investigated before model fitting.

### 4.2 Statistical Forecasting: ARIMA and SARIMA

The forecasting investigation progressed from simple benchmarks to statistical time-series models.

**ARIMA — Autoregressive Integrated Moving Average**

ARIMA models represent demand using autoregressive relationships, differencing and moving-average error terms.

The model is expressed as:

\[
ARIMA(p,d,q)
\]

Where:

- \(p\): autoregressive order.
- \(d\): differencing order.
- \(q\): moving-average order.

The investigation considered:

- Time-series stationarity.
- First and seasonal differencing.
- ACF and PACF analysis.
- Candidate model identification.
- Model fitting and convergence.
- Information criteria.
- Forecast error evaluation.

A non-seasonal ARIMA candidate was evaluated against simpler benchmarks rather than being assumed superior because it was statistically more complex.

**SARIMA — Seasonal ARIMA**

Retail demand frequently contains recurring patterns that non-seasonal ARIMA models do not explicitly represent.

SARIMA extends ARIMA with seasonal autoregressive, differencing and moving-average components:

\[
SARIMA(p,d,q)(P,D,Q)_s
\]

Where \(s\) represents the seasonal period.

For this daily series, a weekly seasonal period of seven observations was investigated.

The selected configuration was:

\[
\boxed{SARIMA(1,1,1)(0,1,1)_7}
\]

This model was selected through empirical validation, not by relying exclusively on AIC or BIC.

### 4.3 SARIMAX and Exogenous Feature Evaluation

A forecasting model can sometimes improve when additional information explains demand variation beyond the historical sales sequence.

This motivated the investigation of exogenous variables using the SARIMAX framework.

Conceptually:

\[
y_t = \text{ARIMA dynamics} + \beta X_t + \epsilon_t
\]

Where:

- \(y_t\): observed demand.
- \(X_t\): external explanatory variables.
- \(\beta\): estimated feature coefficients.
- \(\epsilon_t\): unexplained error.

The investigation considered:

- Weekday indicators.
- Month indicators.
- Applicable holidays and events.
- Promotion intensity.
- Combinations of calendar, event and promotion features.

The objective was not simply to add more variables. It was to determine whether additional information improved out-of-sample forecasting performance.

A critical consideration was **feature availability at forecast time**.

A promotion variable observed historically is not automatically available for a future forecasting horizon. Retrospective promotion experiments were therefore distinguished from configurations that could be used operationally with known future inputs.

### 4.4 Chronological Validation and Backtesting

Random train-test splitting is inappropriate for this forecasting task because it can allow future observations to influence model development.

Instead, the project used expanding-window validation.

Each fold trains on historical observations and evaluates predictions on a subsequent 30-day period.

The five validation windows were:

| Fold | Validation period |
|---|---|
| 1 | 17 Feb – 18 Mar 2017 |
| 2 | 19 Mar – 17 Apr 2017 |
| 3 | 18 Apr – 17 May 2017 |
| 4 | 18 May – 16 Jun 2017 |
| 5 | 17 Jun – 16 Jul 2017 |

This provides a more realistic evaluation of how forecasting models behave when predicting future demand from information available at the training cutoff.

Model performance was assessed using:

**Mean Absolute Error (MAE)**

\[
MAE = \frac{1}{n}\sum_{t=1}^{n}|y_t-\hat y_t|
\]

MAE provides an interpretable measure of average absolute forecast error.

**Root Mean Squared Error (RMSE)**

\[
RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t-\hat y_t)^2}
\]

RMSE penalizes larger forecasting errors more heavily.

**Forecast bias**

\[
Bias = \frac{1}{n}\sum_{t=1}^{n}(\hat y_t-y_t)
\]

Bias helps identify systematic overforecasting or underforecasting.

The evaluation also considered fold-level variation, convergence, failed fits and residual diagnostics.

## 5. Results

### Stage 2 — Forecasting Model Evaluation

The selected seasonal SARIMA model achieved:

| Metric | Result |
|---|---:|
| Mean fold MAE | 318.61 |
| Mean fold RMSE | 513.50 |
| Validation folds | 5 |
| Forecast horizon | 30 days |

The weekly seasonal-naive benchmark achieved a mean fold MAE of approximately 318.93.

This comparison is important: a more sophisticated statistical model should demonstrate measurable improvement over a simple seasonal benchmark before being considered useful.

### Stage 3 — Exogenous Feature Investigation

Seven forecasting configurations were evaluated over five chronological folds.

| Model | Configuration | Mean MAE | Mean RMSE |
|---|---|---:|---:|
| M0 | SARIMA baseline | 318.61 | 513.50 |
| M1 | Weekday features | 318.62 | 513.51 |
| M2 | Calendar features | 527.28 | 705.84 |
| M3 | Event features | **303.68** | **423.59** |
| M4 | Calendar + events | 355.53 | 479.73 |
| M5 | Retrospective promotions | 368.31 | 531.16 |
| M6 | Calendar + events + retrospective promotions | 378.64 | 506.00 |

**Best experimental configuration: M3_EVENTS**

Compared with the SARIMA baseline:

| Metric | Improvement |
|---|---:|
| Mean MAE | 4.69% |
| Mean RMSE | 17.51% |

The event-feature configuration achieved lower MAE in four of the five paired validation folds.

This is a meaningful experimental result, but it does not establish universal superiority. One validation fold showed deterioration, and the experiment covers one store-product-family series.

The results also demonstrate why adding features indiscriminately is not a sound modelling strategy: the calendar-only and combined configurations did not outperform the baseline.

### What the Results Tell Us

1. Seasonal structure matters: the weekly seasonal benchmark provides a strong reference point.
2. Model complexity must earn its place: the selected ARIMA-family model was evaluated against simpler alternatives.
3. Exogenous information can help, but its value depends on the feature and validation period.
4. More features do not necessarily produce better forecasts.
5. Fold-level evaluation reveals instability that a single holdout metric can conceal.
6. Retrospective feature availability must be separated from genuine future feature availability.

## 6. Model Diagnostics and Reliability

Forecast accuracy alone is not enough to evaluate a time-series model.

The project also investigates:

- Residual distributions.
- Residual time-series behaviour.
- Residual autocorrelation.
- ACF diagnostics.
- Squared-residual behaviour.
- Ljung–Box test results.
- Model convergence.
- Forecast error distributions.
- Fold-level forecast bias.

The intention is to avoid treating a low validation error as proof that every modelling assumption has been satisfied.

Residual diagnostics are interpreted in context, and statistical tests are not treated as automatic proof of model adequacy.

## 7. Next Development Stages

The project is progressing from forecasting toward inventory decision engineering.

### Stage 4 — Forecast Uncertainty and Inventory Policies

Planned:

- Out-of-fold forecast error modelling.
- Forecast uncertainty estimation.
- Lead-time demand distributions.
- Variable lead-time analysis.
- Safety-stock calculations.
- Reorder-point calculations.
- Inventory position.
- Service-level metrics.

A central objective is to distinguish expected demand from the uncertainty around that demand.

### Stage 5 — Constrained Replenishment Optimization

Planned:

- Replenishment quantity decision variables.
- Minimum order quantities.
- Supplier order multiples.
- Budget constraints.
- Storage capacity constraints.
- Demand coverage requirements.
- Integer optimization.
- Objective-function design.

The optimization problem will be built around explicit business constraints rather than arbitrary replenishment recommendations.

### Stage 6 — Inventory Simulation and Backtesting

Planned:

- Inventory policy simulation.
- Stockout analysis.
- Lost-demand analysis where appropriate.
- Holding costs.
- Ordering costs.
- Fill rate.
- Service level.
- Policy comparisons.

### Stage 7 — Reusable Python Engine

Planned:

- Extract stable forecasting and inventory logic into `src/`.
- Introduce automated tests.
- Separate configuration from analytical logic.
- Improve reproducibility.
- Document assumptions and model limitations.
- Consolidate the engine into a maintainable Python project.

## 8. Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data manipulation | pandas, NumPy |
| Statistical forecasting | statsmodels |
| Visualization | Matplotlib |
| Exploration | Jupyter |
| Data formats | CSV |
| Version control | Git |
| Dependency configuration | pyproject.toml, uv.lock |
| Future optimization | CVXPY and an appropriate optimization solver, subject to evaluation |

## 9. Repository Structure

```text
demand-forecasting-replenishment-engine/
├── data/
│   ├── raw/                     # Excluded from Git
│   └── processed/
├── notebooks/
│   ├── 01_data_reconnaissance.py
│   ├── 02_sarima_diagnostics.py
│   └── 03_exogenous_feature_audit.ipynb
├── src/                         # Reusable modules, planned
├── tests/                       # Automated tests, planned
├── docs/
│   └── PROJECT_STATUS.md
├── reports/
│   ├── stage_3/
│   │   ├── figures/
│   │   ├── tables/
│   │   └── stage3_exogenous_feature_analysis.md
│   └── ...                      # Earlier analytical reports
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## 10. Reproducibility and Setup

The development environment uses Python 3.12.

Clone the repository:

```bash
git clone <repository-url>
cd demand-forecasting-replenishment-engine
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies using the project configuration and lock file:

```bash
uv sync
```

Place the required raw dataset files in:

```text
data/raw/
```

Raw data is intentionally excluded from Git.

The exact dependency configuration should be checked against `pyproject.toml` and `uv.lock` when reproducing the environment.

## 11. Limitations and Research Boundaries

The current results should be interpreted within their experimental scope.

- Only one store-product-family series has been evaluated.
- The dataset represents Ecuadorian retail.
- The validation uses five chronological folds.
- The forecast horizon is 30 days.
- Event-feature results may not generalize to other stores or product families.
- Retrospective promotion features are not assumed to be available in production.
- Supplier lead times and replenishment costs have not yet been established.
- Inventory optimization and simulation are not yet completed.
- Forecast improvements do not automatically imply lower inventory costs or fewer stockouts.

These limitations define the next engineering tasks rather than being hidden behind performance claims.

## 12. Project Status

| Stage | Description | Status |
|---|---|---|
| 1 | Data engineering and demand characterization | Completed |
| 2 | ARIMA/SARIMA forecasting and validation | Completed |
| 3 | SARIMAX exogenous-feature investigation | Completed |
| 4 | Forecast uncertainty and inventory policies | completed |
| 5 | Constrained replenishment optimization | Almost |
| 6 | Inventory simulation and backtesting | Planned |
| 7 | Reusable engine and automated testing | Planned |

---

**The engineering objective:** Build a reproducible system that does not stop at predicting demand, but progressively connects statistical forecasts, uncertainty and operational constraints to defensible replenishment decisions.
