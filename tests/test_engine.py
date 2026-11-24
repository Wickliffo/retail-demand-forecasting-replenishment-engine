import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replenishment.data_loader import (
    DataValidationError,
    load_daily_sales,
)
from replenishment.inventory import calculate_inventory_policy


class DataLoaderTests(unittest.TestCase):

    def test_load_valid_daily_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"

            pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=3),
                "sales": [10, 20, 30],
            }).to_csv(path, index=False)

            result = load_daily_sales(path)

            self.assertEqual(len(result), 3)
            self.assertEqual(result.mean(), 20)

    def test_reject_unexpected_missing_sales(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"

            pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=3),
                "sales": [10, None, 30],
            }).to_csv(path, index=False)

            with self.assertRaises(DataValidationError):
                load_daily_sales(path)

    def test_reject_missing_calendar_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"

            pd.DataFrame({
                "date": ["2025-01-01", "2025-01-03"],
                "sales": [10, 30],
            }).to_csv(path, index=False)

            with self.assertRaises(DataValidationError):
                load_daily_sales(path)

    def test_christmas_missing_sales_are_explicitly_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"

            pd.DataFrame({
                "date": [
                    "2013-12-24",
                    "2013-12-25",
                    "2013-12-26",
                ],
                "sales": [100, None, 120],
                "is_missing_sales": [False, True, False],
            }).to_csv(path, index=False)

            result = load_daily_sales(path)

            self.assertEqual(result.loc["2013-12-25"], 0.0)

            audit = result.attrs["data_quality"]

            self.assertEqual(
                audit["imputed_dates"],
                ["2013-12-25"],
            )

            self.assertEqual(
                audit["original_missing_sales_count"],
                1,
            )

    def test_reject_christmas_imputation_when_policy_is_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.csv"

            pd.DataFrame({
                "date": ["2013-12-25"],
                "sales": [None],
            }).to_csv(path, index=False)

            with self.assertRaises(DataValidationError):
                load_daily_sales(
                    path,
                    christmas_missing_sales_policy="reject",
                )


class InventoryTests(unittest.TestCase):

    def test_inventory_policy_calculation(self):
        demand = pd.Series([10.0, 20.0, 30.0, 40.0])

        result = calculate_inventory_policy(
            historical_demand=demand,
            lead_time_days=7,
            service_level=0.95,
        )

        self.assertAlmostEqual(
            result["average_daily_demand"],
            25.0,
        )

        self.assertAlmostEqual(
            result["expected_lead_time_demand"],
            175.0,
        )

        self.assertGreater(result["safety_stock"], 0)

        self.assertGreater(
            result["reorder_point"],
            result["expected_lead_time_demand"],
        )

    def test_reject_invalid_lead_time(self):
        with self.assertRaises(ValueError):
            calculate_inventory_policy(
                pd.Series([10.0, 20.0]),
                lead_time_days=0,
            )

    def test_reject_invalid_service_level(self):
        with self.assertRaises(ValueError):
            calculate_inventory_policy(
                pd.Series([10.0, 20.0]),
                lead_time_days=7,
                service_level=1.0,
            )

class ForecastingTests(unittest.TestCase):

    def test_reject_non_datetime_index(self):
        from replenishment.forecasting import validate_demand_series

        demand = pd.Series([10.0] * 120)

        with self.assertRaises(TypeError):
            validate_demand_series(demand)

    def test_reject_incomplete_daily_calendar(self):
        from replenishment.forecasting import validate_demand_series

        dates = pd.date_range(
            "2025-01-01",
            periods=120,
            freq="D",
        ).delete(50)

        demand = pd.Series(
            [10.0] * len(dates),
            index=dates,
        )

        with self.assertRaises(ValueError):
            validate_demand_series(demand)

    def test_build_original_five_validation_folds(self):
        from replenishment.forecasting import build_validation_folds

        dates = pd.date_range(
            "2013-01-01",
            "2017-08-15",
            freq="D",
        )

        demand = pd.Series(
            [100.0] * len(dates),
            index=dates,
        )

        folds = build_validation_folds(demand)

        self.assertEqual(len(folds), 5)

        for fold in folds:
            self.assertEqual(
                len(fold["validation"]),
                30,
            )

            self.assertLess(
                fold["training"].index.max(),
                fold["validation"].index.min(),
            )

    def test_forecast_output_dates_and_bounds(self):
        from types import SimpleNamespace

        from replenishment.forecasting import (
            ModelFit,
            forecast_sarima,
        )

        class FakeForecastResult:

            predicted_mean = pd.Series(
                [100.0, 110.0, 120.0]
            )

            def conf_int(self, alpha=0.05):
                return pd.DataFrame({
                    "lower": [80.0, 90.0, 100.0],
                    "upper": [120.0, 130.0, 140.0],
                })

        class FakeFittedModel:

            def get_forecast(self, steps):
                return FakeForecastResult()

        fitted = ModelFit(
            fitted_model=FakeFittedModel(),
            converged=True,
            fit_warnings=[],
            model_name="SARIMA_111_011_7",
            training_observations=100,
            training_start="2025-01-01",
            training_end="2025-04-10",
            fit_iterations=20,
        )

        result = forecast_sarima(
            fitted,
            steps=3,
        )

        self.assertEqual(len(result), 3)

        self.assertEqual(
            str(result.index[0].date()),
            "2025-04-11",
        )

        self.assertTrue(
            (result["lower_bound"] <= result["forecast"]).all()
        )

        self.assertTrue(
            (result["forecast"] <= result["upper_bound"]).all()
        )

    def test_reject_invalid_forecast_horizon(self):
        from replenishment.forecasting import (
            ModelFit,
            forecast_sarima,
        )

        fitted = ModelFit(
            fitted_model=None,
            converged=True,
            fit_warnings=[],
            model_name="SARIMA_111_011_7",
            training_observations=100,
            training_start="2025-01-01",
            training_end="2025-04-10",
            fit_iterations=20,
        )

        with self.assertRaises(ValueError):
            forecast_sarima(fitted, steps=0)
if __name__ == "__main__":
    unittest.main()
