import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from replenishment.data_loader import load_daily_sales
from replenishment.inventory import calculate_inventory_policy


DEFAULT_DATA = "data/processed/store1_grocery1_daily_prepared.csv"
DEFAULT_OUTPUT = "reports/automated/inventory_policy.json"


def run_pipeline(
    data_path: str,
    output_path: str,
    lead_time_days: int,
    service_level: float,
    christmas_missing_sales_policy: str = "christmas-zero",
) -> dict:

    demand = load_daily_sales(
        data_path,
        christmas_missing_sales_policy=christmas_missing_sales_policy,
    )

    data_quality = demand.attrs.get("data_quality", {})

    policy = calculate_inventory_policy(
        historical_demand=demand,
        lead_time_days=lead_time_days,
        service_level=service_level,
    )

    report = {
        "project": "Retail Demand Forecasting and Replenishment Engine",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "observations": len(demand),
        "start_date": str(demand.index.min().date()),
        "end_date": str(demand.index.max().date()),
        "data_quality": data_quality,
        "inventory_policy": policy,
        "methodology": {
            "demand_estimator": "Historical daily mean",
            "uncertainty_estimator": "Historical daily sample standard deviation",
            "safety_stock_method": (
                "Normal approximation with square-root lead-time scaling"
            ),
            "forecast_model": "Not yet integrated",
        },
        "limitations": [
            "Christmas Day missing sales are treated as zero under an explicit working assumption.",
            "The Christmas zero-sales assumption is not a verified business fact.",
            "The policy uses historical demand rather than SARIMA/SARIMAX forecast distributions.",
            "It assumes fixed lead time and approximately independent daily demand.",
            "It does not account for order costs, holding costs or capacity constraints.",
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPIPELINE COMPLETED")
    print("=" * 55)
    print(f"Observations: {len(demand)}")
    print(f"Period: {demand.index.min().date()} to {demand.index.max().date()}")
    print(
        "Christmas observations imputed:",
        len(data_quality.get("imputed_dates", [])),
    )
    print(f"Lead-time demand: {policy['expected_lead_time_demand']:.2f}")
    print(f"Safety stock: {policy['safety_stock']:.2f}")
    print(f"Reorder point: {policy['reorder_point']:.2f}")
    print(f"Report: {output}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Retail replenishment baseline pipeline"
    )

    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--lead-time-days", type=int, default=7)
    parser.add_argument("--service-level", type=float, default=0.95)

    parser.add_argument(
        "--christmas-missing-sales-policy",
        choices=["christmas-zero", "reject"],
        default="christmas-zero",
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        output_path=args.output,
        lead_time_days=args.lead_time_days,
        service_level=args.service_level,
        christmas_missing_sales_policy=(
            args.christmas_missing_sales_policy
        ),
    )


if __name__ == "__main__":
    main()
