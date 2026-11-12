"""Generate a reproducible synthetic finance dataset with injected anomalies.

Outputs (in the data/ folder):
  synthetic_transactions.csv   -> upload this through POST /financial/datasets
  synthetic_ground_truth.csv   -> the answer key (which rows are anomalies)
"""

import csv
import random
from calendar import monthrange
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

SEED = 42
YEAR = 2026
MONTHS = [5, 6, 7, 8, 9, 10]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRANSACTIONS_FILE = DATA_DIR / "synthetic_transactions.csv"
GROUND_TRUTH_FILE = DATA_DIR / "synthetic_ground_truth.csv"

# (department, category): (mean amount, standard deviation, transactions per month)
PROFILE = {
    ("Finance", "Software"): (1200, 300, 5),
    ("Finance", "Audit Fees"): (3500, 600, 2),
    ("Finance", "Travel"): (900, 250, 6),
    ("Finance", "Office Supplies"): (200, 60, 8),
    ("HR", "Recruitment"): (2200, 500, 5),
    ("HR", "Training"): (1500, 350, 4),
    ("HR", "Travel"): (850, 220, 3),
    ("IT", "Hardware"): (1800, 450, 5),
    ("IT", "Software Licenses"): (1400, 300, 5),
    ("IT", "Cloud Services"): (1600, 250, 3),
    ("IT", "Support Services"): (900, 200, 5),
    ("Marketing", "Advertising"): (3000, 700, 6),
    ("Marketing", "Events"): (4000, 900, 3),
    ("Marketing", "Software"): (800, 200, 4),
    ("Operations", "Logistics"): (2800, 600, 7),
    ("Operations", "Maintenance"): (1300, 300, 5),
    ("Operations", "Office Supplies"): (350, 90, 6),
    ("Legal", "Legal Fees"): (5000, 1000, 3),
    ("Legal", "Compliance Training"): (900, 200, 3),
}

# THE STORY: from August, IT cloud spending jumps.
CLOUD_KEY = ("IT", "Cloud Services")
CLOUD_SPIKE_FROM_MONTH = 8
CLOUD_SPIKE_PROFILE = (2200, 300, 9)

# The staged break-ins: (type, department, category, month)
ANOMALY_PLAN = [
    ("amount_spike", "Marketing", "Advertising", 8),
    ("amount_spike", "Finance", "Software", 7),
    ("amount_spike", "IT", "Hardware", 9),
    ("amount_spike", "Operations", "Maintenance", 6),
    ("amount_spike", "HR", "Recruitment", 10),
    ("weekend_payment", "Finance", "Audit Fees", 5),
    ("weekend_payment", "Legal", "Legal Fees", 7),
    ("weekend_payment", "IT", "Cloud Services", 9),
    ("weekend_payment", "Operations", "Logistics", 10),
    ("rare_combination", "HR", "Cloud Services", 8),
    ("rare_combination", "Legal", "Advertising", 9),
    ("rare_combination", "Operations", "Recruitment", 6),
]


def days_in_month(month: int, weekend: bool) -> list[date]:
    last_day = monthrange(YEAR, month)[1]
    all_days = [date(YEAR, month, d) for d in range(1, last_day + 1)]
    return [d for d in all_days if (d.weekday() >= 5) == weekend]


def monthly_profile(key: tuple[str, str], month: int) -> tuple[float, float, int]:
    if key == CLOUD_KEY and month >= CLOUD_SPIKE_FROM_MONTH:
        return CLOUD_SPIKE_PROFILE
    return PROFILE[key]


def make_row(department, category, day, amount, anomaly_type="", note=""):
    return {
        "transaction_date": day,
        "department": department,
        "category": category,
        "amount": round(amount, 2),
        "currency": "USD",
        "description": f"{category} payment ({department})",
        "anomaly_type": anomaly_type,
        "note": note,
    }


def generate_normal(rng: random.Random) -> list[dict]:
    rows = []

    for month in MONTHS:
        weekdays = days_in_month(month, weekend=False)

        for key in PROFILE:
            mean, std, count = monthly_profile(key, month)

            for _ in range(count):
                amount = max(20.0, rng.gauss(mean, std))
                rows.append(
                    make_row(key[0], key[1], rng.choice(weekdays), amount)
                )

    return rows


def generate_anomalies(rng: random.Random) -> list[dict]:
    rows = []

    for kind, department, category, month in ANOMALY_PLAN:
        if kind == "amount_spike":
            mean = PROFILE[(department, category)][0]
            amount = mean * rng.uniform(8, 10)
            day = rng.choice(days_in_month(month, weekend=False))
            note = "Amount 8-10x the usual level for this department and category"

        elif kind == "weekend_payment":
            mean = PROFILE[(department, category)][0]
            amount = mean * rng.uniform(4, 5)
            day = rng.choice(days_in_month(month, weekend=True))
            note = "Large payment dated on a weekend"

        else:  # rare_combination
            amount = rng.uniform(3000, 6000)
            day = rng.choice(days_in_month(month, weekend=False))
            note = "Department/category combination never seen in normal data"

        rows.append(make_row(department, category, day, amount, kind, note))

    return rows


def write_files(rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    with TRANSACTIONS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "transaction_id", "transaction_date", "department",
                "category", "amount", "currency", "description",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["transaction_id"],
                    row["transaction_date"].isoformat(),
                    row["department"],
                    row["category"],
                    f"{row['amount']:.2f}",
                    row["currency"],
                    row["description"],
                ]
            )

    with GROUND_TRUTH_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "transaction_id", "anomaly_type", "transaction_date",
                "department", "category", "amount", "note",
            ]
        )
        for row in rows:
            if row["anomaly_type"]:
                writer.writerow(
                    [
                        row["transaction_id"],
                        row["anomaly_type"],
                        row["transaction_date"].isoformat(),
                        row["department"],
                        row["category"],
                        f"{row['amount']:.2f}",
                        row["note"],
                    ]
                )


def print_summary(rows: list[dict]) -> None:
    def money(value: float) -> Decimal:
        return Decimal(f"{value:.2f}")

    total = sum(money(r["amount"]) for r in rows)
    it_by_month = defaultdict(Decimal)
    cloud_by_month = defaultdict(Decimal)

    for r in rows:
        if r["department"] == "IT":
            it_by_month[r["transaction_date"].month] += money(r["amount"])
            if r["category"] == "Cloud Services":
                cloud_by_month[r["transaction_date"].month] += money(r["amount"])

    print(f"Rows written:         {len(rows)}")
    print(f"Injected anomalies:   {sum(1 for r in rows if r['anomaly_type'])}")
    print(f"Total spend (USD):    {total:,.2f}")
    print("\nIT spending by month (includes any IT anomalies):")
    for month in MONTHS:
        print(f"  2026-{month:02d}: {it_by_month[month]:>12,.2f}"
              f"   (cloud: {cloud_by_month[month]:>10,.2f})")
    print(f"\nFiles:\n  {TRANSACTIONS_FILE}\n  {GROUND_TRUTH_FILE}")


def main() -> None:
    rng = random.Random(SEED)

    normal = generate_normal(rng)
    anomalies = generate_anomalies(rng)

    # Sanity check: normal data must never fall on a weekend.
    assert all(r["transaction_date"].weekday() < 5 for r in normal)

    rows = normal + anomalies
    rows.sort(
        key=lambda r: (
            r["transaction_date"], r["department"], r["category"], r["amount"]
        )
    )

    for index, row in enumerate(rows, start=1):
        row["transaction_id"] = f"SYN-{index:04d}"

    write_files(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()