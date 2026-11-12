from collections import defaultdict
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import FinancialTransaction

ZERO = Decimal("0")
CENT = Decimal("0.01")


def list_departments(
    db: Session,
    dataset_id: int | None = None,
) -> list[str]:
    """Distinct department names (used to detect one in a question)."""
    query = select(FinancialTransaction.department).distinct()

    if dataset_id is not None:
        query = query.where(FinancialTransaction.dataset_id == dataset_id)

    return sorted(db.scalars(query).all())


def get_spending_change(
    db: Session,
    dataset_id: int | None = None,
    department: str | None = None,
) -> dict:
    """Compare average monthly spend in an earlier vs a recent period.

    The available months are split in half: the first half is the
    baseline and the second half is the recent period. With a department
    the drivers are categories; without one the drivers are departments.
    """
    filters = []

    if dataset_id is not None:
        filters.append(FinancialTransaction.dataset_id == dataset_id)

    if department is not None:
        filters.append(FinancialTransaction.department == department)

    month_expression = func.date_trunc(
        "month",
        FinancialTransaction.transaction_date,
    )

    query = (
        select(
            month_expression.label("month"),
            FinancialTransaction.department,
            FinancialTransaction.category,
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).label("total_amount"),
        )
        .where(*filters)
        .group_by(
            month_expression,
            FinancialTransaction.department,
            FinancialTransaction.category,
            FinancialTransaction.currency,
        )
        .order_by(month_expression)
    )

    rows = db.execute(query).all()

    if not rows:
        raise ValueError("No transactions found for variance analysis.")

    # Never mix currencies in one comparison.
    currencies = {row.currency for row in rows}

    if len(currencies) > 1:
        raise ValueError(
            "Variance analysis needs a single currency, but found: "
            f"{', '.join(sorted(currencies))}."
        )

    currency = next(iter(currencies))

    months = sorted({row.month.strftime("%Y-%m") for row in rows})

    if len(months) < 2:
        raise ValueError(
            "At least two months of data are required to compare periods."
        )

    split = len(months) // 2
    baseline_months = months[:split]
    recent_months = months[split:]

    driver_type = "category" if department else "department"

    by_driver: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(lambda: ZERO)
    )
    monthly_total: dict[str, Decimal] = defaultdict(lambda: ZERO)

    for row in rows:
        month = row.month.strftime("%Y-%m")
        label = row.category if department else row.department
        by_driver[label][month] += row.total_amount
        monthly_total[month] += row.total_amount

    def average(values: dict[str, Decimal], month_list: list[str]) -> Decimal:
        total = sum((values.get(m, ZERO) for m in month_list), ZERO)
        return (total / len(month_list)).quantize(CENT)

    baseline_avg = average(monthly_total, baseline_months)
    recent_avg = average(monthly_total, recent_months)
    change_amount = recent_avg - baseline_avg

    change_pct = None
    if baseline_avg != 0:
        change_pct = (change_amount / baseline_avg * 100).quantize(
            Decimal("0.1")
        )

    drivers = []
    for label, values in by_driver.items():
        driver_baseline = average(values, baseline_months)
        driver_recent = average(values, recent_months)
        drivers.append(
            {
                "name": label,
                "baseline_avg": driver_baseline,
                "recent_avg": driver_recent,
                "change_amount": driver_recent - driver_baseline,
            }
        )

    drivers.sort(key=lambda d: abs(d["change_amount"]), reverse=True)

    return {
        "department": department,
        "currency": currency,
        "driver_type": driver_type,
        "baseline_months": baseline_months,
        "recent_months": recent_months,
        "baseline_avg": baseline_avg,
        "recent_avg": recent_avg,
        "change_amount": change_amount,
        "change_pct": change_pct,
        "drivers": drivers,
        "monthly": [
            {"month": m, "total": monthly_total[m]} for m in months
        ],
    }