
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import FinancialDataset, FinancialTransaction


def get_financial_kpis(
    db: Session,
    dataset_id: int | None = None,
) -> dict:
    filters = []

    # Optionally restrict calculations to one uploaded dataset.
    if dataset_id is not None:
        dataset_exists = db.scalar(
            select(FinancialDataset.id).where(
                FinancialDataset.id == dataset_id
            )
        )

        if dataset_exists is None:
            raise ValueError("Dataset not found.")

        filters.append(
            FinancialTransaction.dataset_id == dataset_id
        )

    # Total transaction count.
    transaction_count = db.scalar(
        select(func.count(FinancialTransaction.id)).where(*filters)
    ) or 0

    # Total expenses grouped by currency.
    totals_query = (
        select(
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).label("total_amount"),
        )
        .where(*filters)
        .group_by(FinancialTransaction.currency)
        .order_by(FinancialTransaction.currency)
    )

    totals = db.execute(totals_query).all()

    # Department spending grouped by department and currency.
    department_query = (
        select(
            FinancialTransaction.department,
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).label("total_amount"),
        )
        .where(*filters)
        .group_by(
            FinancialTransaction.department,
            FinancialTransaction.currency,
        )
        .order_by(
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).desc(),
        )
    )

    departments = db.execute(department_query).all()

    # Category spending grouped by category and currency.
    category_query = (
        select(
            FinancialTransaction.category,
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).label("total_amount"),
        )
        .where(*filters)
        .group_by(
            FinancialTransaction.category,
            FinancialTransaction.currency,
        )
        .order_by(
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).desc(),
        )
    )

    categories = db.execute(category_query).all()

    # Monthly spending grouped by month and currency.
    month_expression = func.date_trunc(
        "month",
        FinancialTransaction.transaction_date,
    )

    monthly_query = (
        select(
            month_expression.label("month"),
            FinancialTransaction.currency,
            func.sum(FinancialTransaction.amount).label("total_amount"),
        )
        .where(*filters)
        .group_by(
            month_expression,
            FinancialTransaction.currency,
        )
        .order_by(
            month_expression,
            FinancialTransaction.currency,
        )
    )

    monthly = db.execute(monthly_query).all()

    return {
        "dataset_id": dataset_id,
        "transaction_count": transaction_count,
        "total_expenses": [
            {
                "currency": row.currency,
                "total_amount": row.total_amount,
            }
            for row in totals
        ],
        "department_spending": [
            {
                "department": row.department,
                "currency": row.currency,
                "total_amount": row.total_amount,
            }
            for row in departments
        ],
        "category_spending": [
            {
                "category": row.category,
                "currency": row.currency,
                "total_amount": row.total_amount,
            }
            for row in categories
        ],
        "monthly_trends": [
            {
                "month": row.month,
                "currency": row.currency,
                "total_amount": row.total_amount,
            }
            for row in monthly
        ],
    }
