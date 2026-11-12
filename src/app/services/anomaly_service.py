from collections import Counter, defaultdict

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest

from app.models import FinancialDataset, FinancialTransaction


MIN_TRANSACTIONS = 5

# Rule-based check for department/category pairs that are almost never used.
# Only active on datasets large enough for "rare" to mean something.
RARE_RULE_MIN_ROWS = 100
RARE_PAIR_MAX_COUNT = 2
RARE_PAIR_MAX_SHARE = 0.01


def _add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add context-aware features that a plain amount column cannot express."""
    grouped_amount = features.groupby(["department", "category"])["amount"]

    # 1. How large is this amount compared with what is normal for the
    #    same department and category? (8x normal -> ratio of about 8)
    typical = grouped_amount.transform("median").abs()
    typical = typical.where(typical > 0, 1.0)
    features["amount_ratio"] = features["amount"] / typical

    # 2. How common is this department/category pair?
    #    A rare pair has a very small count.
    features["pair_count"] = grouped_amount.transform("count")

    # 3. Weekend flag (Saturday = 5, Sunday = 6).
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    return features


def detect_financial_anomalies(
    db: Session,
    dataset_id: int | None = None,
    contamination: float | str = "auto",
    engineered_features: bool = True,
) -> dict:
    filters = []

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

    transactions = db.scalars(
        select(FinancialTransaction)
        .where(*filters)
        .order_by(FinancialTransaction.transaction_date)
    ).all()

    if not transactions:
        return {
            "dataset_id": dataset_id,
            "analyzed_transactions": 0,
            "anomaly_count": 0,
            "anomalies": [],
        }

    # Separate currencies: never compare USD amounts with EGP amounts.
    currency_groups = defaultdict(list)

    for transaction in transactions:
        currency_groups[transaction.currency].append(transaction)

    results = []

    for currency, records in currency_groups.items():
        if len(records) < MIN_TRANSACTIONS:
            raise ValueError(
                f"Currency {currency} has only {len(records)} "
                f"transactions. At least {MIN_TRANSACTIONS} "
                "are required for anomaly detection."
            )

        # Build numerical and categorical features.
        feature_data = []

        for transaction in records:
            feature_data.append({
                "amount": float(transaction.amount),
                "day_of_week": transaction.transaction_date.weekday(),
                "day_of_month": transaction.transaction_date.day,
                "month": transaction.transaction_date.month,
                "department": transaction.department,
                "category": transaction.category,
            })

        features = pd.DataFrame(feature_data)

        # Context features must be computed BEFORE one-hot encoding,
        # because they need the department and category columns.
        if engineered_features:
            features = _add_engineered_features(features)

        # Convert categorical fields into numerical features.
        features = pd.get_dummies(
            features,
            columns=["department", "category"],
            dtype=int,
        )

        # Train the unsupervised anomaly detection model.
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )

        predictions = model.fit_predict(features)

        # Higher anomaly_score means more unusual.
        scores = -model.decision_function(features)

        # Rule: department/category pairs that almost never occur.
        pair_counts = Counter(
            (t.department, t.category) for t in records
        )
        rare_rule_active = (
            engineered_features and len(records) >= RARE_RULE_MIN_ROWS
        )

        for transaction, prediction, score in zip(
            records,
            predictions,
            scores,
        ):
            pair_count = pair_counts[
                (transaction.department, transaction.category)
            ]

            is_rare_pair = (
                rare_rule_active
                and pair_count <= RARE_PAIR_MAX_COUNT
                and pair_count / len(records) <= RARE_PAIR_MAX_SHARE
            )
            is_model_outlier = prediction == -1

            reasons = []
            if is_model_outlier:
                reasons.append("statistical outlier")
            if is_rare_pair:
                reasons.append("rare department/category combination")

            is_anomaly = is_model_outlier or is_rare_pair

            results.append({
                "transaction_id": transaction.transaction_id,
                "transaction_date": transaction.transaction_date,
                "department": transaction.department,
                "category": transaction.category,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "anomaly_score": round(float(score), 6),
                "is_anomaly": bool(is_anomaly),
                "classification": (
                    "Anomaly" if is_anomaly else "Normal"
                ),
                "reasons": reasons,
            })

    # Display the most unusual transactions first.
    results.sort(
        key=lambda item: item["anomaly_score"],
        reverse=True,
    )

    return {
        "dataset_id": dataset_id,
        "analyzed_transactions": len(results),
        "anomaly_count": sum(
            item["is_anomaly"] for item in results
        ),
        "anomalies": results,
    }