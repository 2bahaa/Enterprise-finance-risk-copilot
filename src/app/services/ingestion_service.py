
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FinancialDataset, FinancialTransaction


def ingest_dataset(
    db: Session,
    filename: str,
    file_hash: str,
    transactions: list[dict],
) -> FinancialDataset:
    """
    Persist a validated financial dataset and its transactions.
    """

    # Prevent uploading the same file twice
    existing_dataset = db.scalar(
        select(FinancialDataset).where(
            FinancialDataset.file_hash == file_hash
        )
    )

    if existing_dataset:
        raise ValueError("This file has already been uploaded.")

    dataset = FinancialDataset(
        filename=filename,
        file_hash=file_hash,
        row_count=len(transactions),
        status="completed",
    )

    try:
        db.add(dataset)
        db.flush()

        transaction_records = [
            FinancialTransaction(
                dataset_id=dataset.id,
                **transaction
            )
            for transaction in transactions
        ]

        db.add_all(transaction_records)
        db.commit()
        db.refresh(dataset)

        return dataset

    except Exception:
        db.rollback()
        raise
