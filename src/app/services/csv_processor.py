
import hashlib
import re

from datetime import date
from decimal import Decimal, InvalidOperation
from io import BytesIO

import pandas as pd


# Required and optional CSV columns
REQUIRED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "department",
    "category",
    "amount",
    "currency",
}

OPTIONAL_COLUMNS = {"description"}


def calculate_file_hash(file_content: bytes) -> str:
    """
    Generate a SHA-256 fingerprint for the uploaded CSV file.
    Used to identify duplicate uploads.
    """
    return hashlib.sha256(file_content).hexdigest()


def process_csv(file_content: bytes) -> tuple[str, list[dict]]:
    """
    Parse and validate financial transactions from CSV bytes.

    Returns:
        A tuple containing:
        - SHA-256 file hash
        - List of validated transaction dictionaries

    Raises:
        ValueError: If the CSV is empty, malformed, or contains
                    invalid financial data.
    """

    # 1. Check whether the uploaded file is empty
    if not file_content:
        raise ValueError("Uploaded CSV file is empty.")

    # 2. Calculate file fingerprint
    file_hash = calculate_file_hash(file_content)

    # 3. Read CSV using Pandas
    try:
        df = pd.read_csv(
            BytesIO(file_content),
            dtype=str,
            keep_default_na=False
        )

    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError
    ) as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc

        # 4. Normalize column names
    columns = [str(col).strip() for col in df.columns]

    # 5. Check for duplicate column names
    if len(columns) != len(set(columns)):
        raise ValueError("CSV contains duplicate column names.")

    df.columns = pd.Index(columns)

    # 6. Validate required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 7. Check whether the CSV contains transactions
    if df.empty:
        raise ValueError("CSV contains no transactions.")

    # 8. Process individual transactions
    transactions = []
    seen_ids = set()

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=2
    ):

        # Extract and normalize values
        transaction_id = row["transaction_id"].strip()
        department = row["department"].strip()
        category = row["category"].strip()
        currency = row["currency"].strip().upper()

        # Validate required text fields
        if not transaction_id or not department or not category:
            raise ValueError(
                f"Row {row_number}: Required text fields cannot be empty."
            )

        # Validate duplicate transaction IDs within the CSV
        if transaction_id in seen_ids:
            raise ValueError(
                f"Row {row_number}: Duplicate transaction_id "
                f"'{transaction_id}'."
            )

        seen_ids.add(transaction_id)

        # Validate transaction date
        try:
            transaction_date = date.fromisoformat(
                row["transaction_date"].strip()
            )

        except ValueError as exc:
            raise ValueError(
                f"Row {row_number}: Invalid transaction_date. "
                "Expected YYYY-MM-DD."
            ) from exc

        # Validate financial amount
        try:
            amount = Decimal(row["amount"].strip())

            # Reject NaN and Infinity
            if not amount.is_finite():
                raise ValueError("Amount must be finite.")

            # Reject amounts with more than two decimal places
            exponent = amount.as_tuple().exponent

            if not isinstance(exponent, int) or exponent < -2:
                raise ValueError(
                    "Amount cannot have more than 2 decimals."
                )

        except (InvalidOperation, ValueError) as exc:
            raise ValueError(
                f"Row {row_number}: Invalid amount."
            ) from exc

        # Validate currency format
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError(
                f"Row {row_number}: Invalid currency code."
            )

        # Build normalized transaction dictionary
        transactions.append({
            "transaction_id": transaction_id,
            "transaction_date": transaction_date,
            "department": department,
            "category": category,
            "amount": amount,
            "currency": currency,
            "description": (
                row["description"].strip()
                if "description" in df.columns
                else None
            ),
        })

    # 9. Return file hash and validated transactions
    return file_hash, transactions
